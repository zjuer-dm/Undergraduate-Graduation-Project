import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens_new_30
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import cv2
import copy
from collections import OrderedDict
import hashlib
import pickle

@baseline_registry.register_trainer(name="PPO-30-no-drop-env-efficient")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.GRPO_ORM.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0
        # GRPO specific hyperparameters
        self.grpo_epsilon = config.GRPO_ORM.grpo_epsilon  # Clipping for PPO objective

        self.gamma = config.PPO.gamma
        self.gae_lambda = config.PPO.gae_lambda
        self.value_loss_coef = config.PPO.value_loss_coef
        self.entropy_coef = config.PPO.entropy_coef

        # self.grpo_beta = config.GRPO_ORM.grpo_beta        # Coefficient for KL divergence loss
        # if self.grpo_beta < 1e-6:
        #     self.need_ref_policy = False
        # else:
        #     self.need_ref_policy = True
        self.max_grad_norm = config.GRPO_ORM.max_grad_norm
        # Storing the initial number of environments for reward grouping
        self.initial_num_envs = config.NUM_ENVIRONMENTS # Make sure this is set in your config, typically the overall batch size for rollouts
        self.grpo_update_epochs = config.GRPO_ORM.update_epochs
        # If NUM_ENVIRONMENTS is not available, you might need to pass it or set it based on initial self.envs.num_envs
        # For now, I'll assume self.batch_size (set in _set_config) holds this value if NUM_ENVIRONMENTS is not directly in config
        # self.initial_num_envs = self.config.GRPO_ORM.batch_size # As per your _set_config
        self.enable_amp = config.GRPO_ORM.enable_amp
        self.enable_all_dropouts = config.GRPO_ORM.enable_all_dropouts
        self.dropout_in_sampling = config.GRPO_ORM.dropout_in_sampling
        self.dropout_rate = config.GRPO_ORM.dropout_rate
        self.scaler = GradScaler(enabled=self.enable_amp)
        # print("config.GRPO_ORM:\n", config.GRPO_ORM)
        # print(f"GRPO params: grpo_epsilon {self.grpo_epsilon}, grpo_beta {self.grpo_beta}, max_grad_norm {self.max_grad_norm}, grpo_update_epochs {self.grpo_update_epochs} \
        #       enable_amp {self.enable_amp}, need_ref_policy {self.need_ref_policy}, enable_all_dropouts {self.enable_all_dropouts}, dropout_rate {self.dropout_rate}, dropout_in_sampling {self.dropout_in_sampling}")
        print(f"PPO params: clip_epsilon {self.grpo_epsilon}, gamma {self.gamma}, gae_lambda {self.gae_lambda}")
        print(f"Loss coeffs: value {self.value_loss_coef}, entropy {self.entropy_coef}")
    
    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        if self.config.ONLY_LAST_SAVEALL and (not iteration == self.config.GRPO_ORM.iters):
            torch.save(
                        obj={
                            "state_dict": self.policy.state_dict(), # 网络权重
                            "config": self.config, # 配置参数
                        },
                        f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
                    )
        else:
            torch.save(
                obj={
                    "state_dict": self.policy.state_dict(), # 网络权重
                    "config": self.config, # 配置参数
                    "optim_state": self.optimizer.state_dict(), # 保存时优化器的状态（可以读取后进行继续训练，而不是从头开始的训练）
                    "scheduler_state": self.scheduler.state_dict(), # 保存规划器状态
                    "iteration": iteration, # 保存时训练了几步
                },
                f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
            )
    
    def _verify_ddp_weights_synchronized(self, step_name="check"):
        """
        校验所有DDP进程中的模型权重是否完全相同。
        这是一个调试工具，用于确保分布式训练正确同步。
        """
        if self.world_size <= 1:
            # 如果不是多卡环境，则无需校验
            return

        # 1. 在当前进程（GPU）上获取所有模型参数并“压平”成一个一维张量
        # 我们只关心可训练的参数，因为这才是DDP同步的对象
        local_params = [p.data for p in self.policy.net.module.vln_bert.parameters() if p.requires_grad]
        if not local_params:
            if self.local_rank == 0:
                print(f"[{step_name}] DDP校验警告：模型中没有找到可训练的参数。")
            return
            
        with torch.no_grad():
            local_weights_flat = torch.cat([p.flatten() for p in local_params])

            # 2. 创建一个列表，用于从所有进程收集压平后的权重张量
            # all_weights_list 中每个张量的大小都应与 local_weights_flat 相同
            all_weights_list = [torch.zeros_like(local_weights_flat) for _ in range(self.world_size)]

            # 3. 使用 `all_gather` 操作
            # 这个操作会从所有进程收集 local_weights_flat，并将结果分发回所有进程
            # 执行后，每个进程的 all_weights_list 中都包含了所有其他进程的权重
            distr.all_gather(all_weights_list, local_weights_flat)

            # 4. 只在主进程（rank 0）上进行比较和打印，避免信息刷屏
            if self.local_rank == 0:
                is_synced = True
                # 从第二个进程（rank 1）开始，将其权重与主进程（rank 0）的权重进行比较
                for rank in range(1, self.world_size):
                    # torch.allclose 是比较浮点数张量的推荐方法
                    if not torch.allclose(all_weights_list[0], all_weights_list[rank]):
                        # 如果发现任何不一致，就打印错误信息并标记为未同步
                        print(
                            f"!! DDP 权重校验失败 !! 在步骤 '{step_name}': "
                            f"Rank 0 和 Rank {rank} 的权重不一致。"
                        )
                        # 计算并打印差异的范数，以了解差异程度
                        diff = torch.norm(all_weights_list[0] - all_weights_list[rank])
                        print(f"    差异的L2范数: {diff.item()}")
                        is_synced = False
                        break # 发现不一致后即可退出循环
                
                if is_synced:
                    # 如果所有进程的权重都一致，打印成功信息
                    print(
                        f"✓ DDP 权重校验成功。在步骤 '{step_name}': "
                        f"所有 {self.world_size} 个进程的权重保持一致。"
                    )
        
        # 添加一个 barrier，确保所有进程都完成了校验后才继续执行后续代码
        # 这可以防止某些进程提前进入下一个训练步骤，导致状态不一
        distr.barrier()


    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        print(f"init camera information: resize_config:{resize_config}, crop_config:{crop_config}, new_camera_heading:{camera_orientations}")
        # init camera information: resize_config:[('rgb', (224, 298)), ('depth', (256, 341))], crop_config:[('rgb', (224, 224)), ('depth', (256, 256))], new_camera:11个
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.MODEL.task_type == 'rxr':
            # self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_TRAIN', 'STEPS_TAKEN', 'COLLISIONS', 'NDTW', 'SDTW']
            self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_TRAIN', 'STEPS_TAKEN', 'COLLISIONS']
        elif self.config.MODEL.task_type == 'r2r':
            self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_TRAIN', 'STEPS_TAKEN', 'COLLISIONS']
        self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.GRPO_ORM.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank * 10 # 在这里为每个gpu进程下的env数据集提供了不同的随机性，即训练时每个gpu进程的episode循环模式是不同的
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        # 如果设定的传感器尺寸就等于CenterCropperPerSensor设定的尺寸，那么就不会做任何操作（实际上也是如此）
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def setup_training_parts(self):
        """
        设置模型的训练状态：
        1. 整体 policy.eval() 和冻结所有参数。
        2. 对 self.trainable_parts 设为 .train() 和解冻参数。
        """
        # 1. 将整个 policy 设置为评估模式并冻结所有参数
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False

        # 2. 对指定的可训练部分进行设置
        if not self.trainable_parts:
            print("Warning: No specific parts designated as trainable.")
            return

        print("Setting up specific parts for training...")
        for module_to_train in self.trainable_parts:
            # 将此子模块设置为训练模式
            # 这会覆盖全局的 self.policy.eval() 对此部分的影响
            module_to_train.train()
            print(f"  Module {type(module_to_train).__name__} set to TRAIN mode.")
            # 解冻此子模块的参数
            for param in module_to_train.parameters():
                param.requires_grad = True
            print(f"    Parameters for {type(module_to_train).__name__} UNFROZEN.")

    def set_policy_mode(self, mode):
        if not self.enable_all_dropouts:
            if not self.trainable_parts:
                return
            if mode == 'train':
                # print("Switching trainable parts to TRAIN mode.")
                for module_part in self.trainable_parts:
                    module_part.train()
            elif mode == 'eval':
                # print("Switching trainable parts to EVAL mode.")
                for module_part in self.trainable_parts:
                    module_part.eval()
            else:
                raise ValueError("Mode must be 'train' or 'eval'.")
        else:
            if mode == 'train':
                if self.world_size > 1:
                    self.policy.net.module.vln_bert.train()
                else:
                    self.policy.net.vln_bert.train()
            elif mode == 'eval':
                self.policy.eval()
            else:
                raise ValueError("Mode must be 'train' or 'eval'.")
            
    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name) # PolicyViewSelectionETP
        # 下面的action_space实际上内部并没有用到，而observation_space是用于resnetencoder确定输入尺寸
        # if not self.enable_all_dropouts:
        #     policy_dropout_rate = 0.05
        # else:
        #     policy_dropout_rate = 0.1
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            dropout_rate=self.dropout_rate,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        # 下面load的返回值是只含有一个'predictor'一个key的字典，而'predictor'内部又是一个含有['epoch', 'state_dict', 'optimizer']三个key的字典
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict']) 
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        # self.device就是当前进程对应的单个gpu编号
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers # 似乎没用

        try:
            vln_bert_module = self.policy.net.vln_bert
            self.trainable_parts = [
                vln_bert_module.global_encoder,
                vln_bert_module.graph_query_text,
                vln_bert_module.graph_attentioned_txt_embeds_transform,
                vln_bert_module.global_sap_head
            ]
            # 验证模块是否确实是 nn.Module
            for part in self.trainable_parts:
                if not isinstance(part, torch.nn.Module):
                    raise TypeError(f"Part {part} is not an nn.Module")
        except AttributeError as e:
            print(f"Error accessing specified submodules: {e}")
            print("Please ensure the paths to trainable submodules are correct.")
            self.trainable_parts = []
        self.setup_training_parts()

        if self.config.GPU_NUMBERS > 1:
            # 即使是多卡时，waypoint_predictor也不需要封装，因为一方面不涉及多卡训练，另一方面也不涉及多卡推理时的数据集自动切分
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)

        not_trainable_parameters = [p for p in self.policy.parameters() if not p.requires_grad]
        trainable_parameters = [(n, p) for n, p in self.policy.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in trainable_parameters
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in trainable_parameters
                        if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        if trainable_parameters:
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.GRPO_ORM.lr)
            print(f"Optimizer configured with {len(trainable_parameters)} trainable parameters.")
            print(f"Remaining {len(not_trainable_parameters)} untrainable parameters.")
        else:
            self.optimizer = None
            print("Warning: No parameters were set to trainable. Optimizer not configured.")
        
        num_warmup_steps = self.config.GRPO_ORM.warmup_iters
        num_training_steps = self.config.GRPO_ORM.iters
        min_lr_ratio = self.config.GRPO_ORM.min_lr_ratio
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            return decayed_lr_multiplier
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # self.scheduler.step()

        if load_from_ckpt: # infer中为True
            if config.GRPO_ORM.is_requeue: # 意思是是否需要在中断后恢复训练
                # print()
                import glob
                # ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                search_pattern = os.path.join(config.CHECKPOINT_FOLDER, "*.pth")
                ckpt_list = glob.glob(search_pattern)
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1] # 获取最新的权重文件来加载self.policy
            else:
                ckpt_path = config.GRPO_ORM.ckpt_to_load
            # 下面load的返回值是含有['state_dict', 'config', 'optim_state', 'iteration']为key的字典，对应了self.save_checkpoint保存代码
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            # start_iter = ckpt_dict["iteration"]
            if config.GRPO_ORM.is_requeue:
                start_iter = ckpt_dict["iteration"]
            else:
                start_iter = 0

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module # net是ILPolicy中的属性，在这里就代表着ETP这个class的实例
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            elif 'module' not in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS > 1:
                new_state_dict = OrderedDict()
                for k, v in ckpt_dict['state_dict'].items():
                    if k.startswith("net."):
                        name = k.replace("net.", "net.module.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                incompatible_keys = self.policy.load_state_dict(new_state_dict, strict=False)
            else:
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)

            print("\n" + "="*25 + " 权重加载不匹配报告 " + "="*25)
            if incompatible_keys.missing_keys:
                print("以下网络层在模型中存在，但在权重文件中缺失 (将使用初始值):")
                for key in sorted(incompatible_keys.missing_keys):
                    print(f"  - {key}")
            else:
                print("所有模型中存在的网络层都在权重文件中找到了。")
            if incompatible_keys.unexpected_keys:
                print("\n以下网络层在权重文件中存在，但在模型中缺失 (将被忽略):")
                for key in sorted(incompatible_keys.unexpected_keys):
                    print(f"  - {key}")
            else:
                print("\n权重文件中没有多余的网络层。")
            print("="*75 + "\n")

            if config.GRPO_ORM.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                print("optimizer is load from checkpoint")
                if "scheduler_state" in ckpt_dict:
                    self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
            else:
                print("optimizer is initialized")
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
		
        

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            # 哪些参数是不可训练的是在模型定义的时候确定了的，例如可见vlnce_baselines/models/etp/vlnbert_init.py中的GlocalTextPathNavCMT中的init函数部分
            # 默认情况下，整个网络就只有net.rgb_encoder和net.depth_encoder的权重不可训练，其他都可以训练
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")
        return start_iter

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []
        
        for i in range(self.envs.num_envs):
            # 这边的torch.cat(rgb_fts, dim=0)尺寸为Q*512，其中Q取值可以是12，13，14等，因为obs['cand_img_idxes'][i]中的元素可能有重复，例如两个waypoints都属于同一个30度区间
            # nav_types同样也是Q的长度
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, task_type):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        batch_gmap_task_embeddings = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            gmap_step_ids_tensor = torch.LongTensor(gmap_step_ids)
            batch_gmap_step_ids.append(gmap_step_ids_tensor)
            batch_gmap_task_embeddings.append(torch.full_like(gmap_step_ids_tensor, task_type))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_task_embeddings = pad_sequence(batch_gmap_task_embeddings, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left, 'gmap_task_embeddings': batch_gmap_task_embeddings
        }

    def train(self):
        # train的数据集就直接在r2r_vlnce.yaml里设置
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        # 下面返回的start_iter值的是从第几个iteration开始训练(如果是重新开始,那就是0,如果是load之前训练的权重,那就是对应权重名字中的数字)（一个iteration就是一次rollout）

        start_iter = self._initialize_policy(
            self.config,
            self.config.GRPO_ORM.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.GRPO_ORM.iters # default: 15000
        log_every  = self.config.GRPO_ORM.log_every # default: 200
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        if self.config.local_rank < 1:
            config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
            with open(config_path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Configuration saved to {config_path}")

        logger.info('Traning Starts... GOOD LUCK!')
        if self.world_size > 1:
            self._verify_ddp_weights_synchronized()
        
        self.data_buffer = []
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0)) # interval表示当前这次循环需要进行几次iteration
            cur_iter = idx + interval # cur_iter表示当前这次循环完成后一共进行了几次iteration, idx表示当前这次循环完成前一共进行了几次iteration

            # sample_ratio = self.config.GRPO_ORM.sample_ratio ** (idx // self.config.GRPO_ORM.decay_interval)
            logs = self._train_interval(interval)

            if self.world_size > 1:
                self._verify_ddp_weights_synchronized()

            final_logs = {}
            if self.world_size > 1:
                # 步骤 1: 收集所有进程上的所有日志键
                local_keys = list(logs.keys())
                all_keys_list = [None] * self.world_size
                torch.distributed.all_gather_object(all_keys_list, local_keys)

                # 步骤 2: 在 rank 0 上创建所有键的唯一集合，并广播
                if self.local_rank == 0:
                    # 使用 set 来自动去重
                    all_unique_keys = sorted(list(set(key for sublist in all_keys_list for key in sublist)))
                else:
                    all_unique_keys = None
                
                # 使用 broadcast_object_list 将统一的键列表从 rank 0 发送到所有其他进程
                # 这确保所有进程都将以相同的顺序迭代相同的键
                obj_list = [all_unique_keys]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                synced_keys = obj_list[0]

                # 步骤 3: 迭代这个同步过的键列表
                for k in synced_keys:
                    # 如果当前进程有这个键，就计算它的值；否则，值为0
                    if k in logs and logs[k]:
                        local_sum = sum(logs[k])
                        local_count = len(logs[k])
                    else:
                        local_sum = 0.0
                        local_count = 0
                    
                    metric_tensor = torch.tensor([local_sum, local_count], dtype=torch.float64, device=self.device)
                    # 现在所有进程都会为同一个键 k 调用 all_reduce，不会再有分歧
                    distr.all_reduce(metric_tensor, op=distr.ReduceOp.SUM)
                    
                    if self.local_rank == 0:
                        global_sum, global_count = metric_tensor[0].item(), metric_tensor[1].item()
                        final_logs[k] = global_sum / global_count if global_count > 0 else 0.0
            else: # 单卡环境逻辑保持不变
                for k, v_list in logs.items():
                    if not v_list: continue
                    final_logs[k] = np.mean(v_list)

            # -- 4. 只有 rank 0 进程执行打印和写入Tensorboard的操作 --
            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, avg_val in final_logs.items():
                    loss_str += f'{k}: {avg_val:.3f}, '
                    writer.add_scalar(f'grpo/{k}', avg_val, cur_iter)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', current_lr, cur_iter)
                logger.info(loss_str)
                logger.info(f"lr: {current_lr}")
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval):
        # self.policy.train()
        # self.set_policy_mode("train")
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            # 下面是进行了一次采样和训练
            # self.optimizer.zero_grad()
            # self.loss = 0. # 每次rollout重置一下self.loss
            with torch.no_grad():
                with autocast(enabled=self.enable_amp):
                    self.sample_data(self.config.GRPO_ORM.sample_num)
            self.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})

        return deepcopy(self.logs)
    


    # =====================================================================================
# =================== 请将此完整函数粘贴到 PPO_30_no_drop_env_efficient.py 中 ===================
# =====================================================================================

    def update(self):
        """
        PPO update function. This function replaces the original GRPO logic.
        It computes advantages using GAE and updates the policy using the clipped
        surrogate objective, value loss, and an entropy bonus.
        """
        if not self.data_buffer:
            logger.info("Data buffer is empty. Skipping update.")
            return

        self.set_policy_mode("train")  # Switch trainable parts to train mode

        # --- 1. Data Preparation and GAE Calculation (FINAL, ROBUST VERSION) ---
        gamma = self.config.PPO.gamma
        gae_lambda = self.config.PPO.gae_lambda
        value_loss_coef = self.config.PPO.value_loss_coef
        entropy_coef = self.config.PPO.entropy_coef
        
        all_advantages = []
        all_value_targets = []
        all_old_values = [] # --- (MODIFIED) --- 新增：用于存储采样时的价值，为价值裁剪做准备

        for sample_batch in self.data_buffer:
            # --- Prepare padded tensors to handle variable-length trajectories ---
            max_steps = len(sample_batch['data_buffer'])
            num_envs = self.initial_num_envs
            
            padded_rewards = torch.zeros(max_steps, num_envs, dtype=torch.float32)
            padded_values = torch.zeros(max_steps, num_envs, dtype=torch.float32)
            padded_dones = torch.ones(max_steps, num_envs, dtype=torch.bool)
            mask = torch.zeros(max_steps, num_envs, dtype=torch.bool)

            for t, step_data in enumerate(sample_batch['data_buffer']):
                indices = step_data['indices']
                padded_rewards[t, indices] = step_data['reward']
                padded_values[t, indices] = step_data['value'].squeeze() 
                padded_dones[t, indices] = step_data['done']
                mask[t, indices] = True
            
            # --- Perform GAE calculation on the padded tensors ---
            gae = torch.zeros(num_envs, dtype=torch.float32)
            advantages_batch = torch.zeros(max_steps, num_envs, dtype=torch.float32)

            for t in reversed(range(max_steps)):
                next_value = padded_values[t+1] if t < max_steps - 1 else torch.zeros(num_envs)
                next_value_masked = next_value * (1.0 - padded_dones[t].float())
                delta = padded_rewards[t] + gamma * next_value_masked - padded_values[t]
                gae = delta + gamma * gae_lambda * gae * (1.0 - padded_dones[t].float())
                advantages_batch[t] = gae

            value_targets_batch = advantages_batch + padded_values

            # --- (MODIFIED) --- 收集所有有效步骤的优势、价值目标和旧价值
            all_advantages.append(advantages_batch[mask])
            all_value_targets.append(value_targets_batch[mask])
            all_old_values.append(padded_values[mask]) # <-- 收集旧价值

        # --- (MODIFIED) --- 将所有批次的数据合并成一个大张量
        all_advantages = torch.cat(all_advantages).to(self.device)
        all_value_targets = torch.cat(all_value_targets).to(self.device)
        all_old_values = torch.cat(all_old_values).to(self.device) # <-- 合并旧价值
        
        # Normalizing advantages is a key trick for PPO stability
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # --- 2. Multi-epoch training loop over the collected data ---
        for epoch in range(self.grpo_update_epochs):
            step_counter = 0
            for sample_batch in self.data_buffer:
                initial_txt_embeds_cuda = sample_batch["initial_txt_embeds"].to(self.device, non_blocking=True)
                initial_txt_masks_cuda = sample_batch["initial_txt_masks"].to(self.device, non_blocking=True)
                
                for step_data in sample_batch["data_buffer"]:
                    self.optimizer.zero_grad()
                    active_indices = step_data['indices']
                    if not active_indices: continue
                    
                    batch_size = len(active_indices)
                    
                    # --- (MODIFIED) --- 从大张量中切分出当前批次需要的数据
                    step_advantages = all_advantages[step_counter : step_counter + batch_size]
                    step_value_targets = all_value_targets[step_counter : step_counter + batch_size]
                    step_old_values = all_old_values[step_counter : step_counter + batch_size] # <-- 切分旧价值
                    step_counter += batch_size

                    nav_inputs_cpu = step_data["input"]
                    taken_actions_cpu = step_data["action"]
                    old_probs_at_sampling_cpu = step_data["probs"]
                    
                    nav_inputs_cuda = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in nav_inputs_cpu.items()}
                    nav_inputs_cuda['txt_embeds'] = initial_txt_embeds_cuda[active_indices]
                    nav_inputs_cuda['txt_masks'] = initial_txt_masks_cuda[active_indices]
                    nav_inputs_cuda['mode'] = 'navigation'
                    taken_actions_cuda = taken_actions_cpu.to(self.device)
                    
                    old_log_probs_taken_action = torch.log(
                        old_probs_at_sampling_cpu.to(self.device).gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1) + 1e-9
                    )

                    with autocast(enabled=self.enable_amp):
                        current_policy_outputs = self.policy.net(**nav_inputs_cuda)
                        current_logits = current_policy_outputs['global_logits']
                        current_values = current_policy_outputs['value'].squeeze(-1)

                        dist = torch.distributions.Categorical(logits=current_logits)
                        current_log_probs_taken_action = dist.log_prob(taken_actions_cuda)
                        dist_entropy = dist.entropy().mean()

                        # --- 策略损失 (Policy Loss) - (不变) ---
                        ratio = torch.exp(current_log_probs_taken_action - old_log_probs_taken_action)
                        surr1 = ratio * step_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.grpo_epsilon, 1.0 + self.grpo_epsilon) * step_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # --- (MODIFIED) --- 价值损失 (Value Loss) - 加入价值裁剪逻辑
                        # 1. 裁剪新的价值预测，使其不会离旧的预测太远
                        v_clipped = step_old_values + torch.clamp(
                            current_values - step_old_values,
                            -self.grpo_epsilon,  # 使用和策略裁剪相同的 epsilon
                            self.grpo_epsilon,
                        )
                        
                        # 2. 计算未裁剪(v_loss_unclipped)和裁剪后(v_loss_clipped)的两种价值损失
                        v_loss_unclipped = F.mse_loss(current_values, step_value_targets, reduction='none')
                        v_loss_clipped = F.mse_loss(v_clipped, step_value_targets, reduction='none')
                        
                        # 3. 取两者中较大的那个作为最终的价值损失, 并乘以系数0.5
                        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        
                        # --- 总损失 (Total Loss) ---
                        total_loss = (policy_loss +
                                    value_loss_coef * value_loss -
                                    entropy_coef * dist_entropy)

                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
                    if trainable_params:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                        self.logs['grad_norm'].append(grad_norm.item())
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.logs['policy_loss'].append(policy_loss.item())
                    self.logs['value_loss'].append(value_loss.item())
                    self.logs['entropy'].append(dist_entropy.item())
                    self.logs['total_loss'].append(total_loss.item())

        # --- 5. Cleanup ---
        self.data_buffer.clear()
        self.scheduler.step()


    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def copy_nav_inputs_dict(self, input_dict):
        copied_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                copied_dict[key] = value.cpu()
            elif isinstance(value, list):
                # copied_dict[key] = [inner_list[:] for inner_list in value]
                copied_dict[key] = copy.deepcopy(value)
            elif isinstance(value, str):
                copied_dict[key] = value
            else:
                copied_dict[key] = copy.deepcopy(value) # Fallback for other types
        return copied_dict

    def sample_data(self, sample_num):
        if not self.dropout_in_sampling:
            self.set_policy_mode("eval")
        else:
            self.set_policy_mode("train")
        
        for i in range(sample_num):
            self.envs.resume_all()
            if i == 0:
                observations = self.envs.reset()
            else:
                observations = self.envs.call(['reset_current_episode']*self.envs.num_envs)
            
            episodes_reset_ids = [ep.episode_id for i, ep in enumerate(self.envs.current_episodes())]
            # print("current reset episodes: ", episodes_reset_ids)

            data_this_sample = self.sample_once(observations)
            self.data_buffer.append(data_this_sample)


    # In PPO_30_no_drop_env_efficient.py


# # =====================================================================================
    # # =================== 请将此完整函数粘贴到 PPO_30_no_drop_env_efficient.py 中 ===================
    # # =====================================================================================

    def sample_once(self, initial_obs):
        mode = 'train'
        
        # --- PPO 数据结构设置 (与之前相同) ---
        instr_max_len = self.config.GRPO_ORM.max_text_len
        instr_pad_id = 1
        task_type = 1 if self.config.MODEL.task_type == 'r2r' else 2
        observations = extract_instruction_tokens_new_30(
            initial_obs, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        with torch.no_grad():
            all_txt_ids = batch['instruction']
            all_txt_task_encoding = batch['txt_task_encoding']
            all_txt_masks = (all_txt_ids != instr_pad_id)
            all_txt_embeds = self.policy.net(
                mode='language', txt_ids=all_txt_ids,
                txt_task_encoding=all_txt_task_encoding, txt_masks=all_txt_masks
            )
        data_this_sample = {
            "data_buffer": [],
            "initial_txt_embeds": all_txt_embeds.cpu(),
            "initial_txt_masks": all_txt_masks.cpu()
        }

        # --- 轨迹 rollout 设置 ---
        not_done_index = list(range(self.envs.num_envs))
        self.gmaps = [GraphMap(True, self.config.GRPO_ORM.loc_noise, self.config.MODEL.merge_ghost, self.config.GRPO_ORM.ghost_aug) 
                      for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        
        # (关键) 初始化一个列表来存储上一步到目标的距离
        curr_eps = self.envs.current_episodes()
        previous_distances = [ep.info.get('geodesic_distance', 30.0) for ep in curr_eps]

        # --- 主循环 ---
        for stepk in range(self.max_len):
            txt_embeds = all_txt_embeds
            
            # --- (这部分和原来一样，直到 nav_outs) ---
            with torch.no_grad():
                wp_outputs = self.policy.net(mode="waypoint", waypoint_predictor=self.waypoint_predictor, observations=batch, in_train=True)
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({'mode': 'panorama'})
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(pano_masks, 1, keepdim=True)
            
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node_new(cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            cand_real_pos = [[self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis}) for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])] for i in range(self.envs.num_envs)]

            for i in range(self.envs.num_envs):
                self.gmaps[i].update_graph(prev_vp[i], stepk+1, cur_vp[i], cur_pos[i], avg_pano_embeds[i], cand_vp[i], cand_pos[i], pano_embeds[i][vp_inputs['nav_types'][i]==1], cand_real_pos[i])
            
            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type)
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_inputs_for_gpu = {**nav_inputs, 'mode': 'navigation', 'txt_embeds': txt_embeds, 'txt_masks': all_txt_masks}
            nav_inputs_copy_for_cpu = self.copy_nav_inputs_dict(nav_inputs)

            with torch.no_grad():
                nav_outs = self.policy.net(**nav_inputs_for_gpu)
                nav_probs = F.softmax(nav_outs['global_logits'], 1)
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample()
            
            # --- (PPO-MODIFIED) 重构数据收集与奖励计算流程 ---
            # 1. 在执行动作前，存储所有PPO需要的信息
            data_this_stepk = {
                "input": nav_inputs_copy_for_cpu,
                "action": a_t.cpu(),
                "probs": nav_probs.cpu(),
                "value": nav_outs['value'].cpu(),
                "indices": copy.deepcopy(not_done_index),
            }
            
            # 2. 准备并执行动作
            cpu_a_t = a_t.cpu().numpy()
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()
            
            env_actions = []
            # ... (这部分env_actions的准备逻辑和原来完全一样，直接复制即可) ...
            use_tryout = (self.config.GRPO_ORM.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # ... stop action ...
                    vp_stop_scores=[(vp,ss)for vp,ss in gmap.node_stop_scores.items()];stop_scores=[s[1]for s in vp_stop_scores];stop_vp=vp_stop_scores[np.argmax(stop_scores)][0];stop_pos=gmap.node_pos[stop_vp]
                    back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]][1:] if self.config.GRPO_ORM.back_algo == 'control' else None
                    env_actions.append({'action':{'act':0,'cur_vp':cur_vp[i],'stop_vp':stop_vp,'stop_pos':stop_pos,'back_path':back_path,'tryout':use_tryout},'vis_info':None})
                else:
                    # ... navigation action ...
                    ghost_vp=nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]];ghost_pos=gmap.ghost_aug_pos[ghost_vp];_,front_vp=gmap.front_to_ghost_dist(ghost_vp);front_pos=gmap.node_pos[front_vp]
                    back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]][1:] if self.config.GRPO_ORM.back_algo == 'control' else None
                    env_actions.append({'action':{'act':4,'cur_vp':cur_vp[i],'front_vp':front_vp,'front_pos':front_pos,'ghost_vp':ghost_vp,'ghost_pos':ghost_pos,'back_path':back_path,'tryout':use_tryout},'vis_info':None})
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost: gmap.delete_ghost(ghost_vp)
            
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # 3. 计算带有强力奖励整形的每一步奖励
            rewards_step = [0.0] * self.envs.num_envs
            current_distances = [info.get('distance_to_goal', previous_distances[i]) for i, info in enumerate(infos)]

            for i in range(self.envs.num_envs):
                # 1. 巨大最终奖励
                final_reward = 0.0
                if dones[i]:
                    if infos[i]['distance_to_goal'] <= 1.5:
                        final_reward = 5.0  # 巨大成功奖励
                    else:
                        final_reward = -1.0 # 失败惩罚
                
                # 2. 距离变化奖励 (关键)
                distance_change = previous_distances[i] - current_distances[i]
                distance_reward = distance_change * 0.2 # 接近奖励，远离惩罚
                
                # 3. 固定步数惩罚
                step_penalty = -0.01

                rewards_step[i] = final_reward + distance_reward + step_penalty
                
                # 日志记录 (只在结束时)
                if dones[i]:
                    success = 1.0 if infos[i]['distance_to_goal'] <= 1.5 else 0.0
                    gt_length = infos[i]['position_train']['distance'][0]
                    path_length = np.linalg.norm(np.array(infos[i]['position_train']['position'])[1:] - np.array(infos[i]['position_train']['position'])[:-1], axis=1).sum()
                    spl = success * gt_length / max(gt_length, path_length, 1e-6)
                    self.logs['spl_final'].append(spl)
                    self.logs['success_final'].append(success)
                    self.logs['dtg_final'].append(current_distances[i])

            # 更新 "上一步" 的距离，为下一次迭代做准备
            previous_distances = current_distances

            # 4. 补充数据包并存入 buffer
            data_this_stepk['reward'] = torch.tensor(rewards_step, dtype=torch.float32)
            data_this_stepk['done'] = torch.tensor(dones, dtype=torch.bool)
            data_this_sample['data_buffer'].append(data_this_stepk)

            # --- 环境管理 ---
            if sum(dones) > 0:
                for i in reversed(range(self.envs.num_envs)):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i); observations.pop(i); self.gmaps.pop(i); prev_vp.pop(i)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i+1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i+1:]), dim=0)

            if self.envs.num_envs == 0: break

            observations = extract_instruction_tokens_new_30(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                
        return data_this_sample
