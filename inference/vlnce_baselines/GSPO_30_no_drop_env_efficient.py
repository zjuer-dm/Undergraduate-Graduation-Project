# 相较于0910，加了一个网络权重一致的check，然后把log记录改成了多卡统一，而不是只记录0卡的数据
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

@baseline_registry.register_trainer(name="GSPO-30-no-drop-env-efficient")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.GRPO_ORM.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0
        # GRPO specific hyperparameters
        self.grpo_epsilon = config.GRPO_ORM.grpo_epsilon  # Clipping for PPO objective
        self.grpo_beta = config.GRPO_ORM.grpo_beta        # Coefficient for KL divergence loss
        if self.grpo_beta < 1e-6:
            self.need_ref_policy = False
        else:
            self.need_ref_policy = True
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
        print("config.GRPO_ORM:\n", config.GRPO_ORM)
        print(f"GRPO params: grpo_epsilon {self.grpo_epsilon}, grpo_beta {self.grpo_beta}, max_grad_norm {self.max_grad_norm}, grpo_update_epochs {self.grpo_update_epochs} \
              enable_amp {self.enable_amp}, need_ref_policy {self.need_ref_policy}, enable_all_dropouts {self.enable_all_dropouts}, dropout_rate {self.dropout_rate}, dropout_in_sampling {self.dropout_in_sampling}")
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
		
        if self.need_ref_policy:
            self.ref_policy = policy.from_config(
                config=config,
                observation_space=observation_space,
                action_space=action_space,
            )
            self.ref_policy.to(self.device)
            self.ref_policy.eval() # 参考策略通常处于评估模式
            for param in self.ref_policy.parameters():
                param.requires_grad = False
            # self.ref_policy.load_state_dict(self.policy.state_dict())
            if self.config.GPU_NUMBERS > 1:
                policy_state_dict = self.policy.state_dict()
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in policy_state_dict.items():
                    if k.startswith("net.module."):
                        name = k.replace("net.module.", "net.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                self.ref_policy.load_state_dict(new_state_dict)
            else:
                self.ref_policy.load_state_dict(self.policy.state_dict())
        else:
            logger.info("BETA == 0, Skip create ref_policy!")

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
                for k, v_list in logs.items():
                    if not v_list: 
                        # 如果某个进程的日志列表为空，也需要创建一个tensor参与all_reduce，否则会因tensor数量不一致而出错
                        local_sum = 0.0
                        local_count = 0
                    else:
                        local_sum = sum(v_list)
                        local_count = len(v_list)
                    
                    metric_tensor = torch.tensor([local_sum, local_count], dtype=torch.float64, device=self.device)
                    
                    # -- 2. 所有进程都参与 all_reduce --
                    distr.all_reduce(metric_tensor, op=distr.ReduceOp.SUM)
                    
                    # all_reduce之后，每个进程的 metric_tensor 中都包含了全局的总和
                    
                    # -- 3. 只有 rank 0 进程负责计算最终结果和记录 --
                    if self.local_rank < 1:
                        global_sum, global_count = metric_tensor[0].item(), metric_tensor[1].item()
                        if global_count > 0:
                            final_logs[k] = global_sum / global_count
                        else:
                            final_logs[k] = 0.0 # 或者其他默认值
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
    

    def update(self):
        if not self.data_buffer:
            logger.info("Data buffer is empty. Skipping update.")
            return

        self.set_policy_mode("train") # Switch trainable parts to train mode

        # --- 1. Pre-calculate advantages (SEQUENCE-LEVEL) ---
        # This part is correct for both GRPO and GSPO, NO CHANGES NEEDED HERE.
        advantages_all_ep_all_samples = [[0.0] * self.initial_num_envs for _ in range(self.config.GRPO_ORM.sample_num)]
        rewards_per_original_env_slot = [[] for _ in range(self.initial_num_envs)]

        for s_idx in range(self.config.GRPO_ORM.sample_num):
            for original_env_idx in range(self.initial_num_envs):
                reward_val = self.data_buffer[s_idx]["reward"][original_env_idx]
                if reward_val is not None:
                    rewards_per_original_env_slot[original_env_idx].append(reward_val)

        for original_env_idx in range(self.initial_num_envs):
            rewards_for_this_slot = rewards_per_original_env_slot[original_env_idx]
            if len(rewards_for_this_slot) > 1:
                mean_r = np.mean(rewards_for_this_slot)
                std_r = np.std(rewards_for_this_slot)
                for s_idx in range(self.config.GRPO_ORM.sample_num):
                    reward_val = self.data_buffer[s_idx]["reward"][original_env_idx]
                    if reward_val is not None:
                        advantages_all_ep_all_samples[s_idx][original_env_idx] = (reward_val - mean_r) / (std_r + 1e-8)
        
        all_spls_in_buffer = []
        for s_idx in range(self.config.GRPO_ORM.sample_num):
            all_spls_in_buffer.extend([r for r in self.data_buffer[s_idx]["reward"] if r is not None])
        if all_spls_in_buffer:
            self.logs['reward'].append(np.mean(all_spls_in_buffer))
        else:
            self.logs['reward'].append(0.0)

        # --- Variables for accumulating losses across all epochs ---
        total_policy_loss_across_epochs = 0.0
        total_kl_loss_across_epochs = 0.0
        total_combined_loss_across_epochs = 0.0
        actual_epochs_processed = 0

        # --- 2. Multi-epoch training loop over the same data buffer ---
        for epoch in range(self.grpo_update_epochs):
            self.optimizer.zero_grad()

            # <--- GSPO Change Start: Process trajectories in batches --->
            # We will collect all steps from the buffer, re-calculate their log_probs under the current policy,
            # and then aggregate them by trajectory to compute sequence-level losses.

            # 1. Collate all data points from the buffer
            all_nav_inputs = []
            all_taken_actions = []
            all_old_log_probs = []
            all_advantages = []
            # This traj_ids tensor is crucial for grouping steps into trajectories
            all_traj_ids = [] 
            
            current_traj_id = 0
            for s_idx in range(self.config.GRPO_ORM.sample_num):
                # Each item in data_buffer corresponds to one trajectory
                trajectory_data = self.data_buffer[s_idx]["data_buffer"]
                initial_txt_embeds = self.data_buffer[s_idx]["initial_txt_embeds"].to(self.device)
                initial_txt_masks = self.data_buffer[s_idx]["initial_txt_masks"].to(self.device)

                # Get advantage for each step in this trajectory. It's constant for all steps.
                # We map the advantage from the original env slot to the active env index at step 0
                original_env_idx = trajectory_data[0]["indices"][0]
                advantage_for_this_traj = advantages_all_ep_all_samples[s_idx][original_env_idx]

                for step_data in trajectory_data:
                    # Append state, action, old_prob for each step
                    nav_inputs_cuda = {}
                    for key, value in step_data["input"].items():
                        if isinstance(value, torch.Tensor):
                            nav_inputs_cuda[key] = value.to(self.device, non_blocking=True)
                        else:
                            nav_inputs_cuda[key] = value # e.g. gmap_vp_ids list
                    
                    active_indices = step_data["indices"]
                    nav_inputs_cuda['txt_embeds'] = initial_txt_embeds[active_indices]
                    nav_inputs_cuda['txt_masks'] = initial_txt_masks[active_indices]
                    nav_inputs_cuda['mode'] = 'navigation'

                    all_nav_inputs.append(nav_inputs_cuda)
                    
                    taken_actions = step_data["action"].to(self.device)
                    all_taken_actions.append(taken_actions)
                    
                    old_probs = step_data["probs"].to(self.device)
                    old_log_probs = torch.log(old_probs.gather(1, taken_actions.unsqueeze(1)).squeeze(1) + 1e-9)
                    all_old_log_probs.append(old_log_probs)

                    # Assign the same trajectory ID and advantage to all steps of this trajectory
                    all_traj_ids.append(torch.full_like(taken_actions, fill_value=current_traj_id))
                    all_advantages.append(torch.full_like(taken_actions, fill_value=advantage_for_this_traj, dtype=torch.float32))

                current_traj_id += 1
            
            if not all_nav_inputs:
                continue

            # 2. Re-compute log_probs for all steps in a single batch
            all_current_log_probs = []
            with autocast(enabled=self.enable_amp):
                # This loop is inefficient. A better implementation would batch the forward passes.
                # For simplicity and correctness, we process them one by one.
                for nav_inputs in all_nav_inputs:
                    current_policy_outputs = self.policy.net(**nav_inputs)
                    current_logits = current_policy_outputs['global_logits']
                    current_log_probs = F.log_softmax(current_logits, dim=1)
                    all_current_log_probs.append(current_log_probs)

            # Concatenate lists of tensors into single large tensors
            actions_tensor = torch.cat(all_taken_actions)
            old_log_probs_tensor = torch.cat(all_old_log_probs)
            current_log_probs_all_actions_tensor = torch.cat(all_current_log_probs)
            traj_ids_tensor = torch.cat(all_traj_ids)
            advantages_tensor = torch.cat(all_advantages)

            current_log_probs_tensor = current_log_probs_all_actions_tensor.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # 3. Aggregate log_probs to sequence-level
            num_trajectories = current_traj_id
            log_p_new_sequence = torch.zeros(num_trajectories, device=self.device)
            log_p_old_sequence = torch.zeros(num_trajectories, device=self.device)
            traj_lengths = torch.zeros(num_trajectories, device=self.device)
            
            # Use scatter_add_ to sum log_probs for each trajectory
            log_p_new_sequence.scatter_add_(0, traj_ids_tensor, current_log_probs_tensor)
            log_p_old_sequence.scatter_add_(0, traj_ids_tensor, old_log_probs_tensor.detach())
            traj_lengths.scatter_add_(0, traj_ids_tensor, torch.ones_like(traj_ids_tensor, dtype=torch.float32))

            # Also get the unique advantage for each trajectory
            # Since advantage is the same for all steps of a traj, we can just take the first one.
            unique_advantages = torch.zeros(num_trajectories, device=self.device)
            unique_advantages.scatter_reduce_(0, traj_ids_tensor, advantages_tensor, reduce="mean", include_self=False)

            # Avoid division by zero for trajectories of length 0 (should not happen)
            traj_lengths = torch.clamp(traj_lengths, min=1)

            # 4. Calculate sequence-level importance ratio (GSPO paper, Eq. 7)
            # Applying length normalization to reduce variance
            s_i = torch.exp((log_p_new_sequence / traj_lengths) - (log_p_old_sequence / traj_lengths))

            # 5. Calculate GSPO clipped surrogate objective (GSPO paper, Eq. 5)
            surr1 = s_i * unique_advantages
            surr2 = torch.clamp(s_i, 1.0 - self.grpo_epsilon, 1.0 + self.grpo_epsilon) * unique_advantages
            
            # The loss is the negative of the objective, averaged over all trajectories
            policy_loss = -torch.min(surr1, surr2).mean()

            # (Optional but recommended) KL-divergence penalty at sequence level
            kl_loss = torch.tensor(0.0, device=self.device)
            if self.need_ref_policy:
                 # A simple approximation for sequence-level KL divergence
                kl_div_per_traj = (s_i - 1) - torch.log(s_i)
                kl_loss = kl_div_per_traj.mean()
            
            total_loss = policy_loss + self.grpo_beta * kl_loss

            # 6. Backward pass and optimizer step
            self.scaler.scale(total_loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
            if trainable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                self.logs['grad_norm'].append(grad_norm.item())
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- Logging ---
            total_policy_loss_across_epochs += policy_loss.item()
            if self.need_ref_policy:
                total_kl_loss_across_epochs += kl_loss.item()
            total_combined_loss_across_epochs += total_loss.item()
            actual_epochs_processed += 1
            # <--- GSPO Change End --->

        # --- Log averaged losses after all epochs for this update cycle ---
        if actual_epochs_processed > 0:
            self.logs['policy_loss'].append(total_policy_loss_across_epochs / actual_epochs_processed)
            if self.need_ref_policy:
                self.logs['kl_loss'].append(total_kl_loss_across_epochs / actual_epochs_processed)
            self.logs['total_loss'].append(total_combined_loss_across_epochs / actual_epochs_processed)
        else:
            self.logs['policy_loss'].append(0.0)
            self.logs['kl_loss'].append(0.0)
            self.logs['total_loss'].append(0.0)

        self.data_buffer.clear()
        self.optimizer.zero_grad()
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

    def sample_once(self, initial_obs):
        mode = 'train'
        # data_this_sample = {"reward": [None] * self.envs.num_envs, "data_buffer": []}


        instr_max_len = self.config.GRPO_ORM.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1
        if self.config.MODEL.task_type == 'r2r':
            task_type = 1
        elif self.config.MODEL.task_type == 'rxr':
            task_type = 2
        else:
            print("self.config.MODEL.task_type Error")
        # 运行下面函数之前，observations[i][instruction_sensor_uuid]是一个由'text',"tokens","trajectory_id"为key组成的字典
        # 运行之后，observations[i][instruction_sensor_uuid]就是一个列表，表示"tokens"
        observations = extract_instruction_tokens_new_30(initial_obs, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, # INSTRUCTION_SENSOR_UUID = instruction
                                                  max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
        batch = batch_obs(observations, self.device) # 返回一个字典，字典中key等于observations[i]的key，value等于每个observations[i][key]的值合并成的一个batch，且转为了torch.tensor
        # print("self.obs_transforms", self.obs_transforms)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms) # infer时self.obs_transforms为空，train时为[CenterCropperPerSensor()]，因为default.py中ENABLED_TRANSFORMS只设了它

        # encode instructions
        all_txt_ids = batch['instruction'] # torch.tensor类型，环境数*80的shape，数值为整数
        all_txt_task_encoding = batch['txt_task_encoding']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_task_encoding=all_txt_task_encoding,
            txt_masks=all_txt_masks,
        )

        data_this_sample = {
            "reward": [None] * self.envs.num_envs,
            "data_buffer": [],
            # 只在轨迹开始时存储一次完整的文本特征到CPU
            "initial_txt_embeds": all_txt_embeds.detach().cpu(),
            "initial_txt_masks": all_txt_masks.detach().cpu()
        }

        total_actions = 0.
        
        not_done_index = list(range(self.envs.num_envs)) # envs.num_envs返回的是没有暂停的envs的数量
        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION) # 注意：or 操作符会返回第一个为真的值，或者最后一个值（如果没有真值），因此非训练模式返回值为[]（空列表代表False）
        ghost_aug = self.config.GRPO_ORM.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, # []
                               self.config.GRPO_ORM.loc_noise, # 0.5
                               self.config.MODEL.merge_ghost, # True
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len): # 15
            # print("check ", stepk, self.max_len, instr_max_len)
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks
            txt_embeds = all_txt_embeds
            
            # cand waypoint prediction
            # outputs = {
            #     'cand_rgb': cand_rgb,               # [K x 2048]
            #     'cand_depth': cand_depth,           # [K x 128]
            #     'cand_angle_fts': cand_angle_fts,   # [K x 4]
            #     'cand_img_idxes': cand_img_idxes,   # [K]
            #     'cand_angles': cand_angles,         # [K]
            #     'cand_distances': cand_distances,   # [K]

            #     'pano_rgb': pano_rgb,               # B x 12 x 2048
            #     'pano_depth': pano_depth,           # B x 12 x 128
            #     'pano_angle_fts': pano_angle_fts,   # 12 x 4
            #     'pano_img_idxes': pano_img_idxes,   # 12 
            # }
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.GRPO_ORM.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            # pano_embeds:self.envs.num_envs*len(nav_types)*768(nav_types详见self._vp_feature_variable函数)
            # pano_masks:self.envs.num_envs*len(nav_types)(该返回值的作用是，将pano_embeds在不同env中长度对齐，因为不同env中len(nav_types)长度不一样，取最大值，padding部分为False)
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            # 下面pano_embeds * pano_masks.unsqueeze(2)进行的是逐元素相乘（例如[2, 14, 768]*[2, 14, 1]）
            # avg_pano_embeds:self.envs.num_envs*768,表示当前观测的一个融合feature
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            # cur_pos:list of array, 长度为self.envs.num_envs，每个array表示一个global坐标（3维）
            # cur_ori:list of array, 长度为self.envs.num_envs，每个array表示一个global转角4元数（4维）
            # cur_vp_i：在self.gmaps[i]中，当前新节点的编号是多少（str，例如：8）；
            # cand_vp_i：在self.gmaps[i]中，当前新出现的waypoints的编号列表（str，例如：['8_0', '8_1', '8_2', '8_3', '8_4']）；
            # cand_pos_i:在self.gmaps[i]中，当前新出现的waypoints的global位置列表(通过当前位置和距离+角度来计算)（array，例如：[array([9.78167991, 2.64581203, 3.03715693]), ...）；
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node_new(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i) # 这里的cand_pos是通过直接根据当前坐标和相对坐标计算得到ghost的坐标（假定高度不变）
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    # 获取frontier的全局坐标，通过让机器人朝对应方向走对应步数得到（应该主要是防止高度不一致的情况）
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1] # 取waypoints对应的K个pano_embeds
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                        cur_vp[i], cur_pos[i], cur_embeds,
                                        cand_vp[i], cand_pos[i], cand_embeds,
                                        cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type)
            nav_inputs.update({
                'mode': 'navigation',
                # 'txt_embeds': txt_embeds,
                # 'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left') # 表示动作空间是否为空，bool类型

            nav_inputs_for_gpu = nav_inputs.copy()
            nav_inputs_for_gpu['txt_embeds'] = txt_embeds
            nav_inputs_for_gpu['txt_masks'] = txt_masks

            # nav_inputs_copy = self.copy_nav_inputs_dict(nav_inputs)
            nav_inputs_copy_for_cpu = self.copy_nav_inputs_dict(nav_inputs)
            # nav_outs = self.policy.net(**nav_inputs)
            nav_outs = self.policy.net(**nav_inputs_for_gpu)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)

            # ref_nav_outs = self.ref_policy.net(**nav_inputs)
            # ref_nav_logits = ref_nav_outs['global_logits']
            # ref_nav_probs = F.softmax(ref_nav_logits, 1)

            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item() # 在导航过程中增量记录每个节点的stop_score，也就是当时做决策时stop动作的概率

            # determine action
            c = torch.distributions.Categorical(nav_probs)
            # 下面a_t的shape就是一维tensor，长度为num_envs（会随着某几个环境运行完毕而减少）
            a_t = c.sample().detach()
            cpu_a_t = a_t.cpu().numpy()

            # ------------------- start store data ------------------- 
            data_this_stepk = {}
            data_this_stepk["input"] = nav_inputs_copy_for_cpu # dict，如果sample不禁止梯度，且网络可训练参数包含了txt和pano的encoder，那么里面的txt和img对应feature就包含梯度
            data_this_stepk["action"] = a_t.detach().cpu() # 长为self.envs.num_envs的一维tensor，不包含梯度，对于在gpu上的变量，.cpu()直接创建一个新的变量，因此a_t变了也没事
            data_this_stepk["probs"] = nav_probs.detach().cpu() # self.envs.num_envs * action_nums的二维tensor，不包含梯度
            data_this_stepk["indices"] = copy.deepcopy(not_done_index) # 长为self.envs.num_envs的列表
            data_this_sample['data_buffer'].append(data_this_stepk)
            # ------------------- end store data ------------------- 

            # make equiv action
            env_actions = []
            use_tryout = (self.config.GRPO_ORM.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING) # 设置tryout并且不允许滑动才启用use_tryout
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]: # 如果的当前动作是stop或步数达到上限或没有动作空间
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]

                    if self.config.GRPO_ORM.back_algo == 'control': # eval和infer进的是这里，也是通过节点路径走过去的
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp) # 计算得到从当前位置到哪个与ghost_vp相邻的常规节点最近，取其为front_vp
                    front_pos = gmap.node_pos[front_vp]
                    vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.GRPO_ORM.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            curr_eps = self.envs.current_episodes()
            if self.config.MODEL.task_type == 'r2r':
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    pred_path = np.array(info['position_train']['position'])
                    distances = np.array(info['position_train']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 1.5 else 0.
                    metric['oracle_success'] = 1. if (distances <= 1.5).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    # if metric['distance_to_goal'] > 1.5:
                    #     distance_to_goal_reward = 1.25 - (metric['distance_to_goal']) / 3
                    #     # distance_to_goal_reward = 2 - 2 * (metric['distance_to_goal']) / 3
                    #     # distance_to_goal_reward = 1 - 1 * (metric['distance_to_goal']) / 3
                    # else:
                    #     distance_to_goal_reward = 1 - (metric['distance_to_goal'])**2 / 9
                    distance_to_goal_reward = 1 - 1 * (metric['distance_to_goal']) / 6
                    data_this_sample["reward"][not_done_index[i]] = metric['spl'] + metric['success'] + distance_to_goal_reward
                    self.logs['spl_reward'].append(metric['spl'])
                    self.logs['success_reward'].append(metric['success'])
                    self.logs['NE'].append(metric['distance_to_goal'])
                    self.logs['distance_to_goal_reward'].append(distance_to_goal_reward)
                    # print(f"ep_id: {ep_id}, spl: {metric['spl']}, suc: {metric['success']}, distance_to_goal: {metric['distance_to_goal']}, distance_to_goal_reward: {distance_to_goal_reward}, total_reward: {data_this_sample['reward'][not_done_index[i]]}")
            elif self.config.MODEL.task_type == 'rxr':
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position_train']['position']) # TODO:为什么teleport还会有逐步的路径？为什么偶尔出现一步走好几米的情况？
                    # print("check len(pred_path)", len(pred_path))
                    # if ep_id == '37632':
                    #     print(pred_path)
                    distances = np.array(info['position_train']['distance'])
                    # gt_length = distances[0]
                    gt_length = max(self.gt_data[str(ep_id)]['forward_steps']*0.25, distances[0])
                    metric = {}
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 1.5 else 0.
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    # metric['habitat_ndtw'] = info['ndtw']
                    # metric['habitat_sdtw'] = info['sdtw']
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    distance_to_goal_reward = 1 - 1 * (metric['distance_to_goal']) / 6
                    data_this_sample["reward"][not_done_index[i]] = metric['ndtw'] + metric['sdtw'] + distance_to_goal_reward + metric['spl']
                    self.logs['ndtw'].append(metric['ndtw'])
                    self.logs['success'].append(metric['success'])
                    self.logs['spl'].append(metric['spl'])
                    self.logs['sdtw'].append(metric['sdtw'])
                    self.logs['NE'].append(metric['distance_to_goal'])
                    self.logs['distance_to_goal_reward'].append(distance_to_goal_reward)
                    # print(f"check ndtw: habitat_ndtw:{metric['habitat_ndtw']}, ndtw:{metric['ndtw']}")
                    # print(f"check sdtw: habitat_sdtw:{metric['habitat_sdtw']}, sdtw:{metric['sdtw']}")
                    # print(f"check: ep_id:{ep_id}, ndtw:{metric['ndtw']}, sdtw:{metric['sdtw']}, NE:{metric['distance_to_goal']}")
                    # print(f"ep_id:{ep_id}, shortest_dist: {distances[0]}, gt_dist: {self.gt_data[str(ep_id)]['forward_steps']*0.25}, path_length: {metric['path_length']}, spl: {metric['spl']}")
                    # print(self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        stopped_env_index = not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        all_txt_ids = torch.cat((all_txt_ids[:i], all_txt_ids[i + 1:]), dim=0)
                        all_txt_task_encoding = torch.cat((all_txt_task_encoding[:i], all_txt_task_encoding[i + 1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i + 1:]), dim=0)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i + 1:]), dim=0)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens_new_30(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, \
                                                                         max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        # print("data_this_sample: ", data_this_sample["reward"], len(data_this_sample['data_buffer']))
        return data_this_sample