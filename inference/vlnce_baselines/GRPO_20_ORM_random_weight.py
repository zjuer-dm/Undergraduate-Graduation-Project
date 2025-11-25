# GRPO_20_ORM_action_mean
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
from vlnce_baselines.common.utils import extract_instruction_tokens_new_20
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

@baseline_registry.register_trainer(name="GRPO-20-ORM-random-weight")
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
        self.train_all = config.GRPO_ORM.train_all
        # self.enable_amp = config.GRPO_ORM.enable_amp or config.GRPO_ORM.train_all
        self.enable_amp = config.GRPO_ORM.enable_amp
        self.enable_all_dropouts = config.GRPO_ORM.enable_all_dropouts
        self.dropout_rate = config.GRPO_ORM.dropout_rate
        self.scaler = GradScaler(enabled=self.enable_amp)
        print("config.GRPO_ORM:\n", config.GRPO_ORM)
        print(f"GRPO params: grpo_epsilon {self.grpo_epsilon}, grpo_beta {self.grpo_beta}, max_grad_norm {self.max_grad_norm}, grpo_update_epochs {self.grpo_update_epochs} \
              train_all {self.train_all}, enable_amp {self.enable_amp}, need_ref_policy {self.need_ref_policy}, enable_all_dropouts {self.enable_all_dropouts}, dropout_rate {self.dropout_rate}")
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
            if self.enable_amp:
                torch.save(
                    obj={
                        "state_dict": self.policy.state_dict(), # 网络权重
                        "config": self.config, # 配置参数
                        "optim_state": self.optimizer.state_dict(), # 保存时优化器的状态（可以读取后进行继续训练，而不是从头开始的训练）
                        "iteration": iteration, # 保存时训练了几步
                        "scaler_state": self.scaler.state_dict()
                    },
                    f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
                )
            else:
                torch.save(
                    obj={
                        "state_dict": self.policy.state_dict(), # 网络权重
                        "config": self.config, # 配置参数
                        "optim_state": self.optimizer.state_dict(), # 保存时优化器的状态（可以读取后进行继续训练，而不是从头开始的训练）
                        "iteration": iteration, # 保存时训练了几步
                    },
                    f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
                )


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
                self.policy.train()
                if self.world_size > 1:
                    self.policy.net.module.rgb_encoder.eval()
                    self.policy.net.module.depth_encoder.eval()
                else:
                    self.policy.net.rgb_encoder.eval()
                    self.policy.net.depth_encoder.eval()
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
            need_gsp=False
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

        if self.config.GPU_NUMBERS > 1:
            # 即使是多卡时，waypoint_predictor也不需要封装，因为一方面不涉及多卡训练，另一方面也不涉及多卡推理时的数据集自动切分
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
            
        try:
            if self.config.GPU_NUMBERS > 1:
                vln_bert_module = self.policy.net.module.vln_bert
            else:
                vln_bert_module = self.policy.net.vln_bert
            if self.train_all:
                self.trainable_parts = [vln_bert_module]
            else:
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
        trainable_parameters = [p for p in self.policy.parameters() if p.requires_grad]
        not_trainable_parameters = [p for p in self.policy.parameters() if not p.requires_grad]
        if trainable_parameters:
            self.optimizer = torch.optim.AdamW(trainable_parameters, lr=self.config.GRPO_ORM.lr)
            print(f"Optimizer configured with {len(trainable_parameters)} trainable parameters.")
            print(f"Remaining {len(not_trainable_parameters)} untrainable parameters.")
        else:
            self.optimizer = None
            print("Warning: No parameters were set to trainable. Optimizer not configured.")

        if load_from_ckpt: # infer中为True
            if config.GRPO_ORM.is_requeue: # 意思是是否需要在中断后恢复训练
                # print()
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
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

            # net.module 是 torch.nn.DataParallel 或 torch.nn.parallel.DistributedDataParallel 封装的实际模型（原始模型）。
            # 当你将一个模型通过 DataParallel 或 DistributedDataParallel 封装后，原始模型会被包装在一个外层模块中，称为 DataParallel 或 DistributedDataParallel 对象。
            # module 是一个属性，它指向被封装的原始模型。
            # 为什么上面用DDP，底下又用torch.nn.DataParallel？
            #   答：多gpu情况下，DDP效率高；而单gpu，且需要load_state_dict的情况下，若state_dict的键包含module前缀，说明之前的模型是在DataParallel或DDP封装下保存的。这种情况下：
            #   暂时用 torch.nn.DataParallel 封装网络，使其兼容state_dict的结构。
            #   加载权重后再移除封装（通过 self.policy.net.module），将网络还原为原始模型。
            # 为什么self.waypoint_predictor封装之后不解封？
            #   答：注释self.waypoint_predictor封装代码发现，不封装也没事（infer和eval阶段实验证明）。
            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                # 这个policy的最基类是nn.Module，因此有load_state_dict等函数，strict=False意思是，如果现在的模型含有一些保存权重没有的新参数，那么忽略它们，只加载有的参数
                # 对于继承了nn.Module类的load_state_dict函数，本质上就是把一个权重字典赋值到当前类中对应属性中，例如fc1.weight, fc1.bias对应类定义中的self.fc1。
                # 所以继承了nn.Module类内部可以定义很多与网络无关的变量，最后load_state_dict(strict=False)时就只会把在权重字典中能找到的属性赋值
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module # net是ILPolicy中的属性，在这里就代表着ETP这个class的实例
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
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
                if "scaler_state" in ckpt_dict and self.enable_amp:
                    self.scaler.load_state_dict(ckpt_dict["scaler_state"])
                    logger.info("Loaded GradScaler state from checkpoint.")
            else:
                print("optimizer is initialized")
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
		
        if self.need_ref_policy:
            self.ref_policy = policy.from_config(
                config=config,
                observation_space=observation_space,
                action_space=action_space,
                need_gsp=False
            )
            self.ref_policy.to(self.device)
            self.ref_policy.eval() # 参考策略通常处于评估模式
            for param in self.ref_policy.parameters():
                param.requires_grad = False
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
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

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
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
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
            'no_vp_left': batch_no_vp_left,
        }

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause, global_subtask_num, global_subtask_stage):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]
            # print("before ", global_subtask_num, global_subtask_stage)
            global_subtask_num = [global_subtask_num[i] for i in state_index]
            global_subtask_stage = [global_subtask_stage[i] for i in state_index]
            # print("after ", global_subtask_num, global_subtask_stage)

        return envs, batch, global_subtask_num, global_subtask_stage

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

        logger.info('Traning Starts... GOOD LUCK!')
        
        self.data_buffer = []
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0)) # interval表示当前这次循环需要进行几次iteration
            cur_iter = idx + interval # cur_iter表示当前这次循环完成后一共进行了几次iteration, idx表示当前这次循环完成前一共进行了几次iteration

            # sample_ratio = self.config.GRPO_ORM.sample_ratio ** (idx // self.config.GRPO_ORM.decay_interval)
            logs = self._train_interval(interval)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
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
            if self.train_all:
                with autocast(enabled=self.enable_amp):
                    self.sample_data(self.config.GRPO_ORM.sample_num)
            else:
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

        # --- 1. Pre-calculate advantages for all trajectories (done once for the entire buffer) ---
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
            # If only one sample or no samples for the slot, advantage remains 0.0
        # Log average raw reward (SPL) from this update batch (done once)
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

        total_processed_actions_for_ratio = 0
        total_unclipped_actions = 0

        # --- 2. Multi-epoch training loop over the same data buffer ---
        for epoch in range(self.grpo_update_epochs):
            # Variables for accumulating losses within this single epoch
            accumulated_policy_loss_this_epoch = 0.0
            accumulated_kl_loss_this_epoch = 0.0
            num_samples_processed_this_epoch = 0 # Counts how many s_idx contributed to loss in this epoch

            self.optimizer.zero_grad() # Zero gradients at the beginning of each epoch

            # Iterate through each sampled trajectory batch for loss calculation
            for s_idx in range(self.config.GRPO_ORM.sample_num):
                current_sample_trajectory_steps_data = self.data_buffer[s_idx]["data_buffer"]
                
                batch_total_policy_loss_for_this_sample = 0.0
                batch_total_kl_loss_for_this_sample = 0.0
                # num_valid_steps_for_this_sample = 0
                total_actions_in_this_sample = 0
                with autocast(enabled=self.enable_amp):
                    for step_data in current_sample_trajectory_steps_data:
                        with autocast(enabled=False):
                            nav_inputs_cpu = step_data["input"]
                            taken_actions_cpu = step_data["action"]
                            old_probs_at_sampling_cpu = step_data["probs"]
                            active_indices_in_original_batch = step_data["indices"] 

                            if not active_indices_in_original_batch:
                                print("ERROR!! NO active_indices_in_original_batch")
                                continue
                        
                            nav_inputs_cuda = {}
                            for key, value in nav_inputs_cpu.items():
                                if isinstance(value, torch.Tensor):
                                    nav_inputs_cuda[key] = value.to(self.device, non_blocking=True)
                                else:
                                    nav_inputs_cuda[key] = value
                            nav_inputs_cuda['mode'] = 'navigation' 

                            taken_actions_cuda = taken_actions_cpu.to(self.device)
                    
                        # 如果self.train_all，那么这里输入的nav_inputs_cuda['txt_embeds']和nav_inputs_cuda['gmap_img_fts']都包含梯度
                        current_policy_outputs = self.policy.net(**nav_inputs_cuda)
                        if self.need_ref_policy:
                            with torch.no_grad():
                                ref_policy_outputs = self.ref_policy.net(**nav_inputs_cuda)

                        with autocast(enabled=False):
                            current_logits = current_policy_outputs['global_logits']
                            current_log_probs = F.log_softmax(current_logits, dim=1)
                            current_log_probs_taken_action = current_log_probs.gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1)
                            if self.need_ref_policy:
                                ref_logits = ref_policy_outputs['global_logits']
                                ref_log_probs = F.log_softmax(ref_logits, dim=1)
                                ref_log_probs_taken_action_no_grad = ref_log_probs.gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1)

                            step_advantages_for_active_envs = torch.tensor(
                                [advantages_all_ep_all_samples[s_idx][orig_idx] for orig_idx in active_indices_in_original_batch],
                                device=self.device, dtype=torch.float32
                            )

                            if self.grpo_update_epochs == -1:
                                # 当只更新一次时，行为类似于直接用当前策略的样本进行更新，IS weight 为 1
                                old_log_probs_taken_action = current_log_probs_taken_action.detach()
                            else:
                                # 多轮更新时，使用采样时记录的旧策略概率
                                old_log_probs_taken_action = torch.log(
                                    old_probs_at_sampling_cpu.to(self.device).gather(1, taken_actions_cuda.unsqueeze(1)).squeeze(1) + 1e-9
                                )
                            
                            ratio = torch.exp(current_log_probs_taken_action - old_log_probs_taken_action)
                            surr1 = ratio * step_advantages_for_active_envs
                            surr2 = torch.clamp(ratio, 1.0 - self.grpo_epsilon, 1.0 + self.grpo_epsilon) * step_advantages_for_active_envs

                            policy_objective = -torch.min(surr1, surr2)
                            random_weights = 2.0 * torch.rand_like(policy_objective)
                            policy_loss_this_step = (policy_objective * random_weights).sum()

                            unclipped_mask = (surr1 <= surr2)
                            total_unclipped_actions += unclipped_mask.sum().item()
                            total_processed_actions_for_ratio += len(unclipped_mask)

                            if self.need_ref_policy:
                                ratio_ref_over_current = torch.exp(ref_log_probs_taken_action_no_grad - current_log_probs_taken_action)
                                log_ratio_ref_over_current = ref_log_probs_taken_action_no_grad - current_log_probs_taken_action
                                kl_objective = (ratio_ref_over_current - log_ratio_ref_over_current - 1)
                                kl_div_this_step = (kl_objective * random_weights).sum()
                                # print(f"random_weights: {random_weights}, policy_objective: {policy_objective}, policy_loss_this_step: {policy_loss_this_step}, kl_objective: {kl_objective}, kl_div_this_step: {kl_div_this_step}")
                            
                            batch_total_policy_loss_for_this_sample += policy_loss_this_step
                            if self.need_ref_policy:
                                batch_total_kl_loss_for_this_sample += kl_div_this_step
                            # num_valid_steps_for_this_sample += 1
                            total_actions_in_this_sample += len(active_indices_in_original_batch)
                
                if total_actions_in_this_sample > 0:
                    # with autocast(enabled=self.enable_amp):
                    avg_policy_loss_for_sample = batch_total_policy_loss_for_this_sample / total_actions_in_this_sample
                    if self.need_ref_policy:
                        avg_kl_loss_for_sample = batch_total_kl_loss_for_this_sample / total_actions_in_this_sample
                    # Total loss for this specific sample (from self.data_buffer)
                    # The gradient accumulation now happens per s_idx within an epoch
                    if self.need_ref_policy:
                        total_loss_for_this_sample = avg_policy_loss_for_sample + self.grpo_beta * avg_kl_loss_for_sample
                    else:
                        total_loss_for_this_sample = avg_policy_loss_for_sample
                    # total_loss_for_this_sample.backward()
                    # scaled_loss_for_this_sample = total_loss_for_this_sample / self.config.GRPO_ORM.sample_num
                    scaled_loss_for_this_sample = total_loss_for_this_sample

                    self.scaler.scale(scaled_loss_for_this_sample).backward()

                    accumulated_policy_loss_this_epoch += avg_policy_loss_for_sample.item()
                    if self.need_ref_policy:
                        accumulated_kl_loss_this_epoch += avg_kl_loss_for_sample.item()
                    num_samples_processed_this_epoch +=1
                else:
                    logger.info("ERROR! total_actions_in_this_sample is 0")
            
            # --- Optimizer step after processing all s_idx for the current epoch ---
            # if num_samples_processed_this_epoch > 0:
            if num_samples_processed_this_epoch == self.config.GRPO_ORM.sample_num:
                self.scaler.unscale_(self.optimizer)

                trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
                if trainable_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    self.logs['grad_norm'].append(grad_norm.item())
                    # print("grad_norm: ", grad_norm)
                
                self.scaler.step(self.optimizer)
                # self.optimizer.step() # Perform optimizer step at the end of each epoch
                self.scaler.update()

                # Accumulate epoch losses for final logging
                total_policy_loss_across_epochs += (accumulated_policy_loss_this_epoch / num_samples_processed_this_epoch)
                if self.need_ref_policy:
                    total_kl_loss_across_epochs += (accumulated_kl_loss_this_epoch / num_samples_processed_this_epoch)
                if self.need_ref_policy:
                    total_combined_loss_across_epochs += ( (accumulated_policy_loss_this_epoch + self.grpo_beta * accumulated_kl_loss_this_epoch) / num_samples_processed_this_epoch )
                else:
                    total_combined_loss_across_epochs += ( (accumulated_policy_loss_this_epoch) / num_samples_processed_this_epoch )
                actual_epochs_processed += 1
            else:
                logger.info(f"Epoch {epoch+1}/{self.grpo_update_epochs}: actual samples != GRPO_ORM.sample_num.")
                # If an epoch had no data, we might skip it for averaging.

        # --- Log averaged losses after all epochs for this update cycle ---
        if actual_epochs_processed > 0:
            self.logs['policy_loss'].append(total_policy_loss_across_epochs / actual_epochs_processed)
            if self.need_ref_policy:
                self.logs['kl_loss'].append(total_kl_loss_across_epochs / actual_epochs_processed)
            self.logs['total_loss'].append(total_combined_loss_across_epochs / actual_epochs_processed)

            if total_processed_actions_for_ratio > 0:
                unclipped_ratio = total_unclipped_actions / total_processed_actions_for_ratio
                self.logs['unclipped_ratio'].append(unclipped_ratio)
            else:
                self.logs['unclipped_ratio'].append(0.0) # Or handle as missing
        else:
            logger.info("No valid data processed across all epochs. Skipping logging of losses.")
            self.logs['policy_loss'].append(0.0)
            self.logs['kl_loss'].append(0.0)
            self.logs['total_loss'].append(0.0)

        self.data_buffer.clear() # Clear buffer after all epochs are done
        self.optimizer.zero_grad()
        # self.set_policy_mode("eval") # Switch back to eval mode for next sampling phase
    
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
        if not self.enable_all_dropouts:
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
        feedback = 'sample'
        mode = 'train'
        data_this_sample = {"reward": [None] * self.envs.num_envs, "data_buffer": []}


        instr_max_len = self.config.GRPO_ORM.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        # 运行下面函数之前，observations[i][instruction_sensor_uuid]是一个由'text',"tokens","trajectory_id"为key组成的字典
        # 运行之后，observations[i][instruction_sensor_uuid]就是一个列表，表示"tokens"
        global_subtask_stage = [1] * self.envs.num_envs
        # global_subtask_stage = self._teacher_stage_new(global_subtask_stage)
        observations, global_subtask_num = extract_instruction_tokens_new_20(initial_obs, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, # INSTRUCTION_SENSOR_UUID = instruction
                                                  max_length=instr_max_len, pad_id=instr_pad_id, output_subtask_num=True)
        batch = batch_obs(observations, self.device) # 返回一个字典，字典中key等于observations[i]的key，value等于每个observations[i][key]的值合并成的一个batch，且转为了torch.tensor
        # print("self.obs_transforms", self.obs_transforms)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms) # infer时self.obs_transforms为空，train时为[CenterCropperPerSensor()]，因为default.py中ENABLED_TRANSFORMS只设了它

        # encode instructions
        all_txt_ids = batch['instruction'] # torch.tensor类型，环境数*80的shape，数值为整数
        all_subtask_type_ids = batch['subtask_type_ids']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            subtask_type_ids=all_subtask_type_ids,
            txt_masks=all_txt_masks,
        )

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

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            global_subtask_len_tensor = torch.LongTensor(global_subtask_num).cuda() + 1
            # print("global_subtask_num ", global_subtask_len_tensor)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'subtask_lens': global_subtask_len_tensor
            })
            no_vp_left = nav_inputs.pop('no_vp_left') # 表示动作空间是否为空，bool类型



            nav_inputs_copy = self.copy_nav_inputs_dict(nav_inputs)
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            stage_logits = nav_outs['global_stage_logits']
            nav_probs = F.softmax(nav_logits, 1)

            # ref_nav_outs = self.ref_policy.net(**nav_inputs)
            # ref_nav_logits = ref_nav_outs['global_logits']
            # ref_stage_logits = ref_nav_outs['global_stage_logits']
            # ref_nav_probs = F.softmax(ref_nav_logits, 1)

            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item() # 在导航过程中增量记录每个节点的stop_score，也就是当时做决策时stop动作的概率

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                # 下面a_t的shape就是一维tensor，长度为num_envs（会随着某几个环境运行完毕而减少）
                a_t = c.sample().detach()
                # if stepk == 0:
                #     print("nav_probs ", nav_probs)
                #     print("ref nav probs ", ref_nav_probs)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()


            # ------------------- start store data ------------------- 
            data_this_stepk = {}
            data_this_stepk["input"] = nav_inputs_copy # dict，如果sample不禁止梯度，且网络可训练参数包含了txt和pano的encoder，那么里面的txt和img对应feature就包含梯度
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
                distance_to_goal_reward = 1 - 1 * (metric['distance_to_goal']) / 3
                data_this_sample["reward"][not_done_index[i]] = metric['spl'] + metric['success'] + distance_to_goal_reward
                self.logs['spl_reward'].append(metric['spl'])
                self.logs['success_reward'].append(metric['success'])
                self.logs['NE'].append(metric['distance_to_goal'])
                self.logs['distance_to_goal_reward'].append(distance_to_goal_reward)
                # print(f"ep_id: {ep_id}, spl: {metric['spl']}, suc: {metric['success']}, distance_to_goal: {metric['distance_to_goal']}, distance_to_goal_reward: {distance_to_goal_reward}, total_reward: {data_this_sample['reward'][not_done_index[i]]}")
            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        stopped_env_index = not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        global_subtask_stage.pop(i)
                        global_subtask_num.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        all_txt_ids = torch.cat((all_txt_ids[:i], all_txt_ids[i + 1:]), dim=0)
                        all_subtask_type_ids = torch.cat((all_subtask_type_ids[:i], all_subtask_type_ids[i + 1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i + 1:]), dim=0)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i + 1:]), dim=0)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens_new_20(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, \
                                                                         max_length=instr_max_len, pad_id=instr_pad_id)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        # print("data_this_sample: ", data_this_sample["reward"], len(data_this_sample['data_buffer']))
        return data_this_sample