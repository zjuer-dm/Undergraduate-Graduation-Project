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
from collections import OrderedDict

@baseline_registry.register_trainer(name="MY-SS-ETP-30-PPO")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        if self.config.ONLY_LAST_SAVEALL and (not iteration == self.config.IL.iters):
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
        self.batch_size = self.config.IL.batch_size
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
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank # 在这里为每个gpu进程下的env数据集提供了不同的随机性，即训练时每个gpu进程的episode循环模式是不同的
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
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        logger.info(f"-------------------Load pretrain weight: {config.MODEL.pretrained_path}-------------------")
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
        
        param_optimizer = list(self.policy.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.IL.lr)
        num_warmup_steps = self.config.IL.warmup_iters
        num_training_steps = self.config.IL.iters
        min_lr_ratio = self.config.IL.min_lr_ratio
        # def lr_lambda(current_step: int):
        #     # 预热阶段: 学习率从 0 线性增加到 1 * base_lr
        #     if current_step < num_warmup_steps:
        #         return float(current_step) / float(max(1, num_warmup_steps))
        #     # 预热后: 学习率从 1 * base_lr 线性衰减到 min_lr_ratio * base_lr
        #     # 计算衰减阶段的进度
        #     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        #     # 确保学习率不会低于最小值
        #     # (1.0 - progress) 会从1.0线性下降到0.0
        #     # 所以最终的乘数会从 1.0 线性下降到 min_lr_ratio
        #     return max(min_lr_ratio, 1.0 - progress)
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
            if config.IL.is_requeue: # 意思是是否需要在中断后恢复训练
                # print()
                import glob
                # ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                search_pattern = os.path.join(config.CHECKPOINT_FOLDER, "*.pth")
                ckpt_list = glob.glob(search_pattern)
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1] # 获取最新的权重文件来加载self.policy
            else:
                ckpt_path = config.IL.ckpt_to_load
            # 下面load的返回值是含有['state_dict', 'config', 'optim_state', 'iteration']为key的字典，对应了self.save_checkpoint保存代码
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            # start_iter = ckpt_dict["iteration"]
            if config.IL.is_requeue:
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
            elif 'module' not in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS > 1:
                new_state_dict = OrderedDict()
                for k, v in ckpt_dict['state_dict'].items():
                    # <--- 关键修改 --->
                    # 仅当键以'net.'开头时，才将其替换为'net.module.'，以避免意外修改其他参数
                    if k.startswith("net."):
                        # 使用 replace 并设置 count=1，确保只替换第一个匹配项
                        name = k.replace("net.", "net.module.", 1)
                        new_state_dict[name] = v
                    else:
                        # 对于其他非网络部分的参数（如果存在的话），保持原样
                        new_state_dict[k] = v
                incompatible_keys = self.policy.load_state_dict(new_state_dict, strict=False)
            else:
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            
            if self.local_rank < 1:
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

            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                if "scheduler_state" in ckpt_dict:
                    self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
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

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left, is_train):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            action = -100 # 默认为无效动作
            
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal", {"is_train": is_train})
            if curr_dis_to_goal < 1.5:
                action = 0 # 停止
            elif no_vp_left:
                action = -100 # 没有可选动作
            else:
                target_ghost_vp = None
                if self.config.IL.expert_policy == 'spl':
                    if not gmap.ghost_real_pos: # 安全检查：如果没有ghost节点
                        action = -100
                    else:
                        ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                        ghost_dis_to_goal = [self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1], "is_train": is_train}) for p in ghost_vp_pos]
                        target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                
                # (为简洁省略了 'ndtw' 的逻辑，您可以按同样模式添加安全检查)
                
                if target_ghost_vp:
                    # --- 【【【 关键修复 】】】 ---
                    # 安全检查：确保目标节点在当前动作空间中
                    if target_ghost_vp in gmap_vp_ids:
                        action = gmap_vp_ids.index(target_ghost_vp)
                    else:
                        # 如果因为某种原因找不到，也标记为无效动作
                        action = -100
            
            teacher_actions.append(action)
            
        return torch.tensor(teacher_actions, device=self.device) # 直接在GPU上创建


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

        # 下面变量requires_grad均为False
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

        # 下面变量中，batch_gmap_img_fts.requires_grad为True，其他均为False
        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left, 'gmap_task_embeddings': batch_gmap_task_embeddings
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts



    @staticmethod
    def _pause_envs(envs, observations, envs_to_pause):
        if len(envs_to_pause) > 0:
            # 从后往前暂停，避免索引错乱
            for idx in sorted(envs_to_pause, reverse=True):
                envs.pause_at(idx)

            # 只保留未暂停环境的 observations
            if observations:
                keep_mask = [i not in envs_to_pause for i in range(len(observations))]
                observations = [obs for i, obs in enumerate(observations) if keep_mask[i]]
            
        return envs, observations

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
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters # default: 15000
        log_every  = self.config.IL.log_every # default: 200
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')

        if self.config.local_rank < 1:
            config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
            with open(config_path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Configuration saved to {config_path}")
        
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0)) # interval表示当前这次循环需要进行几次iteration
            cur_iter = idx + interval # cur_iter表示当前这次循环完成后一共进行了几次iteration, idx表示当前这次循环完成前一共进行了几次iteration

            sample_ratio = self.config.IL.sample_ratio ** ((idx) // self.config.IL.decay_interval + 1)
            # if self.config.IL.warmup_iters > 0:
            #     if idx < self.config.IL.warmup_iters:
            #         sample_ratio = 1.0
            #     else:
            #         sample_ratio = self.config.IL.sample_ratio ** ((idx-self.config.IL.warmup_iters) // self.config.IL.decay_interval + 1)
            # else:
            #     sample_ratio = self.config.IL.sample_ratio ** ((idx) // self.config.IL.decay_interval + 1)
            if sample_ratio <= 0.15:
                sample_ratio = 0.0
            logger.info(f"sample ratio: {sample_ratio}")

            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', current_lr, cur_iter)
                logger.info(loss_str)
                logger.info(f"lr: {current_lr}")
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
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

        self.sap_loss = 0.
        for idx in pbar:
            # 下面是进行了一次rollout和训练
            self.optimizer.zero_grad()
            self.loss = 0. # 每次rollout重置一下self.loss
            
            # 上面的GradScaler和下面的autocast都是为了自动混合精度训练（Automatic Mixed Precision, AMP）而设置的
            # autocast使rollout过程模型切换为FP16形式（默认为FP16）；而GradScaler通过在计算反向传播之前对梯度进行缩放，避免梯度在 FP16 表示范围内下溢（变成 0）。
            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            # 每次rollout后都进行更新训练
            # 即使GPU_fast本地的梯度计算先结束，它的 loss.backward() 也需要等到GPU_slow的梯度贡献进来、所有 all_reduce 都完成后才能真正视为结束，也就是说每次loss.backward()都是一个同步点，是在DDP内部自动完成的
            # 在两个GPU上的 loss.backward() 都执行完毕后，每个GPU上的模型参数都拥有了完全相同的、经过全局平均的梯度；在 optimizer.step() 之后，两个GPU上的模型参数也保持完全一致。
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scheduler.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append('POSITION_INFER')
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            self.config.VIDEO_DIR = self.config.VIDEO_DIR + "_" + self.config.EVAL.SPLIT
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

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return

        if self.config.EVAL.fast_eval:
            episodes_allowed = self.traj[::5]
        elif self.config.EVAL.EPISODE_ID:
            episodes_allowed = self.config.EVAL.EPISODE_ID
        else:
            episodes_allowed = self.traj
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=episodes_allowed,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        # 如果设定的传感器尺寸就等于CenterCropperPerSensor设定的尺寸，那么就不会做任何操作（实际上也是如此）
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        # while len(self.stat_eps) < eps_to_eval:
        while len(self.stat_eps) < eps_to_eval:
            # print(len(self.stat_eps.keys()))
            self.rollout('eval')

        self.envs.close()
        # print("stat_eps\n", self.stat_eps.keys())
        # for key, value in self.stat_eps.items():
        #     print(f"episode id: {key}, high level steps taken: {value['high_level_step']}")
        if self.world_size > 1:
            distr.barrier() # 阻塞组内的所有进程，直到组内的每个进程都调用了这个函数，然后所有进程才能继续执行后续代码。
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes # 当前进程内的所有metric结果的平均值。指标：平均值
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0) # 从所有进程中收集名为 total 的张量，对它们进行某种归约操作（默认为求和），并将最终的结果存储在目标进程（dst 指定的进程）上。（阻塞操作，同步所有进程）
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda() # 把平均数恢复为总和
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f: # 存储每一个episode的运行情况，每个local_rank各自存储各自所运行的那些个episodes
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f: # 存储一个总的结果，即eval数据集平均下来的成功率、spl等
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            print(f"Episodes evaluated: {total}")

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        # _C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
        # _C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
        #     ("rgb", (224, 224)),
        #     ("depth", (256, 256)),
        # ]
        # _C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR = CN()
        # _C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = [
        #     ("rgb", (224, 298)),
        #     ("depth", (256, 341)),
        # ]
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
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
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS # 总共的gpu数目
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        # 按照GPU_NUMBERS进行episodes的平均分配，self.traj是分配到当前gpu下的所有episodes编号list，编号以str的形式
        self.traj = self.collect_infer_traj()
        
        # 这里返回的envs就是当前gpu下启动的所有envs，数量就是NUM_ENVIRONMENTS
        # config.TASK_CONFIG.DATASET.EPISODES_ALLOWED是通过多个gpu之间平均分配所有的episodes实现的,对于每一个具体的env,该参数是一样的；
        # config.TASK_CONFIG.DATASET.CONTENT_SCENES是选取的当前gpu下全部episodes会用到的所有scenes,再将其平均分配给每个envs,对于每一个具体的env,该参数会不一样;
        # 也就是说,最终的效果就是对于每一个env,它只运行所分配到的scenes下所有分配到当前gpu的episodes,没有冗余的运行.
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME), # VLNCEDaggerEnv
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        # 加入下面代码也不影响运行,但是否影响性能有待验证
        # self.config.defrost()
        # self.config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        # self.config.freeze()
        # print("------------------self.config------------------------", self.config)

        # ******************这段代码意义不明******************
        # self.envs.observation_spaces是长度为envs数量的列表
        # 其中每一个元素都是25长度的Box字典,包含了12个rgb,12个depth等
        obs_transforms = get_active_obs_transforms(self.config) # CenterCropperPerSensor,crop的尺寸设置的跟本身的rgb/depth的尺寸一致,似乎不起作用?
        # 如果设定的传感器尺寸就等于CenterCropperPerSensor设定的尺寸，那么就不会做任何操作（实际上也是如此）
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        # ******************这段代码意义不明******************

        # ActionSpace(HIGHTOLOW:EmptySpace(), MOVE_FORWARD:EmptySpace(), STOP:EmptySpace(), TURN_LEFT:EmptySpace(), TURN_RIGHT:EmptySpace())
        # self.envs.action_spaces是长度为4的列表
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list) # 以list为value的字典
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori


    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        # --- 1. 初始化 ---
        value_loss_coef = self.config.IL.get("value_loss_coef", 0.0)
        gamma = self.config.IL.get("gamma", 0.99)
        is_train = (mode == 'train')
        
        # 根据模式确定反馈类型
        if is_train:
            # SFT 阶段，我们始终跟随教师的决策来生成轨迹
            feedback = 'teacher'
        else: # eval or infer
            feedback = 'argmax'

        self.envs.resume_all()
        observations = self.envs.reset()

        # --- 环境暂停逻辑 (处理已完成的 episodes) ---
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) if ep.episode_id in self.stat_eps]
            if env_to_pause:
                self.envs, observations = self._pause_envs(self.envs, observations, env_to_pause)
            if self.envs.num_envs == 0: return
                
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) if ep.episode_id in self.path_eps]
            if env_to_pause:
                self.envs, observations = self._pause_envs(self.envs, observations, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id, k = curr_eps[i].episode_id, curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # --- 2. 准备初始输入 ---
        instr_max_len = self.config.IL.max_text_len
        instr_pad_id = 1
        task_type = 1 if self.config.MODEL.task_type == 'r2r' else 2
        instruction_sensor_uuid = self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        
        observations = extract_instruction_tokens_new_30(observations, instruction_sensor_uuid, max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        with torch.no_grad():
            all_txt_ids = batch['instruction']
            all_txt_task_encoding = batch['txt_task_encoding']
            all_txt_masks = (all_txt_ids != instr_pad_id)
            all_txt_embeds = self.policy.net(mode='language', txt_ids=all_txt_ids, txt_task_encoding=all_txt_task_encoding, txt_masks=all_txt_masks)

        # --- 3. 初始化 ---
        if is_train:
            self.loss = 0.
            trajectories = defaultdict(lambda: {'values': [], 'rewards': [], 'dones': []})

        self.gmaps = [GraphMap(True, self.config.IL.loc_noise, self.config.MODEL.merge_ghost, self.config.IL.ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        active_ep_ids = [ep.episode_id for ep in self.envs.current_episodes()]

        # --- 4. 轨迹生成循环 ---
        for stepk in range(self.max_len):
            if self.envs.num_envs == 0: break
                
            # --- 模型前向传播 ---
            with torch.set_grad_enabled(is_train):
                wp_outputs = self.policy.net(mode="waypoint", waypoint_predictor=self.waypoint_predictor, observations=batch, in_train=(is_train and self.config.IL.waypoint_aug))
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({'mode': 'panorama'})
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(pano_masks, 1, keepdim=True)
                cur_pos, cur_ori = self.get_pos_ori()
                cur_vp, cand_vp, cand_pos, cand_real_pos = [], [], [], []
                for i in range(self.envs.num_envs):
                    _cur_vp, _cand_vp, _cand_pos = self.gmaps[i].identify_node_new(cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    cur_vp.append(_cur_vp); cand_vp.append(_cand_vp); cand_pos.append(_cand_pos)
                    _cand_real_pos = [self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis}) for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])]
                    cand_real_pos.append(_cand_real_pos)
                    self.gmaps[i].update_graph(prev_vp[i], stepk+1, _cur_vp, cur_pos[i], avg_pano_embeds[i], _cand_vp, _cand_pos, pano_embeds[i][vp_inputs['nav_types'][i]==1], _cand_real_pos)
                
                nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type)
                no_vp_left = nav_inputs.pop('no_vp_left')
                nav_inputs.update({'mode': 'navigation', 'txt_embeds': all_txt_embeds, 'txt_masks': all_txt_masks})
                nav_outs = self.policy.net(**nav_inputs)

            # --- 获取教师动作并即时计算模仿损失 ---
            teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left, is_train)
            
            if is_train and ml_weight > 0:
                imitation_loss = F.cross_entropy(nav_outs['global_logits'], teacher_actions, ignore_index=-100)
                self.loss += ml_weight * imitation_loss
                self.logs['IL_loss'].append(imitation_loss.item())
            
            # --- 执行动作 ---
            a_t = teacher_actions if feedback == 'teacher' else nav_outs['global_logits'].argmax(dim=-1)
            
            # --- 准备 env_actions (完整版) ---
            env_actions = []; cpu_a_t = a_t.cpu().numpy()
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_outs['global_logits'][i, 0].data.item()
                if cpu_a_t[i] == 0 or cpu_a_t[i] == -100 or stepk == self.max_len - 1 or no_vp_left[i]:
                    vp_stop_scores = list(gmap.node_stop_scores.items())
                    if vp_stop_scores:
                        stop_vp = max(vp_stop_scores, key=lambda x: x[1])[0]
                    else:
                        stop_vp = cur_vp[i]
                    
                    if stop_vp in gmap.node_pos: stop_pos = gmap.node_pos[stop_vp]
                    else: stop_vp = cur_vp[i]; stop_pos = cur_pos[i]
                    
                    back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]][1:] if self.config.IL.back_algo == 'control' and cur_vp[i] in gmap.shortest_path and stop_vp in gmap.shortest_path[cur_vp[i]] else None
                    vis_info = {'nodes': list(gmap.node_pos.values()), 'ghosts': list(gmap.ghost_aug_pos.values()), 'predict_ghost': stop_pos}
                    env_actions.append({'action': {'act': 0, 'cur_vp': cur_vp[i], 'stop_vp': stop_vp, 'stop_pos': stop_pos, 'back_path': back_path, 'tryout': use_tryout}, 'vis_info': vis_info})
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    vis_info = None
                    back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]][1:] if self.config.IL.back_algo == 'control' and cur_vp[i] in gmap.shortest_path and front_vp in gmap.shortest_path[cur_vp[i]] else None
                    env_actions.append({'action': {'act': 4, 'cur_vp': cur_vp[i], 'front_vp': front_vp, 'front_pos': front_pos, 'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos, 'back_path': back_path, 'tryout': use_tryout}, 'vis_info': vis_info})
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost: gmap.delete_ghost(ghost_vp)
            
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # --- 5. 记录 Critic 训练所需数据 ---
            if is_train and value_loss_coef > 0:
                for i in range(self.envs.num_envs):
                    ep_id = active_ep_ids[i]
                    reward = -0.01 + (5.01 if dones[i] else 0.0)
                    trajectories[ep_id]['values'].append(nav_outs['value'][i])
                    trajectories[ep_id]['rewards'].append(reward)
                    trajectories[ep_id]['dones'].append(dones[i])

            # --- 6. 指标计算与环境管理 ---
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]: continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    # gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / (stepk + 1)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    # dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    # metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    # metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    metric['high_level_step'] = stepk
                    self.stat_eps[ep_id] = metric
                    if self.pbar: self.pbar.update()
            
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']: # 只记录前进时，转向不记录
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            if sum(dones) > 0:
                envs_to_pause = [i for i, d in enumerate(dones) if d]
                active_ep_ids = [eid for i, eid in enumerate(active_ep_ids) if i not in envs_to_pause]
                self.gmaps = [gmap for i, gmap in enumerate(self.gmaps) if i not in envs_to_pause]
                prev_vp = [vp for i, vp in enumerate(prev_vp) if i not in envs_to_pause]
                
                for idx in sorted(envs_to_pause, reverse=True):
                    self.envs.pause_at(idx)
                
                observations = [obs for i, obs in enumerate(observations) if i not in envs_to_pause]
                to_keep_indices = [i for i in range(len(all_txt_embeds)) if i not in envs_to_pause]
                all_txt_embeds = all_txt_embeds[to_keep_indices]
                all_txt_masks = all_txt_masks[to_keep_indices]
                
            if self.envs.num_envs == 0: break
            
            # --- 为下一步准备 batch ---
            observations = extract_instruction_tokens_new_30(observations, instruction_sensor_uuid, max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # --- 7. 轨迹结束后，计算 Critic 损失 ---
        if is_train and value_loss_coef > 0:
            value_loss = 0.0
            num_valid_traj = 0
            
            for ep_id, traj in trajectories.items():
                if not traj['values']: continue
                
                returns = 0.0
                traj_returns = [0.0] * len(traj['values'])
                for t in reversed(range(len(traj['rewards']))):
                    returns = traj['rewards'][t] + gamma * returns * (1.0 - traj['dones'][t])
                    traj_returns[t] = returns
                
                values_tensor = torch.stack(traj['values']).squeeze(-1)
                returns_tensor = torch.tensor(traj_returns, device=self.device, dtype=values_tensor.dtype).detach()
                
                value_loss += F.mse_loss(values_tensor, returns_tensor)
                num_valid_traj += 1

            if num_valid_traj > 0:
                avg_value_loss = value_loss / num_valid_traj
                self.loss += value_loss_coef * avg_value_loss
                self.logs['SFT_value_loss'].append(avg_value_loss.item())
