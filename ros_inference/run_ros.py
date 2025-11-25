#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
import rospy
import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.config.default import get_config

# [关键修改 1] 补全这个 import，防止注册表报错
import habitat_extensions 
# 注册我们的新模块
import vlnce_baselines.common.ros_env 
import vlnce_baselines.my_ss_trainer_ETP_30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # 初始化 ROS 节点
    rospy.init_node("etp_r1_agent", anonymous=True)
    
    config = get_config(args.exp_config, args.opts)
    
    # 强制单环境配置
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.SIMULATOR_GPU_IDS = [0]
    config.ENV_NAME = "SeekerROSEnv"
    
    # [关键修改 2] 手动注入 local_rank
    # my_ss_trainer_ETP_30.py 强依赖这个属性来判断是否为主进程
    config.local_rank = 0 
    
    config.freeze()

    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    print(f"Trainer: {config.TRAINER_NAME}")
    trainer_cls = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_cls(config)

    print("Starting ROS Inference Loop...")
    # 进入 Trainer 的 inference 方法 -> 调用 construct_envs -> 初始化 SeekerROSEnv
    trainer.inference()

if __name__ == "__main__":
    main()