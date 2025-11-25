#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry





import habitat_extensions  # noqa: F401
# 通过下面这行代码调用vlnce_baselines/__init__.py，而__init__.py中通过import具体模块实现了注册
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
# from vlnce_baselines.nonlearning_agents import (
#     evaluate_agent,
#     nonlearning_inference,
# )

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    # local_rank是在这里被传入程序，应该是自动的传参
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    # 这里得到的config融合了以下渠道的配置信息：default.py中的默认配置（包含vlnce_baselines中的和habitat_baselines中的和habitat_extensions中的和habitat/config中的）、
    # iter_train和r2r_vlnce两个yaml文件中的配置、bash中的额外参数配置
    config = get_config(exp_config, opts)
    # print("here\n", config.TASK_CONFIG)
    config.defrost()

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if run_type == "train":
        config.TENSORBOARD_DIR = config.CHECKPOINT_FOLDER
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.RESULTS_DIR += '/eval_results/'
    config.VIDEO_DIR += exp_name
    # config.TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = config.IL.max_text_len
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    # # 默认设置中policy_name使用的是PolicyViewSelectionETP
    # if 'CMA' in config.MODEL.policy_name and 'r2r' in config.BASE_TASK_CONFIG_PATH:
    #     print("-----------------USING CMA-----------------")
    #     config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz'
    # else:
    #     print("-----------------NO USING CMA-----------------")

    config.local_rank = local_rank # 在这里将环境变量local_rank传到了config里
    config.freeze()
    os.system("mkdir -p data/logs/running_log")
    # logger.add_filehandler('data/logs/running_log/'+config.LOG_FILE)
    os.makedirs('data/logs/checkpoints/'+exp_name, exist_ok=True)
    if run_type == "train":
        logger.add_filehandler('data/logs/checkpoints/'+exp_name+'/'+config.LOG_FILE)
    else:
        logger.add_filehandler('data/logs/running_log/'+config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        # 设置 PyTorch 在 CPU 上进行并行计算时使用的线程数，对于全由cuda进行计算的情况似乎不起作用
        # 但一旦有先得到cpu下的tensor再转到cuda中这个过程，那么就代笔着cpu参与了张量计算
        torch.set_num_threads(1)

    # if run_type == "eval" and config.EVAL.EVAL_NONLEARNING:
    #     evaluate_agent(config)
    #     return

    # if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
    #     nonlearning_inference(config)
    #     return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME) # config.TRAINER_NAME=SS-ETP
    print("trainer_init\n", trainer_init) #  <class 'vlnce_baselines.ss_trainer_ETP.RLTrainer'>
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    # import pdb; pdb.set_trace()
    
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()

if __name__ == "__main__":
    main()
