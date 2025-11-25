#!/bin/bash

# 设置日志级别，减少 Habitat 和 Magnum 的冗余输出
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# 设置机器人上的显卡编号（通常是 0）
export CUDA_VISIBLE_DEVICES=0

# 定义参数
# 注意事项：
# 1. --exp-config 指向我们新建的 real_robot.yaml (路径相对于 ros_inference 根目录)
# 2. 确保 CKPT_PATH 指向你 data/checkpoints 下的文件
# 3. 确保 MODEL.pretrained_path 指向 pretrained/ 下的文件
# 4. 即使配置文件里写了路径，这里的命令行参数优先级更高，会覆盖配置文件

flag_robot="--exp_name limo_inference_test
      --run-type inference
      --exp-config config/real_robot_inference.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/checkpoints/ckpt.iter200.pth
      INFERENCE.PREDICTIONS_FILE data/predictions_robot.json
      IL.back_algo control
      MODEL.pretrained_path pretrained/model_step_367500.pt
      "

# 启动模式
echo "###### Robot Inference Mode ######"
echo "Config: config/real_robot.yaml"
echo "Checkpoint: data/checkpoints/ckpt.iter200.pth"

# 关键修改：
# 1. 不再使用 torch.distributed.launch (单机运行)
# 2. 调用 run_ros.py (对应我们新建的入口文件)
python run_ros.py $flag_robot