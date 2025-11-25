# ETP-R1 实车部署代码执行全流程详解

本文档详细梳理了基于 `ros_inference` 工程架构的 Sim-to-Real 代码执行流。

## 1. 系统启动阶段 (Initialization Phase)

此阶段负责加载配置、初始化 ROS 节点、加载模型权重并建立硬件连接。

### 1.1 启动脚本
* **入口**: 用户在终端运行 `bash run_robot.bash`。
* **操作**: 
    * 设置环境变量 `CUDA_VISIBLE_DEVICES=0`。
    * 指定配置文件路径 `vlnce_baselines/config/r2r_configs/real_robot.yaml`。
    * 调用 Python 入口：`python run_ros.py $flag_robot`。

### 1.2 Python 入口 (`run_ros.py`)
* **ROS 节点**: 调用 `rospy.init_node("etp_r1_agent")`，注册为 ROS Master 的一个节点。
* **注册模块**: 导入 `ros_env` 和 `my_ss_trainer_ETP_30`，触发 `@baseline_registry.register` 装饰器。
* **配置加载**: 解析 YAML 文件，强制设置 `NUM_ENVIRONMENTS=1` 和 `local_rank=0`。
* **Trainer 实例化**: 初始化 `RLTrainer` 类（位于 `my_ss_trainer_ETP_30.py`）。
* **开始推理**: 调用 `trainer.inference()`，进入主逻辑。

### 1.3 环境与模型构建 (`my_ss_trainer_ETP_30.py`)
在 `inference()` 函数内部：
1.  **环境构建**: 调用 `construct_envs`。由于配置指定 `ENV_NAME: SeekerROSEnv`，代码实例化 `vlnce_baselines/common/ros_env.py` 中的 `SeekerROSEnv` 类。
    * **硬件连接**: `SeekerROSEnv` 初始化时会启动 `SeekerSubscriber`（开始订阅 `/seeker/...`）和 `LimoController`（开始订阅 `/odom`）。此时代码会**阻塞**，直到接收到第一帧 ROS 数据。
2.  **策略加载**: 调用 `_initialize_policy`。
    * 加载 **Pretrained Backbone** (`pretrained/model_step_367500.pt`)。
    * 加载 **DINOv2** (`pretrained/dinov2_vitb14_reg4_pretrain.pth`)。
    * 加载 **Waypoint Predictor** (`data/wp_pred/check_val_best_seek_dino`)。
    * 加载 **Policy Weights** (`data/checkpoints/ckpt.iter200.pth`)。

---

## 2. 运行循环阶段 (Run Loop)

系统启动后，进入 `trainer.rollout()` 函数，这是一个持续的 `while` 循环。

### 流程图概览

```mermaid
graph TD
    HW[硬件层: Seeker & Limo] -->|ROS Topic| DRV[驱动层: ros_utils.py]
    DRV -->|Dict (Numpy)| ENV[环境层: ros_env.py]
    ENV -->|Batch (Tensor)| TRN[训练器: my_ss_trainer.py]
    TRN -->|Forward| MDL[模型层: Policy & Net]
    MDL -->|Logits| TRN
    TRN -->|Action Dict| ENV
    ENV -->|Target (x,y)| DRV
    DRV -->|cmd_vel| HW


```

第一步：相机输入 (哪里获取数据？)

物理层：Seeker 相机发布 8 个 ROS 话题。

接收层 (ros_utils.py - SeekerSubscriber 类)：

message_filters 同步接收 4路 RGB 和 4路 Depth。

预处理：这里强制将 RGB Resize 成 224x224，Depth Resize 成 256x256。这是数据进入系统的第一站。

封装层 (ros_env.py - _get_obs 方法)：

将处理好的图像打包成字典，并塞入当前的“文本指令” (instruction)。

Trainer层 (my_ss_trainer_ETP_30.py - batch_obs):

将数据转换为 PyTorch Tensor 并送入 GPU。

第二步：模型思考 (哪里输出动作？)
输入：Tensor 格式的图像 + 指令。

模型推理 (My_Policy_ViewSelection_ETP_30.py):

数据流经 vln_bert、rgb_encoder (DINOv2) 等。

路点预测：waypoint_predictor 预测周围哪些方位可以走（生成 Ghost Nodes）。

决策：模型给每个候选节点打分 (nav_logits)。

动作选择 (my_ss_trainer_ETP_30.py):

代码根据分数选择概率最大的那个节点 ID (Argmax)。

关键转换：Trainer 根据选中的节点 ID，从拓扑图 (gmap) 中提取出该节点的物理坐标 (Ghost Pos)。

输出：生成一个动作字典 env_actions，例如 {'act': 4, 'ghost_pos': [3.5, 0, 1.2], ...}。

第三步：小车控制 (哪里控制小车？)

传递：Trainer 调用 self.envs.step(env_actions)。

适配层 (ros_env.py - step 方法):

接收动作字典。它看到 act: 4 (导航模式)，提取出目标坐标 target_x, target_y。

调用驱动：执行 self.ctl.navigate_to(target_x, target_y)。

驱动层 (ros_utils.py - LimoController 类):

这是控制发生的具体位置。

它在一个 while 循环中，不断计算当前 /odom 位置和目标点的误差。

PID/逻辑控制：计算出需要的线速度 linear.x 和角速度 angular.z。

发送指令：self.vel_pub.publish(cmd) 将速度指令发送给 /cmd_vel 话题，驱动 Limo 物理运动。

第四步：闭环

小车移动到位后，Maps_to 函数返回。ros_env.step 读取新位置的相机图像，返回给 Trainer，开始下一轮循环。

