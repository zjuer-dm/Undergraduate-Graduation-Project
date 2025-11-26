修改 vlnce_baselines/common/ros_utils.py，用一个虚拟控制器（Mock Controller） 替换掉真实的 LimoController

怎么确认到达了目标点？

怎么确认到了最终的目标点？

微小偏差：由 LimoController 的 PID 算法实时修正（它就是干这个的）。

巨大偏差/被迫停车：由 ros_env.get_pos_ori() 的 真值回传 机制处理。模型永远基于机器人脚下踩着的真实坐标来更新地图和规划下一步。

终端 1 (ROS 驱动)：
roslaunch seeker 4depth_image.launch

终端 2 (确认话题 - 仅第一次需要)：
rostopic list

# 检查 RGB 和 Depth 话题名，如有不同，修改 ros_utils.py
终端 3 (推理)：
./run_ros.bash
