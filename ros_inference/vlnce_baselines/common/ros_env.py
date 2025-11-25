import time
import numpy as np
import habitat
from habitat import Config, Dataset
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.common.ros_utils import SeekerSubscriber, LimoController

# 注册名为 SeekerROSEnv，对应 real_robot.yaml
@baseline_registry.register_env(name="SeekerROSEnv")
class SeekerROSEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Dataset = None):
        # 不初始化父类，避免启动 Habitat Sim
        self._config = config
        self._dataset = dataset
        
        # 简单的数据集迭代器
        self._episode_iterator = iter(dataset.episodes) if dataset else iter([])
        self._current_episode = None

        print("[SeekerROSEnv] Connecting to ROS...")
        self.sub = SeekerSubscriber()
        self.ctl = LimoController()
        time.sleep(2) # 等待 ROS 连接

        # 占位符空间 (Shape 必须对)
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=10.0, shape=(256, 256, 1), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(5)
        self.episode_over = False

    @property
    def current_episode(self):
        return self._current_episode

    @property
    def episodes(self):
        return self._dataset.episodes if self._dataset else []

    def reset(self):
        self.episode_over = False
        # 获取下一个 Episode (主要是为了拿 Instruction)
        try:
            self._current_episode = next(self._episode_iterator)
        except StopIteration:
            self._episode_iterator = iter(self._dataset.episodes)
            self._current_episode = next(self._episode_iterator)
        
        print(f"\n[Reset] New Goal: {self._current_episode.episode_id}")
        print(f"[Instr] {self._current_episode.instruction.instruction_text}")
        
        return self._get_obs()

    def step(self, action):
        """
        Action 格式是 my_ss_trainer 中定义的字典:
        {'action': {'act': int, 'ghost_pos': [x,y,z], 'stop_pos': ...}}
        """
        # 提取内部 action 字典
        real_action = action['action'] 
        act_idx = real_action['act']

        if act_idx == 0: # STOP
            print("[Step] Action: STOP")
            self.ctl.stop()
            self.episode_over = True
            # 如果有 stop_pos 修正，可在此调用 navigate_to

        elif act_idx == 4: # HIGHTOLOW (Navigate)
            # 实车忽略 back_path (回溯)，直接去 ghost_pos (目标点)
            # Sim 坐标: [x, y(up), z] -> ROS: [x, y]
            # 假设 Trainer 输出的是 Habitat 坐标，我们需要提取 x, z 作为实车的 x, y
            target = real_action['ghost_pos']
            target_x_ros = target[0]
            target_y_ros = target[2] # Habitat Z maps to ROS Y (平面)
            
            print(f"[Step] Navigating to ROS ({target_x_ros:.2f}, {target_y_ros:.2f})")
            self.ctl.navigate_to(target_x_ros, target_y_ros)
        
        else:
            print(f"[Step] Ignored action index: {act_idx}")

        obs = self._get_obs()
        return obs, 0.0, self.episode_over, {"position": self.get_pos_ori()[0]}

    def _get_obs(self):
        # 阻塞等待最新图像
        ros_data = None
        while ros_data is None:
            ros_data = self.sub.get_observation()
            if ros_data is None: time.sleep(0.1)
        
        # 构造符合 Trainer 预期的字典
        # 必须包含 'instruction'
        obs = ros_data # 已经包含 rgb_front, depth_front 等
        obs['instruction'] = {
            'text': self._current_episode.instruction.instruction_text,
            'tokens': self._current_episode.instruction.instruction_tokens,
            'trajectory_id': self._current_episode.episode_id
        }
        return obs

    # --- 模拟 Habitat 接口 ---

    def get_pos_ori(self):
        """
        返回 Habitat 格式的位姿:
        Pos: [x, y, z] (y is up) -> 对应 ROS [x, 0, y]
        Ori: Quaternion [imag_x, imag_y, imag_z, real_w]
        """
        rx, ry, ryaw = self.ctl.get_current_pose()
        
        # ROS (x,y) -> Habitat (x,z), y=0 (flat ground)
        pos = np.array([rx, 0.0, ry], dtype=np.float32)
        
        # Yaw -> Quaternion [0, sin(y/2), 0, cos(y/2)] ? 
        # Habitat: Y-up rotation. ROS: Z-up.
        # 这里简化处理，假设 Trainer 也是 2D 导航，只关心相对角度
        # 构造一个标准的四元数 [x, y, z, w]
        quat = np.array([0.0, np.sin(ryaw/2), 0.0, np.cos(ryaw/2)]) 
        # 原代码期望: [imag, real] -> [x,y,z,w] 顺序需匹配 graph_utils
        
        return pos, quat

    def get_cand_real_pos(self, forward, angle):
        """ 纯几何预测，不移动 """
        curr_x, curr_y, curr_yaw = self.ctl.get_current_pose()
        tgt_yaw = curr_yaw - angle # 注意坐标系旋转方向可能相反，需实测
        
        nx = curr_x + forward * np.cos(tgt_yaw)
        ny = curr_y + forward * np.sin(tgt_yaw)
        
        return np.array([nx, 0.0, ny], dtype=np.float32)

    # 兼容 VectorEnv 的调用
    def call_at(self, i, func_name, args=None):
        if args is None: args = {}
        return getattr(self, func_name)(**args)
        
    def current_dist_to_goal(self, *args, **kwargs): return 10.0
    def point_dist_to_goal(self, *args, **kwargs): return 10.0