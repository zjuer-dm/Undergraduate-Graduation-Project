import time
import numpy as np
import habitat
from habitat import Config, Dataset
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.common.ros_utils import SeekerSubscriber, LimoController

# 定义一个简单的 Dummy 类来模拟 Episode 对象，骗过 Trainer
class InteractiveEpisode:
    def __init__(self, instruction_text, episode_id="0"):
        self.episode_id = episode_id
        self.scene_id = "real_world"
        self.instruction = self.Instruction(instruction_text)
        self.goals = [] # 实车没有真值目标

    class Instruction:
        def __init__(self, text):
            self.instruction_text = text
            self.instruction_tokens = None # Trainer 会自动处理 Tokenization

@baseline_registry.register_env(name="SeekerROSEnv")
class SeekerROSEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Dataset = None):
        self._config = config
        
        print("\n" + "="*50)
        print("[SeekerROSEnv] Initializing Hardware...")
        self.sub = SeekerSubscriber()
        # 如果是无底盘测试，这里的 LimoController 会自动变成 Mock 版
        self.ctl = LimoController() 
        time.sleep(1) 
        print("[SeekerROSEnv] Hardware Ready.")
        print("="*50 + "\n")

        # 定义 Observation Space 占位符
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=10.0, shape=(256, 256, 1), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(5)
        self.episode_over = False
        self._current_episode = None

    @property
    def current_episode(self):
        return self._current_episode

    @property
    def episodes(self):
        return []

    def reset(self):
        self.episode_over = False
        
        # ===========================================================
        # 【核心修改】 交互式输入指令
        # ===========================================================
        print("\n" + "!"*60)
        print("WAITING FOR INSTRUCTION...")
        # 阻塞等待用户输入指令
        user_text = input(">>> Please type navigation instruction (e.g., 'Go forward'): ")
        print(f"Received: \"{user_text}\"")
        print("!"*60 + "\n")
        
        # 构建虚拟 Episode
        self._current_episode = InteractiveEpisode(user_text, episode_id=str(int(time.time())))
        
        # 归位控制器（重置虚拟坐标或里程计）
        # self.ctl.reset() # 如果 Controller 有 reset 方法可以调用
        
        return self._get_obs()

    def step(self, action):
        real_action = action['action'] 
        act_idx = real_action['act']
        info = {"position": self.get_pos_ori()[0]}

        if act_idx == 0: # STOP
            print(f"[Step] ===> DECISION: STOP")
            self.ctl.stop()
            self.episode_over = True

        elif act_idx == 4: # NAVIGATE
            target = real_action['ghost_pos']
            # 坐标映射: Habitat (x, z) -> ROS (x, y)
            target_x_ros = target[0]
            target_y_ros = target[2] 
            
            print(f"[Step] ===> DECISION: Move to ({target_x_ros:.2f}, {target_y_ros:.2f})")
            self.ctl.navigate_to(target_x_ros, target_y_ros)
        
        else:
            print(f"[Step] Ignored action: {act_idx}")

        obs = self._get_obs()
        return obs, 0.0, self.episode_over, info

    def _get_obs(self):
        ros_data = None
        while ros_data is None:
            ros_data = self.sub.get_observation()
            if ros_data is None: time.sleep(0.1)
        
        obs = ros_data 
        # 将刚才输入的指令注入到 observation 中
        obs['instruction'] = {
            'text': self._current_episode.instruction.instruction_text,
            'tokens': None, # Trainer 会处理
            'trajectory_id': self._current_episode.episode_id
        }
        return obs

    # --- 兼容接口 ---
    def get_pos_ori(self):
        rx, ry, ryaw = self.ctl.get_current_pose()
        # ROS [x, y] -> Habitat [x, 0, y] (y is up)
        pos = np.array([rx, 0.0, ry], dtype=np.float32)
        quat = np.array([0.0, np.sin(ryaw/2), 0.0, np.cos(ryaw/2)]) 
        return pos, quat

    def get_cand_real_pos(self, forward, angle):
        curr_x, curr_y, curr_yaw = self.ctl.get_current_pose()
        tgt_yaw = curr_yaw - angle 
        nx = curr_x + forward * np.cos(tgt_yaw)
        ny = curr_y + forward * np.sin(tgt_yaw)
        return np.array([nx, 0.0, ny], dtype=np.float32)

    def call_at(self, i, func_name, args=None):
        if args is None: args = {}
        return getattr(self, func_name)(**args)
        
    def current_dist_to_goal(self, *args, **kwargs): return 10.0
    def point_dist_to_goal(self, *args, **kwargs): return 10.0

