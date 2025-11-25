import rospy
import message_filters
import numpy as np
import cv2
import threading
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from PIL import Image as PILImage

class SeekerSubscriber:
    """
    负责从 ROS 订阅 Seeker 相机的 4路 RGB 和 4路 Depth 数据，
    并进行时间同步和预处理，使其格式与 Habitat 仿真一致。
    """
    def __init__(self, config):
        self.config = config
        self.bridge = CvBridge()
        self.mutex = threading.Lock()
        self.latest_obs = None
        
        # 定义 View 与 ROS Topic 的映射 (根据 Seeker 手册和通用命名推断)
        # 请根据实际运行 `rostopic list` 的结果修改这里
        self.rgb_topics = {
            'front': '/seeker/front/undistort/image_raw', 
            'right': '/seeker/right/undistort/image_raw',
            'back':  '/seeker/back/undistort/image_raw',
            'left':  '/seeker/left/undistort/image_raw'
        }
        self.depth_topics = {
            'front': '/front/depth/image_raw', # Seeker手册第8页提及 [cite: 1]
            'right': '/right/depth/image_raw',
            'back':  '/back/depth/image_raw',
            'left':  '/left/depth/image_raw'
        }

        # 设置图像预处理尺寸 (需与 training config 一致)
        # 假设 ETP-R1 训练时 RGB 224x224, Depth 256x256
        self.rgb_size = (224, 224) 
        self.depth_size = (256, 256) 

        # 初始化 Subscriber
        subs = []
        # 为了保证同步，我们按 Front, Right, Back, Left 的顺序订阅 RGB 和 Depth
        # 注意：8路强同步可能会很困难，如果丢帧严重，需要改为“软同步”策略
        for view in ['front', 'right', 'back', 'left']:
            subs.append(message_filters.Subscriber(self.rgb_topics[view], Image))
        for view in ['front', 'right', 'back', 'left']:
            subs.append(message_filters.Subscriber(self.depth_topics[view], Image))

        # 使用 ApproximateTimeSynchronizer 进行时间同步
        # slop=0.1 表示允许 0.1s 的时间误差
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.2)
        self.ts.registerCallback(self.callback)
        
        print(f"[SeekerSubscriber] Waiting for topics: {list(self.rgb_topics.values())}")

    def callback(self, f_rgb, r_rgb, b_rgb, l_rgb, f_dep, r_dep, b_dep, l_dep):
        """
        回调函数，接收同步后的 8 帧图像
        """
        with self.mutex:
            raw_data = {
                'rgb':   {'front': f_rgb, 'right': r_rgb, 'back': b_rgb, 'left': l_rgb},
                'depth': {'front': f_dep, 'right': r_dep, 'back': b_dep, 'left': l_dep}
            }
            self.latest_obs = self._process_frames(raw_data)

    def _process_frames(self, raw_data):
        """
        将 ROS Image 转换为 Numpy/Tensor 格式，并Resize
        """
        obs = {}
        views = ['front', 'right', 'back', 'left']
        
        for view in views:
            # --- 处理 RGB ---
            # encoding="bgr8" 转为 OpenCV 格式
            cv_rgb = self.bridge.imgmsg_to_cv2(raw_data['rgb'][view], desired_encoding="bgr8")
            # 转换为 RGB 顺序 (Habitat 通常使用 RGB)
            cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)
            # Resize
            cv_rgb = cv2.resize(cv_rgb, self.rgb_size)
            # 归一化 (0-255 -> 0-1) ? Habitat 内部通常处理 uint8，这里保持 uint8 
            # 此时 shape 为 (H, W, 3)
            obs[f'rgb_{view}'] = cv_rgb

            # --- 处理 Depth ---
            # encoding="passthrough" (通常是 16UC1 毫米单位 或 32FC1 米单位)
            cv_dep = self.bridge.imgmsg_to_cv2(raw_data['depth'][view], desired_encoding="passthrough")
            
            # Seeker 手册提及深度图精度精度±5% @3m [cite: 34]，通常是 mm (uint16) 或 m (float)
            # 假设输入是 mm (uint16)，需要转为 m (float)
            if cv_dep.dtype == np.uint16:
                cv_dep = cv_dep.astype(np.float32) / 1000.0
            
            # Resize (注意：深度图 resize 最好用 NEAREST 或特定的插值，防止产生虚假深度)
            cv_dep = cv2.resize(cv_dep, self.depth_size, interpolation=cv2.INTER_NEAREST)
            
            # Habitat 期望深度图 shape 为 (H, W, 1)
            if len(cv_dep.shape) == 2:
                cv_dep = cv_dep[:, :, np.newaxis]
                
            obs[f'depth_{view}'] = cv_dep

        return obs

    def get_observation(self):
        """供 Environment 调用，返回最新的观测字典"""
        with self.mutex:
            if self.latest_obs is None:
                rospy.logwarn_throttle(2.0, "[SeekerSubscriber] No data received yet!")
                return None
            return self.latest_obs.copy()


class LimoController:
    """
    负责控制 Limo 小车移动到指定的全局坐标 (Waypoint)。
    基于 /odom 反馈进行闭环 P-Control。
    """
    def __init__(self):
        # 订阅里程计 /odom 
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        # 发布控制指令 /cmd_vel 
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.curr_pos = None # [x, y]
        self.curr_yaw = None # rad
        self.mutex = threading.Lock()
        
        # 控制参数 (根据 Limo 实车响应调整)
        self.k_rho = 0.5   # 距离增益
        self.k_alpha = 1.5 # 角度增益
        self.max_v = 0.3   # 最大线速度 (m/s) (Limo 不要太快)
        self.max_w = 0.8   # 最大角速度 (rad/s)
        self.dist_tol = 0.15 # 到达目标的距离容差 (m)
        self.angle_tol = 0.1 # 角度容差 (rad)

    def _odom_cb(self, msg):
        """解析 /odom 获取当前位姿"""
        with self.mutex:
            self.curr_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            # 四元数转欧拉角 (yaw)
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
            self.curr_yaw = yaw

    def get_current_pose(self):
        """返回 (x, y, yaw)，供 Env 传给 Trainer 更新拓扑图"""
        with self.mutex:
            if self.curr_pos is None:
                return (0.0, 0.0, 0.0) # 默认值
            return (self.curr_pos[0], self.curr_pos[1], self.curr_yaw)

    def navigate_to(self, target_x, target_y):
        """
        阻塞式函数：驱动小车移动到 (target_x, target_y)。
        逻辑：
        1. 旋转对准目标 (Turn)
        2. 直线行驶 (Move)
        注意：假设 Limo 处于四轮差速模式 
        """
        rate = rospy.Rate(10) # 10Hz 控制循环
        
        rospy.loginfo(f"[LimoController] Navigating to ({target_x:.2f}, {target_y:.2f})")
        
        while not rospy.is_shutdown():
            with self.mutex:
                if self.curr_pos is None:
                    continue
                cur_x, cur_y = self.curr_pos
                cur_yaw = self.curr_yaw

            # 计算误差
            dx = target_x - cur_x
            dy = target_y - cur_y
            dist = np.sqrt(dx**2 + dy**2)
            target_angle = np.arctan2(dy, dx)
            
            # 计算角度误差 (归一化到 -pi ~ pi)
            alpha = target_angle - cur_yaw
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

            cmd = Twist()

            # 状态机逻辑
            if dist < self.dist_tol:
                # 到达目标
                self.stop()
                rospy.loginfo("[LimoController] Reached target.")
                return True
            
            if abs(alpha) > self.angle_tol:
                # 角度偏差大，优先原地旋转
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(self.k_alpha * alpha, -self.max_w, self.max_w)
            else:
                # 角度对准了，边走边调
                cmd.linear.x = np.clip(self.k_rho * dist, -self.max_v, self.max_v)
                cmd.angular.z = np.clip(self.k_alpha * alpha, -self.max_w, self.max_w)

            self.vel_pub.publish(cmd)
            rate.sleep()
        
        return False

    def stop(self):
        """强制停车"""
        cmd = Twist()
        self.vel_pub.publish(cmd)