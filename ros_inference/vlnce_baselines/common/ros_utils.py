
# TODO: 根据实际 Topic 名称修改
# 在 LimoController.navigate_to 中使用了“原地旋转+直线行驶”的逻辑。
# 这要求 Limo 必须处于 四轮差速 (Four-wheel differential)  模式。
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

class SeekerSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.mutex = threading.Lock()
        self.latest_obs = None
        
        # Seeker Topics (需在实车上最终确认)
        self.rgb_topics = {
            'front': '/front/undistort/image_raw', 
            'right': '/right/undistort/image_raw',
            'back':  '/back/undistort/image_raw',
            'left':  '/left/undistort/image_raw'
        }
        # 深度图通常不需要 undistort，或者 SDK 已处理
        self.depth_topics = {
            'front': '/front/depth/image_raw',
            'right': '/right/depth/image_raw',
            'back':  '/back/depth/image_raw',
            'left':  '/left/depth/image_raw'
        }

        # 严格对应 r2r_vlnce.yaml 中的尺寸
        self.rgb_size = (224, 224) 
        self.depth_size = (256, 256) 

        subs = []
        # 顺序：RGB(F,R,B,L) -> Depth(F,R,B,L)
        for view in ['front', 'right', 'back', 'left']:
            subs.append(message_filters.Subscriber(self.rgb_topics[view], Image))
        for view in ['front', 'right', 'back', 'left']:
            subs.append(message_filters.Subscriber(self.depth_topics[view], Image))

        # 时间同步 (slop=0.2s)
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.2)
        self.ts.registerCallback(self.callback)
        
        rospy.loginfo("[SeekerSubscriber] Waiting for synchronized images...")

    def callback(self, *args):
        # args顺序: rgb_f, rgb_r, rgb_b, rgb_l, dep_f, dep_r, dep_b, dep_l
        with self.mutex:
            self.latest_obs = self._process_frames(args)

    def _process_frames(self, frames):
        obs = {}
        views = ['front', 'right', 'back', 'left']
        
        for i, view in enumerate(views):
            # --- RGB 处理 ---
            rgb_msg = frames[i]
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)
            cv_rgb = cv2.resize(cv_rgb, self.rgb_size) # 强制 Resize 224x224
            obs[f'rgb_{view}'] = cv_rgb

            # --- Depth 处理 ---
            dep_msg = frames[i + 4]
            cv_dep = self.bridge.imgmsg_to_cv2(dep_msg, desired_encoding="passthrough")
            
            # 深度单位转换 (假设输入是mm, 目标是m)
            if cv_dep.dtype == np.uint16:
                cv_dep = cv_dep.astype(np.float32) / 1000.0
            
            # 强制 Resize 256x256 (使用最近邻插值避免虚假深度)
            cv_dep = cv2.resize(cv_dep, self.depth_size, interpolation=cv2.INTER_NEAREST)
            
            if len(cv_dep.shape) == 2:
                cv_dep = cv_dep[:, :, np.newaxis]
            
            # Habitat 深度范围通常归一化到 [0, 1] 或 [0, 10]，这里保持真实米单位
            # Trainer 内部可能会有 min_depth / max_depth 截断，这里给原始值
            obs[f'depth_{view}'] = cv_dep

        return obs

    def get_observation(self):
        with self.mutex:
            if self.latest_obs is None:
                return None
            return self.latest_obs.copy()



# MYTODO:这里还是我的测试版本，以后要按实物小车修改
class LimoController: # [Mock Auto Mode]
    def __init__(self):
        self.curr_pos = [0.0, 0.0] 
        self.curr_yaw = 0.0
        self.mutex = threading.Lock()
        print("[MockController] AUTO Simulation Mode Initialized.")

    def get_current_pose(self):
        with self.mutex:
            return (self.curr_pos[0], self.curr_pos[1], self.curr_yaw)

    def stop(self):
        print(">>> [AUTO] Robot STOPPED.")

    def navigate_to(self, target_x, target_y):
        """
        自动模拟移动过程，不需要人工干预
        """
        with self.mutex:
            start_x, start_y = self.curr_pos
            
        dx = target_x - start_x
        dy = target_y - start_y
        dist = np.sqrt(dx**2 + dy**2)
        target_yaw = np.arctan2(dy, dx)
        
        # 1. 计算模拟耗时 (假设速度 0.3 m/s)
        duration = dist / 0.3 
        # 限制最小和最大等待时间，方便调试
        duration = np.clip(duration, 1.0, 5.0)
        
        print(f"\n>>> [AUTO] Navigating to ({target_x:.2f}, {target_y:.2f})...")
        print(f">>> [AUTO] Distance: {dist:.2f}m. Moving for {duration:.1f}s...")
        
        # 2. 阻塞等待 (模拟物理移动时间)
        # 在这段时间内，模型是等待的，你的相机如果拿着走，正好模拟了移动后的视角变化
        time.sleep(duration)
        
        # 3. 自动更新坐标 (假装到了)
        with self.mutex:
            self.curr_pos = [target_x, target_y]
            self.curr_yaw = target_yaw
            
        print(f">>> [AUTO] Arrived at ({target_x:.2f}, {target_y:.2f}).\n")
        return True


# RealController (用于真实 Limo 小车) - ODOM 闭环
# class LimoController: # [Real Robot Mode]
#     def __init__(self):
#         self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odom_cb)
#         self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
#         self.curr_pos = None 
#         self.curr_yaw = None
#         self.mutex = threading.Lock()
        
#         # PID 参数
#         self.dist_tol = 0.15 # 到达阈值 (15cm)
#         self.angle_tol = 0.1 # 角度阈值 (0.1 rad)

#     # ... (省略 _odom_cb 和 get_current_pose，同之前) ...

#     def navigate_to(self, target_x, target_y):
#         rate = rospy.Rate(10) # 10Hz 控制频率
#         timeout = 15.0 # 超时时间 15秒
#         start_time = time.time()
        
#         rospy.loginfo(f"[Nav] Start moving to ({target_x:.2f}, {target_y:.2f})")
        
#         while not rospy.is_shutdown():
#             # 1. 超时检查
#             if time.time() - start_time > timeout:
#                 rospy.logwarn("[Nav] Timeout! Forced stop.")
#                 self.stop()
#                 return False # 返回失败，告诉模型我们没走到

#             # 2. 获取当前位置
#             with self.mutex:
#                 if self.curr_pos is None: continue
#                 cx, cy, cyaw = self.curr_pos[0], self.curr_pos[1], self.curr_yaw

#             # 3. 计算误差
#             dist = np.sqrt((target_x - cx)**2 + (target_y - cy)**2)
            
#             # 4. 判断到达 (Fast System 的核心退出条件)
#             if dist < self.dist_tol:
#                 self.stop()
#                 rospy.loginfo("[Nav] Target Reached.")
#                 return True # 成功返回

#             # 5. 计算控制律 (P-Control)
#             # ... (计算 cmd_vel 的逻辑同之前) ...
            
#             self.vel_pub.publish(cmd)
#             rate.sleep()
            
#         return False

