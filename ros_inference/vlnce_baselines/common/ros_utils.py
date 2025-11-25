
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
            'front': '/seeker/front/undistort/image_raw', 
            'right': '/seeker/right/undistort/image_raw',
            'back':  '/seeker/back/undistort/image_raw',
            'left':  '/seeker/left/undistort/image_raw'
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

class LimoController:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.curr_pos = None # [x, y]
        self.curr_yaw = None
        self.mutex = threading.Lock()
        
        # 控制参数
        self.k_rho = 0.5
        self.k_alpha = 1.2
        self.max_v = 0.25
        self.max_w = 0.6
        self.dist_tol = 0.15
        self.angle_tol = 0.1

    def _odom_cb(self, msg):
        with self.mutex:
            self.curr_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            q = msg.pose.pose.orientation
            (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.curr_yaw = yaw

    def get_current_pose(self):
        with self.mutex:
            if self.curr_pos is None:
                return (0.0, 0.0, 0.0)
            return (self.curr_pos[0], self.curr_pos[1], self.curr_yaw)

    def stop(self):
        self.vel_pub.publish(Twist())

    def navigate_to(self, target_x, target_y):
        """ 简单的 P-Controller 导航到目标点 (阻塞式) """
        rate = rospy.Rate(10)
        rospy.loginfo(f"[Nav] Moving to ({target_x:.2f}, {target_y:.2f})")
        
        while not rospy.is_shutdown():
            with self.mutex:
                if self.curr_pos is None: continue
                cx, cy, cyaw = self.curr_pos[0], self.curr_pos[1], self.curr_yaw

            dx = target_x - cx
            dy = target_y - cy
            dist = np.sqrt(dx**2 + dy**2)
            target_angle = np.arctan2(dy, dx)
            
            alpha = target_angle - cyaw
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

            if dist < self.dist_tol:
                self.stop()
                return True

            cmd = Twist()
            # 优先转向，再前进
            if abs(alpha) > self.angle_tol:
                cmd.angular.z = np.clip(self.k_alpha * alpha, -self.max_w, self.max_w)
            else:
                cmd.linear.x = np.clip(self.k_rho * dist, -self.max_v, self.max_v)
                cmd.angular.z = np.clip(self.k_alpha * alpha, -self.max_w, self.max_w)
            
            self.vel_pub.publish(cmd)
            rate.sleep()
        return False