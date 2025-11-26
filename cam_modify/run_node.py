# 文件路径: ~/cam_modify/run_node.py
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rospy
import cv2
import message_filters
from sensor_msgs.msg import Image, CameraInfo

# 引入我们自己写的 bridge_utils
from bridge_utils import imgmsg_to_cv2, cv2_to_imgmsg

# --- 1. 动态挂载 FoundationStereo 源码路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
fs_path = os.path.join(current_dir, 'FoundationStereo')
sys.path.append(fs_path)

try:
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
except ImportError as e:
    rospy.logerr(f"CRITICAL: Import failed. Error: {e}")
    sys.exit(1)

class OmniStereoNode:
    def __init__(self, args):
        rospy.init_node('foundation_omni_depth', anonymous=True)
        self.device = torch.device('cuda')
        
        # --- 2. 加载配置与模型 ---
        rospy.loginfo(f"Loading Checkpoint: {args.ckpt_path}")
        
        ckpt_dir = os.path.dirname(args.ckpt_path)
        cfg_file = os.path.join(ckpt_dir, 'cfg.yaml')
        
        if not os.path.exists(cfg_file):
            rospy.logerr(f"Config file not found at {cfg_file}!")
            sys.exit(1)
            
        cfg = OmegaConf.load(cfg_file)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
            
        cfg_runtime = OmegaConf.create({
            'mixed_precision': True, 
            'restore_ckpt': args.ckpt_path
        })
        cfg = OmegaConf.merge(cfg, cfg_runtime)
        
        self.model = FoundationStereo(cfg)
        
        # 加载权重
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)
            
        self.model.to(self.device)
        self.model.eval()
        
        rospy.loginfo("FoundationStereo Model Loaded Successfully!")

        # --- 3. 几何对齐参数 ---
        # 支持通过命令行调整角度，解决"斜着"的问题
        # 默认 -45.0，如果歪了请尝试 45.0
        self.rot_angle = args.angle 
        self.update_rotation_matrix()
        
        self.coords_grid = None
        self.valid_iters = args.valid_iters
        self.inference_scale = 0.5 

        # --- 4. 定义四个方向 ---
        self.cameras = [
            {'ns': '/front', 'sub_l': '/front/left/image_raw', 'sub_r': '/front/right/image_raw', 'sub_info': '/front/right/camera_info'},
            {'ns': '/right', 'sub_l': '/right/left/image_raw', 'sub_r': '/right/right/image_raw', 'sub_info': '/right/right/camera_info'},
            {'ns': '/back',  'sub_l': '/back/left/image_raw',  'sub_r': '/back/right/image_raw',  'sub_info': '/back/right/camera_info'},
            {'ns': '/left',  'sub_l': '/left/left/image_raw',  'sub_r': '/left/right/image_raw',  'sub_info': '/left/right/camera_info'},
        ]

        self.subs = []
        for cam in self.cameras:
            self.setup_subscriber(cam)

    def update_rotation_matrix(self):
        angle_rad = np.deg2rad(self.rot_angle) 
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        # 绕 Y 轴旋转
        self.R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=self.device)
        rospy.loginfo(f"Rotation Matrix Updated: Angle = {self.rot_angle} deg")

    def setup_subscriber(self, cam_dict):
        sl = message_filters.Subscriber(cam_dict['sub_l'], Image)
        sr = message_filters.Subscriber(cam_dict['sub_r'], Image)
        si = message_filters.Subscriber(cam_dict['sub_info'], CameraInfo)
        
        pub = rospy.Publisher(cam_dict['ns'] + '/depth/aligned_image', Image, queue_size=2)
        cam_dict['pub'] = pub
        
        ts = message_filters.ApproximateTimeSynchronizer([sl, sr, si], queue_size=5, slop=0.1)
        ts.registerCallback(lambda l, r, i: self.callback(l, r, i, cam_dict))
        self.subs.append(ts)

    def callback(self, img_l_msg, img_r_msg, info_msg, cam_dict):
        torch.cuda.empty_cache()
        with torch.no_grad():
            img_l = imgmsg_to_cv2(img_l_msg, "rgb8").copy()
            img_r = imgmsg_to_cv2(img_r_msg, "rgb8").copy()
            
            if img_l is None or img_r is None: return

            t_l = torch.from_numpy(img_l).permute(2,0,1).unsqueeze(0).to(self.device).float()
            t_r = torch.from_numpy(img_r).permute(2,0,1).unsqueeze(0).to(self.device).float()
            
            # Downsample
            orig_H, orig_W = t_l.shape[-2:]
            t_l = F.interpolate(t_l, scale_factor=self.inference_scale, mode='bilinear')
            t_r = F.interpolate(t_r, scale_factor=self.inference_scale, mode='bilinear')
            
            # Padding
            padder = InputPadder(t_l.shape, divis_by=32)
            l_pad, r_pad = padder.pad(t_l, t_r)

            # Inference
            with torch.amp.autocast('cuda', enabled=True):
                pred_dict = self.model(l_pad, r_pad, iters=self.valid_iters, test_mode=True)
            
            disp = pred_dict['flow_preds'][-1] if isinstance(pred_dict, dict) else pred_dict[-1]
            if disp.ndim == 3: disp = disp.unsqueeze(1)
            
            # Unpad & Upsample
            disp = padder.unpad(disp) 
            disp = F.interpolate(disp, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            disp = disp * (1.0 / self.inference_scale)
            
            # Depth Calc
            fx = info_msg.P[0]
            Tx = info_msg.P[3] 
            baseline = abs(Tx / fx) if abs(Tx) > 1e-5 else 0.10
            disp = torch.clamp(disp, min=0.1)
            depth_tensor = (fx * baseline) / disp

            # Align
            aligned_depth = self.align_depth(depth_tensor.squeeze(), fx, info_msg.P[5], info_msg.P[2], info_msg.P[6])
            
            # --- FIX: 增加孔洞填充 (Inpainting) ---
            # 1. 转回 CPU numpy
            depth_np = aligned_depth
            
            # 2. 生成掩码：值为0的地方就是空洞
            mask = (depth_np == 0).astype(np.uint8)
            
            # 3. 只有当确实有空洞时才修复
            if np.sum(mask) > 0:
                # 先将深度图转为 8bit (为了 inpaint 函数)，处理完再映射回来精度会丢，
                # 所以我们用更简单的"形态学闭运算"来填补小洞，这对浮点数有效
                kernel = np.ones((5,5), np.uint8)
                # 闭运算：先膨胀后腐蚀，填补内部黑点
                depth_filled = cv2.morphologyEx(depth_np, cv2.MORPH_CLOSE, kernel)
                
                # 对于边缘大块缺失，用简单的 Navier-Stokes 修复 (需要转格式，暂略，形态学通常够用)
                # 或者简单的：如果还是0，就填一个极远值，避免导航认为是障碍物
                depth_filled[depth_filled == 0] = 10.0 # 假设10米外
                aligned_depth = depth_filled

            out_msg = cv2_to_imgmsg(aligned_depth, encoding="32FC1")
            out_msg.header = img_l_msg.header
            cam_dict['pub'].publish(out_msg)

    def align_depth(self, depth_map, fx, fy, cx, cy):
        H, W = depth_map.shape
        if self.coords_grid is None or self.coords_grid.shape[1] != H*W:
            v, u = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device))
            self.coords_grid = torch.stack((u.flatten(), v.flatten(), torch.ones(H*W, device=self.device)), dim=0)

        z = depth_map.flatten()
        z = torch.clamp(z, max=20.0) 
        
        x = (self.coords_grid[0] - cx) * z / fx
        y = (self.coords_grid[1] - cy) * z / fy
        xyz = torch.stack((x, y, z), dim=0)
        
        xyz_rot = torch.matmul(self.R, xyz)
        xr, yr, zr = xyz_rot[0], xyz_rot[1], xyz_rot[2]
        
        zr = torch.clamp(zr, min=0.1)
        u_proj = (fx * xr / zr) + cx
        v_proj = (fy * yr / zr) + cy
        
        valid = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
        u_final = u_proj[valid].long()
        v_final = v_proj[valid].long()
        z_final = zr[valid]
        
        aligned = torch.zeros((H, W), device=self.device, dtype=torch.float32)
        sort_idx = torch.argsort(z_final, descending=True)
        aligned[v_final[sort_idx], u_final[sort_idx]] = z_final[sort_idx]
        
        aligned = aligned.unsqueeze(0).unsqueeze(0)
        # ETPNav 256x256
        out = F.interpolate(aligned, size=(256, 256), mode='nearest') # 用 nearest 防止边缘模糊
        return out.squeeze().cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='FoundationStereo/pretrained_models/23-51-11/model_best_bp2-001.pth')
    parser.add_argument('--valid_iters', type=int, default=5)
    
    # 新增：角度参数，方便你调试
    parser.add_argument('--angle', type=float, default= 0, help='Rotation angle correction')
    
    args = parser.parse_args()
    
    node = OmniStereoNode(args)
    print(f"FoundationStereo Node Started. Rotation: {args.angle} degrees")
    rospy.spin()