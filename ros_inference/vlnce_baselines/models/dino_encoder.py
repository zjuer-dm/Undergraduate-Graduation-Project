import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os
import math
import sys

# ==============================================================================
# 【路径配置】确保能找到本地的 dinov2 源码
# ==============================================================================
# 请确认这个路径在你的 SFT 训练环境中也是存在的
sys.path.append('ros_inference/dinov2')
from dinov2.models.vision_transformer import vit_base

# ==============================================================================
# 【补丁 1】修复权重加载兼容性
# ==============================================================================
try:
    from torch._tensor import _rebuild_from_type_v2
except ImportError:
    def _rebuild_from_type_v2(func, new_type, args, state):
        ret = func(*args)
        if type(ret) is not new_type:
            ret = ret.as_subclass(new_type)
        return ret
    import torch._tensor
    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2

# ==============================================================================
# 【补丁 2】修复注意力机制缺失
# ==============================================================================
if not hasattr(F, 'scaled_dot_product_attention'):
    def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            if attn_mask is not None:
                attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout_p > 0.0:
            attn_weight = F.dropout(attn_weight, p=dropout_p)
        return attn_weight @ value
    F.scaled_dot_product_attention = manual_sdpa

# ==========================================
# DINOv2 Encoder 定义
# ==========================================
class DINOv2Encoder(nn.Module):
    def __init__(self, checkpoint_path='data/dinov2_vitb14_reg4_pretrain.pth'):
        super().__init__()
        print(f"Loading DINOv2 model from local: {checkpoint_path} ...")
        
        self.model = vit_base(
            patch_size=14, 
            img_size=518, 
            init_values=1.0, 
            block_chunks=0,
            num_register_tokens=4 
        )
        
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # 这里为了防止报错，如果路径不对，可以尝试打印警告而不是直接崩溃（可选）
            raise FileNotFoundError(f"Model path not found: {checkpoint_path}")

        # 冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 预处理
        self.preprocess = T.Compose([
            T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, x):
        # ====================================================
        # 【新增】适配 Habitat 的字典输入
        # 如果输入是字典（例如 {'rgb': ..., 'depth': ...}），自动提取 rgb
        # ====================================================
        if isinstance(x, dict):
            if "rgb" in x:
                x = x["rgb"]
            else:
                raise KeyError("DINOv2Encoder received a dict but could not find 'rgb' key.")
        
        # ====================================================
        # 原有逻辑保持不变
        # ====================================================
        # 1. 处理 5 维输入 (Batch, Views, ...) -> 合并前两维
        if x.dim() == 5:
            b, n, d1, d2, d3 = x.shape
            x = x.reshape(b * n, d1, d2, d3)
        
        # 2. 自动检测通道位置 (HWC vs CHW)
        if x.shape[1] == 3:
            pass 
        elif x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2) # HWC -> CHW
            if x.max() > 1.0: 
                x = x.float() / 255.0
        
        # 3. 类型转换
        if x.dtype != torch.float32 and x.dtype != torch.float16:
             x = x.float()
             if x.max() > 1.0:
                 x = x / 255.0

        # 4. DINO 预处理
        x = self.preprocess(x)
        
        # 5. 提取特征
        features_dict = self.model.forward_features(x)
        features = features_dict['x_norm_clstoken']
        
        return features