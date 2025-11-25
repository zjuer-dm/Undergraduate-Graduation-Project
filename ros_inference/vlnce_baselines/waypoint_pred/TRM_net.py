import torch
import torch.nn as nn
import numpy as np
import vlnce_baselines.waypoint_pred.utils as utils

from .transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

class BinaryDistPredictor_TRM(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=12, device=None):
        super(BinaryDistPredictor_TRM, self).__init__()

        self.device = device

        # ====================================================
        # 1. 参数对齐 (Hardcoded to match training args)
        # ====================================================
        self.num_angles = 120
        self.num_imgs = 4         # 对应 args.NUM_IMGS
        self.n_classes = 12 
        self.TRM_LAYER = 2        # 对应 args.TRM_LAYER
        self.TRM_NEIGHBOR = 1     # 对应 args.TRM_NEIGHBOR
        self.HEATMAP_OFFSET = 15  # 对应 args.HEATMAP_OFFSET (非常重要，方向对齐)
        
        # ====================================================
        # 2. RGB 模块 (DINOv2, 768 dim)
        # ====================================================
        self.dino_input_dim = 768
        self.visual_fc_rgb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dino_input_dim, hidden_dim), 
            nn.ReLU(True),
        )

        # ====================================================
        # 3. Depth 模块 (ResNet, 2048 dim)
        # ====================================================
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),
            nn.ReLU(True),
        )
        
        # ====================================================
        # 4. 融合模块
        # ====================================================
        self.visual_merge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(True),
        )

        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = self.TRM_LAYER
        self.waypoint_TRM = WaypointBert(config=config)

        # ====================================================
        # 5. LayerNorm 对齐 (关键！)
        # 你的训练代码里在 __init__ 中定义了它，所以这里必须定义。
        # 否则 load_state_dict 会报错 Missing Key。
        # ====================================================
        self.mergefeats_LayerNorm = BertLayerNorm(hidden_dim, eps=1e-12)
        
        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=self.TRM_NEIGHBOR).to(self.device)

        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),
        )

    def forward(self, rgb_feats, depth_feats):
        bsi = rgb_feats.size(0) // self.num_imgs

        # ====================================================
        # 6. 前向传播逻辑对齐
        # ====================================================
        
        # 1. 处理 RGB (DINO)
        rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
            bsi, self.num_imgs, -1)
            
        # 2. 处理 Depth (ResNet)
        depth_x = self.visual_fc_depth(depth_feats).reshape(
            bsi, self.num_imgs, -1)
        
        # 3. 融合 (Merge)
        vis_x = self.visual_merge(
            torch.cat((rgb_x, depth_x), dim=-1)
        )
        
        # 4. LayerNorm (保持注释)
        # 训练代码里这里被注释了，所以这里也必须注释，不要执行它。
        # vis_x = self.mergefeats_LayerNorm(vis_x)

        attention_mask = self.mask.repeat(bsi,1,1,1)
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        )

        vis_logits = self.vis_classifier(vis_rel_x)
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes)

        # heatmap offset
        vis_logits = torch.cat(
            (vis_logits[:,self.HEATMAP_OFFSET:,:], vis_logits[:,:self.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias