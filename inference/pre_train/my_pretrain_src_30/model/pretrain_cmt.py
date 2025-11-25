from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT, gelu, BertOutAttention
from .ops import pad_tensors_wgrad, gen_seq_masks, extend_neg_masks

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, config, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=config.layer_norm_eps),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    def __init__(self, config, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=config.layer_norm_eps),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class ResidualTransformBlock(nn.Module):
    def __init__(self, config, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = gelu
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + residual
        x = self.LayerNorm(x)
        return x
    
class NextActionPrediction(nn.Module):
    def __init__(self, config, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size*2, hidden_size*2),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size*2, eps=config.layer_norm_eps),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size*2, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)
        # self.cls_transform = CLSTransformBlock(config.hidden_size, config.pred_head_dropout_prob)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config, self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config, self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None
        if 'sap' in config.pretrain_tasks:
            self.graph_query_text = BertOutAttention(config)
            self.graph_attentioned_txt_embeds_transform = ResidualTransformBlock(self.config, self.config.hidden_size, self.config.hidden_dropout_prob)
            self.global_sap_head = NextActionPrediction(self.config, self.config.hidden_size, self.config.pred_head_dropout_prob)
            # self.global_sap_head = ClsPrediction(self.config.hidden_size)
            # self.local_sap_head = ClsPrediction(self.config.hidden_size)
            # if config.glocal_fuse:
            #     self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            # else:
            #     self.sap_fuse_linear = None
        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config, self.config.hidden_size)
        if 'gsp' in config.pretrain_tasks:
            self.gsp_query_feature = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))
            self.gsp_query_text = BertOutAttention(self.config)
            self.gsp_attentioned_txt_embeds_transform = ResidualTransformBlock(self.config, self.config.hidden_size, self.config.hidden_dropout_prob)
            self.global_stage_head = nn.Linear(self.config.hidden_size, self.config.subtask_num+1)
            

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        # 虽然 nn.Embedding 和 Linear 的功能不同，但它们的权重可以共享，因为本质上两者都是权重矩阵，它们的矩阵可以被视为“转置关系”。
        # nn.Embedding 层的权重矩阵：大小为 (vocab_size, hidden_size)。
        # Linear 层的权重矩阵：大小为 (hidden_size, vocab_size)。
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['txt_task_encoding'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_task_embeddings'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'], 
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['txt_task_encoding'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_task_embeddings'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss
            )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], compute_loss
            )
        elif task.startswith('valid_sap_og'):
            return self.forward_sap_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'], batch['global_act_labels'], batch['local_act_labels'], 
                batch['obj_labels']
            )
        elif task.startswith('gsp'):
            return self.forward_gsp(
                batch['txt_ids'], batch['txt_lens'], batch['subtask_type_encoding'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['global_stage_labels'], batch['subtask_lens'], compute_loss
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
        self, txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        txt_labels, compute_loss
    ):
        # txt_ids: torch.Size([32, 40])
        # txt_labels: torch.Size([32, 40])
        txt_embeds, _ = self.bert(
            txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        )

        # txt_embeds: torch.Size([32, 40, 768])
        # (txt_labels != -1): torch.Size([32, 40])
        # txt_labels[txt_labels != -1]: torch.Size([103])
        # masked_output: torch.Size([103, 768])
        # prediction_scores: torch.Size([103, 30522])
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden) # 将 mask 从 [32, 40] 变为 [32, 40, 1]；然后再用expand_as扩展为 [32, 40, 768]（值复制自身原本[32, 40, 1]的值）
        # hidden[mask]是一个一维tensor，形如torch.Size([90624])；.contiguous()：确保内存连续，避免在 .view() 时出错；.view将提取出的特征重新 reshape 为 [N, 768]
        # .contiguous()能重新创建一个与之前变量不共享内存的变量，view操作要求底层内存是连续的（contiguous）。而经过布尔索引（例如 hidden[mask]）操作后，返回的张量可能不是连续内存排列的，新变量一定是连续的。
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1)) 
        return hidden_masked

    def forward_mrc(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):
        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )
        
        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len+1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )   # [stop] at 0
        # vp_view_mrc_masks = vp_view_mrc_masks[:, :vp_view_embeds.size(1)]
        
        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len+1:view_len+obj_len+1] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
        self, txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        gmap_visited_masks, global_act_labels, local_act_labels, compute_loss
    ):
        batch_size = txt_ids.size(0)

        # 下方输出为torch.Size([32, 40, 768])与torch.Size([32, 27, 768])
        txt_embeds, gmap_embeds = self.bert(
            txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        )

        txt_masks = gen_seq_masks(txt_lens)
        extended_txt_masks = extend_neg_masks(txt_masks)
        graph_attentioned_txt_embeds, _ = self.graph_query_text(gmap_embeds, txt_embeds, attention_mask=extended_txt_masks) # torch.Size([32, 27, 768])
        graph_attentioned_txt_embeds = self.graph_attentioned_txt_embeds_transform(graph_attentioned_txt_embeds)
        fusion_input = torch.cat([gmap_embeds, graph_attentioned_txt_embeds], dim=-1) # torch.Size([32, 27, 768*2])
        global_logits = self.global_sap_head(fusion_input).squeeze(2) # 输出为torch.Size([32, 27])，通过线性层对torch.Size([32, 27, 768])中每个节点单独输出logits，可以认为是单独评分

        # 下方代码，用负无穷代替global_logits中visited_vpids对应位置的logits
        global_logits.masked_fill_(gmap_visited_masks, -float('inf')) # gmap_visited_masks：torch.Size([32, 27])，输出的global_logits仍旧是torch.Size([32, 27])
        # 下方代码，用负无穷代替global_logits中最后padding对应位置的logits，因此最终留下的动作空间就只有stop和unvisited_vpids
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf')) # 输出的global_logits仍旧是torch.Size([32, 27])

        if compute_loss:
            # global_logits：torch.Size([32, 27])
            # global_act_labels：torch.Size([32])
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            losses = global_losses
            return losses
        else:
            return global_logits,  global_act_labels

    def forward_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        obj_labels, compute_loss
    ):
        gmap_embeds, vp_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sap_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, obj_labels
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        
        return global_logits, local_logits, fused_logits, obj_logits

    def forward_gsp(
        self, txt_ids, txt_lens, subtask_type_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        global_stage_labels, subtask_lens, compute_loss
    ):

        # 下方输出为torch.Size([32, 40, 768])与torch.Size([32, 27, 768])
        txt_embeds, gmap_embeds = self.bert(
            txt_ids, txt_lens, subtask_type_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        )

        txt_masks = gen_seq_masks(txt_lens)
        extended_txt_masks = extend_neg_masks(txt_masks)
        B = txt_embeds.size(0)
        gsp_query_feature = self.gsp_query_feature.expand(B, -1, -1)  # [B, 1, D]
        gsp_attentioned_txt_embeds, _ = self.gsp_query_text(gsp_query_feature, txt_embeds, attention_mask=extended_txt_masks) # [B, 1, D]
        gsp_attentioned_txt_embeds = self.gsp_attentioned_txt_embeds_transform(gsp_attentioned_txt_embeds).squeeze(1) # [B, D]
        global_stage_logits = self.global_stage_head(gsp_attentioned_txt_embeds)

        # 下方代码，用负无穷代替多余的subtasks对应位置的logits，因此最终留下的动作空间就只有stop和这个episode的子任务本身的长度
        global_stage_logits.masked_fill_(gen_seq_masks(subtask_lens, max_len=self.config.subtask_num+1).logical_not(), -float('inf')) # 输出的global_logits仍旧是torch.Size([32, 27])

        if compute_loss:
            # global_stage_logits：torch.Size([32, subtask_num+1])
            # global_stage_labels：torch.Size([32])
            # print("subtask_lens ", subtask_lens.shape, subtask_lens)
            # print("global_stage_logits ", global_stage_logits.shape, global_stage_logits)
            # print("global_stage_labels ", global_stage_labels.shape, global_stage_labels)
            global_losses = F.cross_entropy(global_stage_logits, global_stage_labels, reduction='none')
            losses = global_losses
            return losses
        else:
            return global_stage_logits,  global_stage_labels