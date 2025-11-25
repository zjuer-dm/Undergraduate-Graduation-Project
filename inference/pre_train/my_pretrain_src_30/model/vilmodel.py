import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad


logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # padding_idx=1的核心功能是将指定索引的词向量“固定”住，使其在模型训练过程中不被更新，并通常将其初始化为零向量。
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=1)
        # 不管mlm还是sap，输入token_type_embeddings的token均全为0；而self.img_embeddings的输入给到了self.token_type_embeddings这一权重矩阵，然后在其内部通过token为1的索引用到了它
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.subtask_type_embeddings = nn.Embedding(config.max_subtask_type_embeddings, config.hidden_size, padding_idx=0)
        self.task_type_encoding = nn.Embedding(config.max_txt_task_embeddings, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = 1
        self.task_embedding_dropout_prob = config.hidden_dropout_prob

    def forward(self, input_ids, txt_task_encoding, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        if position_ids is None:
            # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = torch.arange(
            self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long, device=input_ids.device
        )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # TODO: 是否要弄一个关于子任务内部位置的position_embeddings？lxmert也有token_type_embeddings，跟此处使用一致

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        task_type_encoding = self.task_type_encoding(txt_task_encoding)
        if self.training:
            lang_keep_mask = (torch.rand((batch_size, 1, 1), device=input_ids.device) > self.task_embedding_dropout_prob).float()
            task_type_encoding = task_type_encoding * lang_keep_mask
        embeddings = words_embeddings + position_embeddings + token_type_embeddings + task_type_encoding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions # false

        self.num_attention_heads = config.num_attention_heads # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # torch.Size([32, 40, 12, 64])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 下面输出之后全都是torch.Size([32, 12, 40, 64])
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_mask: mlm时为torch.Size([32, 1, 1, 40])，它的值不是0就是-10000，前者表示这个位置有txt，后者表示是padding；sap的visn_self_att则是torch.Size([32, 1, 27, 27])
        # attention_scores：torch.Size([32, 12, 40, 40])，B*num_attention_heads*query length*key length 
        # mlm时下面的attention_mask其实是对应于nn.MultiheadAttention中的key_padding_mask，而不是attn_mask。后者是个batch＊二维矩阵的形式，表示所有要屏蔽的query－key对，而这里只能屏蔽key
        attention_scores = attention_scores + attention_mask # -10000的作用就是把padding部分的attention分数变得很小，从而让后续softmax之后的权重变得接近0，即不起作用

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # torch.Size([32, 12, 40, 40])

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # 对于每个 batch 和每个 head，它会将 attention 权重 [40, 40]（一个查询对所有键的注意力）乘上 [40, 64]（每个键对应的值向量）
        context_layer = torch.matmul(attention_probs, value_layer) # torch.Size([32, 12, 40, 64])

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # torch.Size([32, 40, 12, 64])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # 在这里完成了对多头注意力输出的拼接
        # recurrent vlnbert use attention scores
        # context_layer：torch.Size([32, 40, 768])，输入和输出同样shape
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask) # 输出是tuple，长度为1
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 输出是一个tuple，长度为1
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # 768*3072
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size) # 3072*768
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        # 一层BertLayer的全部流程：（1.计算自注意力（包含softmax非线性），2.同维度FC+dropout+与输入的残差连接+layernorm），（3.维度放大FC+激活），（4.维度缩小FC+dropout+与2输出的残差连接+layernorm）。
        # 一层BertLayer包含两个非线性模块，因此具备万能拟合的效果；残差连接跨越非线性，即残差连接的中间包含了一个非线性模块（不是末尾）
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask) # 输出是一个tuple，长度为1
        attention_output = attention_outputs[0] #取出真正的输出，跟hidden_states的shape一样
        intermediate_output = self.intermediate(attention_output) # 这里有activation（一层BertLayer只需要这里一个激活），并且把特征维数放大
        layer_output = self.output(intermediate_output, attention_output) # 把特征维数重新变小
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        # 输出是一个tuple，长度为1
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size)) # 为什么单独给出bias变量？因为self.decoder 这一线性层的权重通常会与词嵌入层（word embeddings）的权重共享，而嵌入层是没有bias的

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module): # 与BertSelfAttention的区别就是它适用于cross attention，而BertSelfAttention只能用于自注意力，其他没区别
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class GraphLXRTXLayer(nn.Module): # 对应lxmert中的LXRTXLayer，lxmert中这里的forward返回两个模态各自与对方cross-attention之后，再self-attention+output层之后的结果
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)
        self.text_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):
        # 输入的两个mask都已经是extend_neg_masks之后的结果
        # lang_feats: torch.Size([32, 40, 768])，40这一维度表示text长度
        # lang_attention_mask: torch.Size([32, 1, 1, 40])
        # visn_feats: torch.Size([32, 27, 768])，27这一维度表示节点数
        # visn_attention_mask: torch.Size([32, 1, 1, 27])
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]
        lang_att_output = self.text_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]

        if graph_sprels is not None:
            # visn_attention_mask before: torch.Size([32, 1, 1, 27])
            # graph_sprels: torch.Size([32, 1, 27, 27])
            # visn_attention_mask after: torch.Size([32, 1, 27, 27])
            # graph_sprels原本是正的表示距离，但经过线性层之后（也就是这里的graph_sprels）都是负数了，表达的应该是距离当前节点越远的节点越不需要关注，其attention分数应该越低
            # 但是为什么经过线性层会变负数呢，即使是训练一开始？
            visn_attention_mask = visn_attention_mask + graph_sprels

        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]
        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        lang_att_output = self.lang_self_att(lang_att_output, lang_attention_mask)[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)

        return lang_output, visn_output

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers # 9
        self.update_lang_bert = config.update_lang_bert # True

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks) # torch.Size([32, 1, 1, 40])
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0] # 取出真正的输出，shape与输入一致
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            # 在原版lxmert中，下面的每一层layer_module的输出都是txt_embeds, img_embeds两个，而不是只更新一个
            # 后续也考虑原本的模式
            txt_embeds, img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return txt_embeds, img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size) # 512*768
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.loc_linear = nn.Linear(config.angle_feat_size, config.hidden_size) # 4*768
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.depth_feat_size > 0:
            self.dep_linear = nn.Linear(config.depth_feat_size, config.hidden_size) # 128*768
            self.dep_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.dep_linear = self.dep_layer_norm = None

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.obj_linear = self.obj_layer_norm = None #取的是这个条件

        # 0: non-navigable, 1: navigable
        self.nav_type_embedding = nn.Embedding(2, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None # False

        # print("-------------------traj_view_img_embeds---------------------", traj_view_img_fts.shape)
        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts)) # torch.Size([159, 39, 512]) -> torch.Size([159, 39, 768])
        if self.dep_linear is not None:
            traj_view_img_embeds = traj_view_img_embeds + \
                                   self.dep_layer_norm(self.dep_linear(traj_view_dep_fts))

        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_fts))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        # traj_loc_fts: torch.Size([159, 39, 4]) -> torch.Size([159, 39, 768])
        # traj_nav_types: torch.Size([159, 39]) -> torch.Size([159, 39, 768])
        # (torch.ones(1, 1).long().to(device))：torch.Size([1, 1])，且为Long型整数
        # type_embed_layer: torch.Size([1, 1, 768])
        # traj_embeds：torch.Size([159, 39, 768])
        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device)) # 这项有什么用？
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        # traj_vp_lens: torch.Size([159])
        # traj_masks：torch.Size([159, 39])，布尔值，表示对应节点的39个视角哪些是真的有具体值的，还是说是padding
        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            # traj_embeds：torch.Size([159, 39, 768])
            # 这里自注意力的范围是每个节点内部，即节点内部的全部views的信息都会互相进行attention
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not() # logical_not是逻辑取反，也就是True和False对调，src_key_padding_mask就是上面BertSelfAttention中的attention_mask
            )
        # torch.split将第一个参数张量按照第二个参数列表里给出的值进行切分，第三个参数指的是在哪个维度上进行切分。返回一个包含多个张量的元组，每个张量都是原张量的一部分。
        # 下面两个返回值len都为32。
        # split_traj_embeds元素的shape形如torch.Size([7, 39, 768])，其中7表示这个episode的graph长度
        # split_traj_vp_lens元素是一维tensor张量，形如tensor([36, 37, 36, 36, 36], device='cuda:0')，表示这个episode下每个节点的视角数量
        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens

class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.gmap_task_embeddings = nn.Embedding(config.max_gmap_task_embeddings, config.hidden_size, padding_idx=0)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels: # True
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

        self.task_embedding_dropout_prob = config.hidden_dropout_prob

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {} # visited_vp_fts的value都是torch.Size([768])的特征向量，而unvisited_vp_fts的value是由特征向量组成的list
            # vp_masks: torch.Size([5, 37])，布尔值
            vp_masks = gen_seq_masks(split_traj_vp_lens[i]) # 输入形如tensor([36, 37, 36, 36, 36], device='cuda:0')，表示一个episode下各个节点的视角个数
            max_vp_len = max(split_traj_vp_lens[i])
            # i_traj_embeds：torch.Size([5, 37, 768])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2) # 就是把padding的粒度给变得更细了，原先用整个batch中最大的视角数进行padding，现在变成一个episode中
            for t in range(len(split_traj_embeds[i])): # 遍历这个episode的全部graph上的节点
                # torch.sum把i_traj_embeds[t]所有37个通道的feature都给加了起来（其中padding部分都是0），形成一个形状为 [768] 的张量，然后除以加的项数取平均。（完全精确的求平均，不包含任何padding feature值）
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]): # 这个episode，这个节点中所有的邻居节点
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j]) # 为什么直接用j就索引出了这个视角下的特征，因为当时traj_view_img_fts排布的时候就是先按照全部邻居节点进行append的
            gmap_img_fts = [] # 仍旧是按照先visited_vp_fts，再unvisited_vp_fts的顺序，每个元素是形状为 [768] 的张量
            for vp in gmap_vpids[i][1:]: # 排除第一个None
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0)) #仍旧是计算平均来表征这个视角特征，只不过是全部看到这个待探索节点的视角特征平均
            gmap_img_fts = torch.stack(gmap_img_fts, 0) # torch.Size([N, 768])，N表示这个episode构建的地图下所有已探索和未探索的节点数之和
            batch_gmap_img_fts.append(gmap_img_fts)

        # torch.Size([32, 26, 768])，其中26是全部episodes中最大的那个N（episode构建的地图下所有已探索和未探索的节点数之和），padding用全0feature
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        ) 
        return batch_gmap_img_fts # torch.Size([32, 27, 768])，在开头多加了一维全零feature，表示stop。27这一维度的排布：[stop_ft, visited_vp_fts, unvisited_vp_fts, padding_fts]
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        batch_size = gmap_task_embeddings.size(0)
        task_type_encoding = self.gmap_task_embeddings(gmap_task_embeddings)
        if self.training:
            gmap_keep_mask = (torch.rand((batch_size, 1, 1), device=gmap_task_embeddings.device) > self.task_embedding_dropout_prob).float()
            task_type_encoding = task_type_encoding * gmap_keep_mask

        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      task_type_encoding + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens) # torch.Size([32]) -> torch.Size([32, 27])
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_lens
        )
        if self.sprel_linear is not None:
            # graph_sprels before: torch.Size([32, 27, 27])
            # graph_sprels after: torch.Size([32, 1, 27, 27])
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1) # 对graph_sprels中的每个scalar进行一个线性变换，然后调整维度为[32, 1, 27, 27]。
        else:
            graph_sprels = None

        txt_embeds, gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return txt_embeds, gmap_embeds
       

class GlocalTextPathCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config) # 原本的架构下完全继承了原先lxmert预训练的权重，目前新加一个subtask_type_embedding
        self.lang_encoder = LanguageEncoder(config) # 完全继承了原先lxmert预训练的权重

        self.img_embeddings = ImageEmbeddings(config) # 应该是完全没继承，全部默认初始化权重

        # self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config) # 有部分没继承，cross model层原先有5层，这里只用了4层。我这边改了网络前馈的方式（改成了对称双向互注意力），但没有增删网络层
        
        self.init_weights()

    def forward(
        self, txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids
    ):        
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, txt_task_encoding, token_type_ids=txt_token_type_ids) # 输出为torch.Size([32, 40, 768])
        txt_masks = gen_seq_masks(txt_lens) # torch.Size([32, 40])，值是布尔量，表示这个位置是否真的有txt，还是说是padding
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks) # 输出为torch.Size([32, 40, 768])

        # trajectory embedding
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens,
            self.embeddings.token_type_embeddings
        )
        
        txt_embeds, gmap_embeds = self.global_encoder(
            txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_lens, graph_sprels=gmap_pair_dists,
        )

        return txt_embeds, gmap_embeds # torch.Size([32, 40, 768]), torch.Size([32, 27, 768])

    
