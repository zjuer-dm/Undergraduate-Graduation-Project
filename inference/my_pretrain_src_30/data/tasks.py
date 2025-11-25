import random
import math
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .common import pad_tensors, gen_seq_masks

############### Masked Language Modeling ###############
def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_tokens, output_label = [], []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_tokens.append(mask)

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_tokens.append(random.choice(list(range(*vocab_range))))

            # -> rest 10% randomly keep current token
            else:
                output_tokens.append(token)

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            output_tokens.append(token)
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        output_tokens[0] = mask

    return output_tokens, output_label    

class MlmDataset(Dataset):
    def __init__(self, nav_db, tok):
        self.nav_db = nav_db
        self.tok = tok

        self.vocab_range = [4, 250000]
        self.cls_token_id = self.tok.cls_token_id   # 0
        self.sep_token_id = self.tok.sep_token_id   # 2
        self.mask_token_id = self.tok.mask_token_id # 250001
        self.pad_token_id = self.tok.pad_token_id   # 1
        print(f"Special tok id check: \
              self.cls_token_id:{self.cls_token_id}, self.sep_token_id:{self.sep_token_id}, self.mask_token_id:{self.mask_token_id}, self.pad_token_id:{self.pad_token_id}")

    def __len__(self):
        return len(self.nav_db)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos')

        output = {}

        # txt_ids：被mask之后的instruction，list
        # txt_labels：真值，-1时表示该位置没有被mask，不需要判断；否则就是被mask了，记录了真值ID，list
        txt_ids, txt_labels = random_word(inputs['instr_encoding'], 
            self.vocab_range, self.mask_token_id)
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_labels'] = torch.LongTensor(txt_labels)

        # 下面output['traj_view_img_fts']这一列表的长度就是graph长度，可能是4-7之间，元素尺寸为torch.Size([k, 512])，其中k是一个大于等于36的数
        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        # print("--------------------traj_view_img_fts--------------------: ", len(output['traj_view_img_fts']), output['traj_view_img_fts'][0].shape)
        output['traj_view_dep_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_dep_fts']]
        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        task_type_encoding = inputs['task_type_encoding']
        if task_type_encoding == 1:
            output['txt_task_encoding'] = torch.full_like(output['txt_ids'], 1)
            output['gmap_task_embeddings'] = torch.full_like(output['gmap_step_ids'], 1)
        elif task_type_encoding == 2:
            output['txt_task_encoding'] = torch.full_like(output['txt_ids'], 2)
            output['gmap_task_embeddings'] = torch.full_like(output['gmap_step_ids'], 2)
        elif task_type_encoding == 3:
            output['txt_task_encoding'] = torch.full_like(output['txt_ids'], 3)
            output['gmap_task_embeddings'] = torch.full_like(output['gmap_step_ids'], 1)
        else:
            print("*************************** task_type_encoding ERROR ***************************")
            output['txt_task_encoding'] = None
            output['gmap_task_embeddings'] = None

        # output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        # output['vp_angles'] = inputs['vp_angles']
        return output

def mlm_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    
    # text batches
    # batch['txt_lens']: torch.Size([32])，记录了每个episode指令长度
    # batch['txt_ids']：torch.Size([32, 40])，记录了每个episode指令的token id，并且通过pad对齐了长度，例如40，表示这个batch中最长的一句话是40个tokens
    # batch['txt_labels']：torch.Size([32, 40])，记录了每个episode指令的真值token id，里面的值为-1表示这个token是不需要预测的，否则就是对应真值
    # batch['subtask_type_encoding']：torch.Size([32, 40])，记录了每个episode指令的subtask_type嵌入，同一个子任务内部序号相同，不同子任务之间不同
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=1)
    batch['txt_labels'] = pad_sequence(batch['txt_labels'], batch_first=True, padding_value=-1)
    batch['txt_task_encoding'] = pad_sequence(batch['txt_task_encoding'], batch_first=True, padding_value=0)
    batch['gmap_task_embeddings'] = pad_sequence(batch['gmap_task_embeddings'], batch_first=True, padding_value=0)
    
    # trajectory batches: traj_cand_vpids, traj_vpids
    # traj关注的是节点内部36及以上个视角下的特征，以视角为单元
    # batch['traj_step_lens']：长为32的list，表示当前graph建立了几个节点
    # batch['traj_vp_view_lens']：torch.Size([159])，记录了这32个graph中每个节点的traj_view_img_fts长度，也就是36及以上；159表示batch['traj_step_lens']中所有元素和，即全部节点数
    # batch['traj_view_img_fts']：torch.Size([159, 39, 512])，记录了这32个graph中每个节点中36及以上个视角下的特征向量，排前面的是candidate节点对应视角下的特征，排后面则没有对应的candidate节点
    # batch['traj_view_dep_fts']：torch.Size([159, 39, 128])，含义同batch['traj_view_img_fts']
    # batch['traj_loc_fts']：torch.Size([159, 39, 4])，记录了这32个graph中每个节点中36及以上个视角下，对应朝向与当前朝向的4维相对角度特征，ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    # batch['traj_nav_types']：torch.Size([159, 39])，记录了这32个graph中每个节点中36及以上个视角对应的导航类别，形如[1,1,1,...,0,0,0,...]，前面的1表示这个位置有对应的candidate动作节点，0表示没有
    # batch['traj_cand_vpids']：三维list，记录了每个episode的每个graph节点的邻居名字（即动作空间）
    #   第一维是batch，长度32；第二维是graph，长度就是7以下；第三维是每个graph节点的全部邻居节点名字列表，长度任意，元素值就如同下一行所示
    # batch['traj_vpids']：二维list，记录每个episode的graph节点名字。第一维是batch，长度32；第二维是每个episode的gt_path，长度就是7以下，元素值形如'fd263d778b534f798d0e1ae48886e5f3'
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    # 执行下面语句之前，batch['traj_view_img_fts']是一个2维嵌套list，其长度为32，即batch_size，元素也是列表，其长度不一定，4-7，表示某个graph的图节点数目，其元素尺寸为[36, 512]，36这个数可以更大
    # 执行下面语句之后，batch['traj_view_img_fts'].shape形式为torch.Size([159, 39, 512])，它把之前的嵌套列表给展平了，将batch维度和图节点维度都摊平，形成了159这个维度；
    # 并且对取节点feature尺寸中最大的来进行pad，形成了39这个维度。因此最后的这个数据无法区分不同batch之间的数据，也无法区分不同节点之间的数据？
    # 答：参考vilmodel.py中ImageEmbeddings对应forward模块，通过其他的数据辅助是可以区分的。
    # 最终预训练输入网络的图像feature就是这个形式
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_view_dep_fts'] = pad_tensors(sum(batch['traj_view_dep_fts'], []))
    if 'traj_obj_img_fts' in batch: # 在R2RTextPathData这一设置下是没有这一项的，原本的ReverieTextPathData有这一项
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    # gmap关注的是episode轨迹上所有节点的特征，以节点为单元
    # batch['gmap_lens']：torch.Size([32])，记录了每个episode的'gmap_step_ids'实际有多长，即不包含pad的长度；也就是每个episode的'gmap_vpids'列表的长度
    #   形如[12, 11, 29, 10, 31，...]，其中最大值就是31
    # batch['gmap_vpids']：二维列表，记录了所有episodes的gt_path下的节点以及所有邻居节点以及开头的None，第一维就是batch；第二维记录了所有当前gt_path中，visited_vpids和unvisited_vpids的名字，
    #   形如[None, '9f5c9d1c2ead4ce2a4f65971f2cd91f3', ...]，第一个None是固定加入的，表示stop，然后先是全部的visited_vpids，然后最末尾的一部分是全部unvisited_vpids
    # batch['gmap_step_ids']：torch.Size([32, 31])，记录了batch['gmap_vpids']对应每一个node的step_id，第一个None为0，第一个节点开始就是从1增大，unvisited_vpids也是0，pad也是0，
    #   第二维元素形如[0, 1, 2, 3, 0, 0, ..., 0]
    # batch['gmap_visited_masks']：torch.Size([32, 31])，记录了每个episode的'gmap_vpids'对应位置节点是否为visited_vpids，True表示探索过了，False表示待探索或者pad，
    #   第二维元素形如[False,  True,  True,  True, False, False, ..., False]
    # batch['gmap_pos_fts']：torch.Size([32, 31, 7])，7是特征维数，即[rel_ang_fts, rel_dists]，其中rel_ang_fts是4维，rel_dists是3维。
    #   表示episode中在当前节点（最后一个节点）视角下，所有其他节点的相对位姿关系。
    # batch['gmap_pair_dists']：torch.Size([32, 31, 31])，gmap_vpids中节点两两之间规一化后的最短实际路径距离（shortest_distances/30，并且加了pad，开头的None一行一列都是0
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    # batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    # batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    return batch


############### Masked Region Modeling ###############
def _get_img_mask(mask_prob, num_images):
    img_mask = [np.random.rand() < mask_prob for _ in range(num_images)]
    if not any(img_mask):
        # at least mask 1
        img_mask[np.random.randint(num_images)] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

def _get_targets(img_soft_label, img_masks):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(-1, soft_label_dim)
    return label_targets

class MrcDataset(Dataset):
    def __init__(self, nav_db, tok, mask_prob, end_vp_pos_ratio=1):
        self.nav_db = nav_db
        self.tok = tok
        self.mask_prob = mask_prob

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

        self.end_vp_pos_ratio = end_vp_pos_ratio
        

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        else:
            end_vp_type = 'neg_in_gt_path'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_img_probs=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        
        # mask image
        view_mrc_masks = _get_img_mask(self.mask_prob, len(output['traj_view_img_fts'][-1]))
        output['traj_view_img_fts'][-1] = _mask_img_feat(output['traj_view_img_fts'][-1], view_mrc_masks)
        output['vp_view_probs'] = torch.from_numpy(inputs['vp_view_probs']) # no [stop]
        output['vp_view_mrc_masks'] = view_mrc_masks
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
            if len(output['traj_obj_img_fts'][-1]) > 0:
                obj_mrc_masks = _get_img_mask(self.mask_prob, len(output['traj_obj_img_fts'][-1]))
                output['traj_obj_img_fts'][-1] = _mask_img_feat(output['traj_obj_img_fts'][-1], obj_mrc_masks)
            else:
                obj_mrc_masks = torch.zeros(0, ).bool()
            output['vp_obj_probs'] = torch.from_numpy(inputs['vp_obj_probs'])
            output['vp_obj_mrc_masks'] = obj_mrc_masks

        return output

def mrc_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # vp labels
    batch['vp_view_mrc_masks'] = pad_sequence(batch['vp_view_mrc_masks'], batch_first=True, padding_value=0)
    batch['vp_view_probs'] = pad_tensors(batch['vp_view_probs'])

    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
        batch['vp_obj_mrc_masks'] = pad_sequence(batch['vp_obj_mrc_masks'], batch_first=True, padding_value=0)
        batch['vp_obj_probs'] = pad_tensors(batch['vp_obj_probs'])

    return batch


############### Single-step Action Prediction ###############
class SapDataset(Dataset):
    def __init__(self, nav_db, tok, end_vp_pos_ratio=0.2):
        '''Instruction Trajectory Matching'''
        self.nav_db = nav_db
        self.tok = tok

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

        self.end_vp_pos_ratio = end_vp_pos_ratio

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        elif r < 0.6:
            end_vp_type = 'neg_in_gt_path'
        else:
            end_vp_type = 'neg_others'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_act_label=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])
        # output['subtask_type_encoding'] = torch.LongTensor(inputs['subtask_type_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_view_dep_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_dep_fts']]
        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        # output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        # output['vp_angles'] = inputs['vp_angles']
        task_type_encoding = inputs['task_type_encoding']
        if task_type_encoding == 1:
            output['txt_task_encoding'] = torch.full_like(output['txt_ids'], 1)
            output['gmap_task_embeddings'] = torch.full_like(output['gmap_step_ids'], 1)
        elif task_type_encoding == 2:
            output['txt_task_encoding'] = torch.full_like(output['txt_ids'], 2)
            output['gmap_task_embeddings'] = torch.full_like(output['gmap_step_ids'], 2)
        elif task_type_encoding == 3:
            output['txt_task_encoding'] = torch.full_like(output['txt_ids'], 3)
            output['gmap_task_embeddings'] = torch.full_like(output['gmap_step_ids'], 1)
        else:
            print("*************************** task_type_encoding ERROR ***************************")
            output['txt_task_encoding'] = None
            output['gmap_task_embeddings'] = None

        output['local_act_labels'] = inputs['local_act_labels']
        output['global_act_labels'] = inputs['global_act_labels']
        return output

def sap_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=1)
    # batch['subtask_type_encoding'] = pad_sequence(batch['subtask_type_encoding'], batch_first=True, padding_value=0)
    batch['txt_task_encoding'] = pad_sequence(batch['txt_task_encoding'], batch_first=True, padding_value=0)
    batch['gmap_task_embeddings'] = pad_sequence(batch['gmap_task_embeddings'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_view_dep_fts'] = pad_tensors(sum(batch['traj_view_dep_fts'], []))
    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    # batch['gmap_visited_masks']：torch.Size([32, 31])，记录了每个episode的'gmap_vpids'对应位置节点是否为visited_vpids，True表示探索过了，False表示待探索或者pad，
    #   第二维元素形如[False,  True,  True,  True, False, False, ..., False]
    # batch['global_act_labels']: torch.Size([32])，元素值为整数，表示每个episode对应的下一个动作真值节点序号，序号就是这个真值动作节点在gmap_vpids里的编号，0表示stop动作
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    # batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    # batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # action labels
    batch['local_act_labels'] = torch.LongTensor(batch['local_act_labels'])
    batch['global_act_labels'] = torch.LongTensor(batch['global_act_labels'])

    return batch


############### Global Stage Prediction ###############
class GspDataset(Dataset):
    def __init__(self, nav_db, tok, end_vp_pos_ratio=0.2):
        '''Instruction Trajectory Matching'''
        self.nav_db = nav_db
        self.tok = tok

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

        self.end_vp_pos_ratio = end_vp_pos_ratio

    def __len__(self):
        return self.nav_db.first_dataset_len

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        elif r < 0.6:
            end_vp_type = 'neg_in_gt_path'
        else:
            end_vp_type = 'neg_others'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_stage_label=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])
        # output['subtask_type_encoding'] = torch.LongTensor(inputs['subtask_type_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_view_dep_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_dep_fts']]
        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        # output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        # output['vp_angles'] = inputs['vp_angles']

        output['global_stage_labels'] = inputs['global_stage_labels']
        output['subtask_lens'] = inputs['subtask_lens']
        return output

def gsp_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=1)
    # batch['subtask_type_encoding'] = pad_sequence(batch['subtask_type_encoding'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_view_dep_fts'] = pad_tensors(sum(batch['traj_view_dep_fts'], []))
    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    # batch['gmap_visited_masks']：torch.Size([32, 31])，记录了每个episode的'gmap_vpids'对应位置节点是否为visited_vpids，True表示探索过了，False表示待探索或者pad，
    #   第二维元素形如[False,  True,  True,  True, False, False, ..., False]
    # batch['global_act_labels']: torch.Size([32])，元素值为整数，表示每个episode对应的下一个动作真值节点序号，序号就是这个真值动作节点在gmap_vpids里的编号，0表示stop动作
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    # batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    # batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # action labels
    batch['global_stage_labels'] = torch.LongTensor(batch['global_stage_labels'])
    batch['subtask_lens'] = torch.LongTensor(batch['subtask_lens'])

    return batch

############### Object Grounding ###############
class OGDataset(Dataset):
    def __init__(self, nav_db, tok):
        self.nav_db = nav_db
        self.tok = tok

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos', return_obj_label=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        output['obj_labels'] = inputs['obj_labels']
        return output

def og_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_vp_obj_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # vp labels
    batch['obj_labels'] = torch.LongTensor(batch['obj_labels'])
    return batch
