'''
Instruction and trajectory (view and object features) dataset
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math

from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax

MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_file, dep_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, depth_feat_size=128, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        val_sample_num=None,
    ):
        self.num_cam = 4
        self.num_cam_3 = 12
        self.img_ft_file = img_ft_file
        self.dep_ft_file = dep_ft_file
        self.obj_ft_file = obj_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.depth_feat_size = depth_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
            self._feature_store_depth = {}

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        # 应该是存储每一个节点周围的所有candidate节点信息,rel表示相对，rel_heading和rel_elevation可能是表示在36栅格的对应栅格内，还相对差多少角度
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        
        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir) # self.graph是字典，以scan名为key，nx.Graph为value
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(self.num_cam_3)] # 36个元素的列表，每一个元素36*2，以30度为水平栅格，计算了向下30度、平视、向上30度三圈的相对角度差
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles] # 得到angle_feature向量，每个元素4维

        self.data = []
        
        # item形如 {'instr_id': '6250_0', 'scan': 'VLzqgDo317F', 'path': 
        # ['af3af33b0120469c9a00daa0d0b36799', '5be145994f974347850a48cecd04cdcd', '79aedad1206b4eea9c4b639ea2182eb7', '1c91ed40af2246f2b126dd0f661970df', 
        # '385019f5d018430fa233d483b253076c', 'fd263d778b534f798d0e1ae48886e5f3'], 'heading': 3.751, 'instr_encoding': 
        # [101, 3328, 2091, 2028, 3462, 1997, 5108, 1998, 2644, 2006, 1996, 4899, 1012, 102]}
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            sel_idxs = np.random.permutation(len(self.data))[:val_sample_num]
            self.data = [self.data[sidx] for sidx in sel_idxs]

    def __len__(self):
        return len(self.data)

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100 # ignore 
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos':
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            # 'vp_objids': last_vp_objids,
            'vp_angles': last_vp_angles,
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            # 模型的输入中关于转角的特征只有当前节点视角下其他节点的方位，因此其他节点的转角没用，因此预训练中直接将除了最后一个cur_vp以外的节点转角全部设为0（即仿真器坐标系下的0转角）
            # 因此下面的代码其实计算的是cur_vp在仿真器下的转角
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % self.num_cam) * math.radians(90)
            elevation = (viewidx // self.num_cam - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[self.num_cam][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(self.num_cam_3) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[self.num_cam][idx] for idx in range(self.num_cam_3) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (self.num_cam_3 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
               last_vp_angles, last_vp_objids
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        # 最后结果，visited_vpids存储了路径点，而unvisited_vpids存储了所有路径上未经过的邻居节点
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys()) # 全部的动作空间
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        # self.act_visited_node == False
        if self.act_visited_node: # 默认为False
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        # line_dist, shortest_dist, shortest_step)
        # 对gmap_vpids中每一个节点进行计算，相对于终点
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]] / MAX_DIST
        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation), line_dist, shortest_dist, shortest_step)
        # cur_heading, cur_elevation是终点在仿真器下的转角（粗略划分，30度栅格），详见get_cur_angle函数
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
       

       

class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_file, dep_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, depth_feat_size=128, angle_feat_size=4,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        val_sample_num=None, start_vp_file=None
    ):
        # --- 1. 定义新的视角配置 ---
        self.horizontal_views = 4
        self.total_views = self.horizontal_views * 3
        # 正前方视角索引 = 中间层的起始索引 (索引从0开始)
        self.forward_view_index = self.horizontal_views

        # --- 2. 正常调用父类的 __init__ 方法 ---
        # 父类会先用默认的36视角配置创建一些属性
        super().__init__(
            anno_files, img_ft_file, dep_ft_file, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size, depth_feat_size=depth_feat_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node, val_sample_num=val_sample_num
        )
        
        # --- 3. 覆盖父类中基于36视角创建的属性 ---
        # 现在 get_view_rel_angles 支持新参数了，我们可以安全地调用它
        self.all_point_rel_angles = [get_view_rel_angles(i, horizontal_views=self.horizontal_views) for i in range(self.total_views)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]
        # self.start_vp_dict = {1:2, 3:4}

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            
            # 使用我们定义的配置参数进行计算
            heading = (viewidx % self.horizontal_views) * (2 * math.pi / self.horizontal_views)
            elevation = (viewidx // self.horizontal_views - 1) * math.radians(30)
            
        return heading, elevation

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts = self._feature_store[key]
            dep_fts = self._feature_store_depth[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)
            with h5py.File(self.dep_ft_file, 'r') as f:
                dep_fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = view_fts
                self._feature_store_depth[key] = dep_fts
        return view_fts, dep_fts

    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1 # [stop] is 0
                    break
        return global_act_label, local_act_label
    
    def get_stage_labels(self, end_idx, subtask_list):
        if end_idx > subtask_list[-1]:
            print("Dataset subtask input error!!!")
        if end_idx == subtask_list[-1]:
            return 0
        stage = 1
        for s in subtask_list:
            if end_idx >= s:
                stage = stage + 1
            else:
                break
        return stage

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, return_stage_label=False, end_vp=None
    ):
        # item形如 {'instr_id': '6250_0', 'scan': 'VLzqgDo317F', 'path': 
        # ['af3af33b0120469c9a00daa0d0b36799', '5be145994f974347850a48cecd04cdcd', '79aedad1206b4eea9c4b639ea2182eb7', '1c91ed40af2246f2b126dd0f661970df', 
        # '385019f5d018430fa233d483b253076c', 'fd263d778b534f798d0e1ae48886e5f3'], 'heading': 3.751, 'instr_encoding': 
        # [101, 3328, 2091, 2028, 3462, 1997, 5108, 1998, 2644, 2006, 1996, 4899, 1012, 102], 'subtask': [2, 3, 5]}
        # global stage向量构成是[stop, subtask1, subtask2, subtask3, ...]，stop表示已经完成了最后一个子任务（但当前不一定要在那个地方），其他表示的是目前已经完成了前一个子任务（所有已完成中最靠后的那一个）
        # subtask中，例如[2, 3, 5]表示有三个子任务，0-2，2-3，3-5。当前位于0/1时，stage应该是1；位于2时，stage应该是2；位于3/4时，stage应该是3；位于5时，stage应该是0
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        # 对于sap任务，end_vp是当前决策的起点。模型收到从episode起点开始到end_vp的状态，然后依据此来推理下一步去哪，即单步推理，根据概率选择下面if end_vp is None:中的一种情况
        # 对于mlm任务，end_vp就是路径最后一个点，模型依据完整的路径观测来还原instruction
        if end_vp is None:
            if end_vp_type == 'pos': 
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1] # 排除最后一个点
                end_idx = np.random.randint(len(end_vps)) # [0, len(gt_path)-2]
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)
            
        gt_path = gt_path[:end_idx+1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading) # 得到终点时在仿真器下的转角（粗略划分，30度栅格），详见get_cur_angle函数

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        # traj_view_img_fts：这段episode轨迹上，每个节点的36*512图像特征，36表示36个视角，其中靠前的方向是有对应方向邻居节点的，靠后的方向没有
        # traj_view_dep_fts：这段episode轨迹上，每个节点的36*128深度特征
        # traj_loc_fts：这段episode轨迹上，每个节点的36*4角度特征，角度是相对于前后节点，靠前的方向是有对应方向邻居节点的，且角度是精确值，靠后的是粗略的栅格角度
        # traj_nav_types：这段episode轨迹上，每个节点的36*1的0-1特征，靠前的为1，1的个数表示该节点的邻居数量
        # traj_cand_vpids：这段episode轨迹上，每个节点的邻居vp名字列表，不定长
        # 上面的36是一般情况，但若一个视角下存在多个邻居节点，那么会大于36（traj_nav_types一定是36）
        traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        # gmap_vpids：轨迹上经过的所有节点以及邻居节点的名字列表，gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        # gmap_step_ids：对应gmap_vpids中所有节点，轨迹什么时候经过了这个节点，没有为0，否则就是对应的step数
        # gmap_visited_masks：gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
        # gmap_pos_fts：shape为len(gmap_vpids)*7，表示gmap_vpids中每个节点相对于终点的相对位置信息，
        #                                           (sin(heading), cos(heading), sin(elevation), cos(elevation), line_dist, shortest_dist, shortest_step)
        # gmap_pair_dists：len(gmap_vpids)*len(gmap_vpids)，表示两两之间的最短节点路径距离
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len], # ID
            'task_type_encoding': item['task_type_encoding'],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_view_dep_fts': [x[:, :self.depth_feat_size] for x in traj_view_dep_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            # 'vp_pos_fts': vp_pos_fts,
            # 'vp_angles': last_vp_angles,
        }

        if return_act_label: # SAP时为True
            # 对于SAP任务，终点是随机的，而不一定是最后一个点
            # global_act_label是给定的终点在gmap_vpids中的列表下标
            # local_act_label是给定的终点在traj_cand_vpids[-1]中的列表下标+1（0下标是stop）
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            # 两个正整数
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs: # MRC时为True
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
        
        if return_stage_label:
            global_stage_label = self.get_stage_labels(end_idx, item['subtask'])
            outs['global_stage_labels'] = global_stage_label
            outs['subtask_lens'] = len(item['subtask']) + 1
        return outs

    def get_traj_pano_fts(self, scan, path):
        traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, dep_fts = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_dep_fts, view_angles, cand_vpids = [], [], [], []
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_dep_fts.append(dep_fts[v[0]])
                
                # 使用 self.forward_view_index
                view_angle = self.all_point_rel_angles[self.forward_view_index][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
                
            # non cand views - 使用 self.total_views
            view_img_fts.extend([view_fts[idx] for idx in range(self.total_views) if idx not in used_viewidxs])
            view_dep_fts.extend([dep_fts[idx] for idx in range(self.total_views) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[self.forward_view_index][idx] for idx in range(self.total_views) if idx not in used_viewidxs])
            
            view_img_fts = np.stack(view_img_fts, 0)
            view_dep_fts = np.stack(view_dep_fts, 0)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            
            traj_view_img_fts.append(view_img_fts)
            traj_view_dep_fts.append(view_dep_fts)
            traj_loc_fts.append(view_ang_fts)
            
            # 使用 self.total_views
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (self.total_views - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)
            
            last_vp_angles = view_angles

        return traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles