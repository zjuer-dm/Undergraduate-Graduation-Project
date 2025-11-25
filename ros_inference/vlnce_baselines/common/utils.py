from typing import Any, Dict, List
import torch
import torch.distributed as dist
import numpy as np
import copy
import math


def split_list_by_pattern(input_list, pattern=[102, 101]):
    """
    将整数列表按指定模式进行切分。
    不会改变输入的list

    Args:
        input_list (list): 原始的整数列表。
        pattern (list): 用于切分的子片段，默认为 [102, 101]。

    Returns:
        list: 切分后的列表（由多个列表组成）。
    """
    result = []
    temp = []
    pattern_len = len(pattern)

    for i in range(len(input_list)):
        temp.append(input_list[i])
        # 检查当前位置是否匹配模式
        if input_list[i:i + pattern_len] == pattern:
            # 如果匹配，将当前片段加入结果并清空临时列表
            if temp:
                result.append(temp)
            temp = []

    # 添加最后一个片段（如果有）
    if temp:
        result.append(temp)

    return result

def count_token(lst, token_id=1064):
    """统计列表中 1064 出现的次数"""
    return lst.count(token_id)

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    global_text_stage: List[int],
    tokens_uuid: str = "tokens",
    max_length: int = 512,
    pad_id: int = 0,
):
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure."""
    if instruction_sensor_uuid not in observations[0]:
        return observations
    global_token_list = [None] * len(observations)
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            
            token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
            new_token = copy.deepcopy(token)
            token_split = split_list_by_pattern(new_token)
            
            for t in token_split:
                if len(t) < max_length:
                    t += [pad_id] * (max_length - len(t))
                # print("0 ", t)
            # print("1 ", token_split[0])
            if len(token) < max_length:
                    token += [pad_id] * (max_length - len(token))
            # TODO: observation是会因为env stop而减少的，因此下面的赋值也需要注意，否则多环境情况下会赋错值
            observations[i][instruction_sensor_uuid] = token_split[global_text_stage[i]]
            global_token_list[i] = token_split
        else:
            break
    # print("QQQQQQQQQQQQQQQQQQ", observations[0][instruction_sensor_uuid])
    return observations, global_token_list

def extract_instruction_tokens_new(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    max_length: int = 512,
    pad_id: int = 0,
    output_subtask_num = False
):
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure."""
    if instruction_sensor_uuid not in observations[0]:
        return observations
    if output_subtask_num:
        global_subtask_num = [None] * len(observations)
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
            if len(token) < max_length:
                token += [pad_id] * (max_length - len(token))
            observations[i][instruction_sensor_uuid] = token
            if output_subtask_num:
                global_subtask_num[i] = count_token(token) + 1 # k个分割符表示有k+1个subtasks，最终分类的动作空间有k+2个
        else:
            break
    if output_subtask_num:
        return observations, global_subtask_num
    else:
        return observations
    
def process_instr_encoding(instr_encoding):
    type_encoding = []
    subtask_type = 4  # 从4开始，CLS是1，SEP是2，1064是3
    for idx, token in enumerate(instr_encoding):
        if token == 101:
            type_encoding.append(1)  # CLS
        elif token == 102:
            type_encoding.append(2)  # SEP
        elif token == 1064:
            type_encoding.append(3)  # |
            subtask_type += 1
        elif token == 0:
            type_encoding.append(0)  # pad
        else:
            type_encoding.append(subtask_type)
    return type_encoding
def extract_instruction_tokens_new_20(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    max_length: int = 512,
    pad_id: int = 0,
    output_subtask_num = False
):
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure."""
    if instruction_sensor_uuid not in observations[0]:
        return observations
    if output_subtask_num:
        global_subtask_num = [None] * len(observations)
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
            if len(token) < max_length:
                token += [pad_id] * (max_length - len(token))
            observations[i][instruction_sensor_uuid] = token
            if output_subtask_num:
                global_subtask_num[i] = count_token(token) + 1 # k个分割符表示有k+1个subtasks，最终分类的动作空间有k+2个
                observations[i]["subtask_type_ids"] = process_instr_encoding(token)
        else:
            break
    if output_subtask_num:
        return observations, global_subtask_num
    else:
        return observations
def extract_instruction_tokens_new_30(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    max_length: int = 512,
    pad_id: int = 1,
    task_type: int = None
):
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure."""
    if instruction_sensor_uuid not in observations[0]:
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
            origin_token_len = len(token)
            if len(token) < max_length:
                token += [pad_id] * (max_length - len(token))
            observations[i][instruction_sensor_uuid] = token
            if task_type == 1 or task_type == 2:
                task_type_token = [task_type] * origin_token_len
                if len(task_type_token) < max_length:
                    task_type_token += [0] * (max_length - len(task_type_token))
                observations[i]['txt_task_encoding'] = task_type_token
            else:
                print("task_type invalid")
                break
        else:
            break
    return observations
    
def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor) # 将每个local_rank中的tensor都发送到共享变量gather_t中对应索引中，覆盖默认值，此后所有进程就都能访问到gather_t这个共享变量了。（阻塞操作，同步所有进程）
    return gather_t

def dis_to_con(path, amount=0.25):
    starts = path[:-1]
    ends = path[1:]
    new_path = [path[0]]
    for s, e in zip(starts,ends):
        vec = np.array(e) - np.array(s)
        ratio = amount/np.linalg.norm(vec[[0,2]])
        unit = vec*ratio
        times = int(1/ratio)
        for i in range(times):
            if i != times - 1:
                location = np.array(new_path[-1])+unit
                new_path.append(location.tolist())
        new_path.append(e)
    
    return new_path

def get_camera_orientations12():
    base_angle_deg = 30
    base_angle_rad = math.pi / 6
    orient_dict = {}
    for k in range(1,12):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict