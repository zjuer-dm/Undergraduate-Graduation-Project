"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

A prefetch loader to speedup data loading
Modified from Nvidia Deep Learning Examples
(https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
"""
import random
from typing import List, Dict, Tuple, Union, Iterator

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import bisect

class MetaLoader:
    # 解决的问题：如何在一个训练流中混合多个不同的训练任务（例如，mlm和sap）
    # 核心功能：“任务调度器”，内部维护了多个 DataLoader 实例（每个任务一个）。
    """wraps multiple data loaders"""

    def __init__(
        self, loaders, ratio_dict, accum_steps: int = 1, distributed: bool = False, device=None
    ):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.name2pre_epoch = {}
        self.names: List[str] = []
        # ratios: List[int] = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r, p = l
            elif isinstance(l, DataLoader):
                r = 1
                p = lambda e: None
            else:
                raise ValueError()
            self.names.append(n)
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2pre_epoch[n] = p
            # ratios.append(r)

        self.accum_steps = accum_steps
        self.device = device
        # self.sampling_ratios = torch.tensor(ratios).float().to(self.device)
        self.sorted_iters = sorted(int(k) for k in ratio_dict)
        self.ratio_list = [torch.tensor(ratio_dict[str(k)]).float().to(device) for k in self.sorted_iters]
        self.distributed = distributed
        self.step = 0

    def get_ratios(self, step):
        # 使用 bisect 查找当前 iteration 落在哪个区间
        index = bisect.bisect_right(self.sorted_iters, step) - 1
        index = max(index, 0)  # 防止 iteration < 最小 key 的情况
        return self.ratio_list[index]
    
    def __iter__(self) -> Iterator[Tuple]:
        """this iterator will run indefinitely"""
        task_id = None
        epoch_id = 0
        while True:
            if self.step % self.accum_steps == 0:
                sampling_ratios = self.get_ratios(self.step)
                # print("sampling_ratios ", sampling_ratios)
                task_id = torch.multinomial(sampling_ratios, 1)
                if self.distributed:
                    # make sure all process is training same task
                    dist.broadcast(task_id, 0)
            self.step += 1
            task = self.names[task_id.cpu().item()]
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                epoch_id += 1
                # In distributed mode, calling the set_epoch() method at the beginning of each epoch 
                # before creating the DataLoader iterator is necessary to make shuffling work properly 
                # across multiple epochs. Otherwise, the same ordering will be always used.
                self.name2pre_epoch[task](epoch_id) # pre_epoch是一个函数对象，因此该语句等于调用sampler.set_epoch(epoch_id)
                iter_ = iter(self.name2loader[task]) #为该任务创建一个新的迭代器，原先的已经迭代空了
                batch = next(iter_)
                self.name2iter[task] = iter_

            # yield的作用：暂停执行，并返回值。但函数不会终止，保留之前的状态，下一次调用会从 yield 语句后继续执行。可以在循环中无限提供数据，而不会重置状态。
            yield task, batch


def move_to_cuda(batch: Union[List, Tuple, Dict, torch.Tensor], device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True) # 关键在这里，是non_blocking=True让代码实现了并行！
    elif isinstance(batch, list):
        return [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t, device) for n, t in batch.items()}
    return batch


class PrefetchLoader(object):
    # 解决的问题：如何减少或隐藏数据从CPU内存到GPU显存的传输延迟，提升GPU利用率？到底高效在哪了？？？
    # 答：move_to_cuda中的non_blocking=True让代码实现了并行！
    # 核心功能：它扮演一个“预取缓冲区”的角色。
    """
    overlap compute and cuda data transfer
    """
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        # for step, (name, batch) in enumerate(meta_loader):执行时运行的入口
        #    for循环开始时，调用 __iter__。由于有yield，此函数返回一个生成器对象。
        #    for循环会拿到这个生成器，然后开始对它调用 next()

        #    第一次对生成器调用next()，代码从这里开始执行
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            # yield的作用：暂停执行，并返回值。但函数不会终止，保留之前的状态，下一次调用会从 yield 语句后继续执行。可以在循环中无限提供数据，而不会重置状态。
            yield batch

            #    for循环完成一次迭代后，再次对生成器调用next()，代码从这里恢复执行
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        batch = self.batch
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def build_dataloader(task, dataset, collate_fn, is_train: bool, opts):

    batch_size = opts.train_batch_size if is_train else opts.val_batch_size
    # if task == 'itm': batch_size = max(1, batch_size // 2)

    if opts.local_rank == -1:
        if is_train:
            sampler: Union[
                RandomSampler, SequentialSampler, DistributedSampler
            ] = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        if size > 1:
            batch_size *= size

    else:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=is_train
        )
        pre_epoch = sampler.set_epoch # 函数赋值，pre_epoch是一个函数对象

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=opts.n_workers,
        pin_memory=opts.pin_mem,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return loader, pre_epoch
