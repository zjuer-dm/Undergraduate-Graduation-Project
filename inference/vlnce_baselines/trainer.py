# 文件名: gpu_placeholder.py
import torch
import torch.distributed as dist
import os
import time

def train():
    """
    初始化DDP
    """
    # 1. 初始化DDP环境
    # torch.distributed.launch 会自动设置 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    # 我们只需要从环境变量中获取 local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    
    # 将当前进程绑定到对应的GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 获取全局rank和world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("="*60)
        print(f"Trainer started on {world_size} GPU(s).")
        print("="*60)

    GB_to_occupy = 16
    bytes_per_gb = 1024 ** 3
    elements_per_gb_float32 = bytes_per_gb // 4

    try:
        tensor_list = []
        for i in range(GB_to_occupy):
            tensor_list.append(torch.randn(elements_per_gb_float32, device=device))
            time.sleep(0.1)
        
        # 在主进程上打印显存占用情况
        if rank == 0:
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            print(f"GPU {rank}: Successfully allocated. Memory Allocated: {allocated_gb:.2f} GB | Memory Reserved: {reserved_gb:.2f} GB")

    except RuntimeError as e:
        if rank == 0:
            print(f"\n[ERROR] GPU {rank}: Failed to allocate {GB_to_occupy} GB. Your GPU might not have enough free memory.")
            print(f"Error message: {e}")
            print("Placeholder will exit.")
        # 确保所有进程都看到了错误并能一起退出
        dist.barrier()
        dist.destroy_process_group()
        return

    # 等待所有进程都完成显存分配
    dist.barrier()

    # 3. 开始无意义的计算（无限循环）
    if rank == 0:
        print("\nStarting infinite computation loop...")
    
    try:
        idx = 0
        while True:
            # 执行一些简单的、需要动用计算核心的操作
            # 这里的操作会就地修改张量，防止创建新张量导致显存溢出
            with torch.no_grad():
                tensor_list[idx] = torch.sin(tensor_list[idx]) + torch.cos(tensor_list[(idx + 1) % GB_to_occupy])
            
            idx = (idx + 1) % GB_to_occupy
            
            # 可以加入一个极短的休眠，如果不需要GPU 100%满载
            # time.sleep(0.001)

    except KeyboardInterrupt:
        if rank == 0:
            print("\nCtrl+C detected. Cleaning up and exiting...")
    finally:
        # 4. 清理DDP进程组
        dist.destroy_process_group()
        if rank == 0:
            print("All processes terminated gracefully.")

if __name__ == "__main__":
    train()