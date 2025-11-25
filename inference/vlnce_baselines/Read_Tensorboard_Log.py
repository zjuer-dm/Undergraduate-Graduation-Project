import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 你的 TensorBoard 日志文件路径
log_file = "/home/ysh/ETPNav/data/logs/checkpoints/release_r2r_grpo_try4_10000_ok/events.out.tfevents.1748603331.sadfgfy-System-Product-Name.3090272.0"

# 检查文件是否存在
if not os.path.exists(log_file):
    print(f"错误：文件不存在 {log_file}")
else:
    print(f"正在读取文件: {log_file}\n")
    
    # 初始化 EventAccumulator
    # size_guidance 是一个指导参数，可以设置大一些以确保加载所有数据
    # 例如，加载所有标量：EventAccumulator.SCALARS
    ea = EventAccumulator(log_file,
        size_guidance={
            'scalars': 0, # 0 表示加载所有标量数据
            'images': 0,
            'histograms': 0,
        })

    # 加载数据
    ea.Reload()

    # 获取所有标量数据的标签
    scalar_tags = ea.Tags()['scalars']
    print("文件中包含的标量标签 (Scalar Tags):")
    print(scalar_tags)
    print("-" * 30)

    # 遍历所有标签，并提取数据
    for tag in scalar_tags:
        print(f"正在处理标签: {tag}")
        
        # 获取该标签下的所有标量事件
        scalar_events = ea.Scalars(tag)
        
        # 遍历列表中的每一个事件并打印
        for event in scalar_events[:5]:
            # event 对象包含 .step 和 .value 属性
            print(f"  Step: {event.step:<8} | Value: {event.value:.6f}")
        
        print("\n")