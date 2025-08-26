#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path

def get_task_names(data_root_path):
    """从数据根目录获取所有任务名称"""
    data_path = Path(data_root_path)
    if not data_path.exists():
        print(f"错误: 路径 {data_root_path} 不存在")
        return []
    
    task_names = []
    for item in data_path.iterdir():
        if item.is_dir():
            task_names.append(item.name)
    
    print(f"找到 {len(task_names)} 个任务: {task_names}")
    return task_names

def load_source_instructions(task_name, source_dir="./task_instruction"):
    """加载源指令文件"""
    source_file = Path(source_dir) / f"{task_name}.json"
    
    if not source_file.exists():
        print(f"警告: 源文件 {source_file} 不存在，跳过任务 {task_name}")
        return None
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'seen' not in data or 'unseen' not in data:
            print(f"警告: {source_file} 缺少 'seen' 或 'unseen' 字段")
            return None
            
        return data
    except Exception as e:
        print(f"错误: 读取 {source_file} 失败: {e}")
        return None

def generate_episode_file(seen_instructions, unseen_instructions, output_path, episode_num):
    """生成单个episode文件"""
    # 随机选择5个seen和5个unseen指令
    selected_seen = random.sample(seen_instructions, min(5, len(seen_instructions)))
    selected_unseen = random.sample(unseen_instructions, min(5, len(unseen_instructions)))
    
    episode_data = {
        "seen": selected_seen,
        "unseen": selected_unseen
    }
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)
    
    print(f"生成: {output_path}")

def generate_all_episodes(task_name, source_data, output_base_path):
    """为单个任务生成所有episode文件"""
    seen_instructions = source_data['seen']
    unseen_instructions = source_data['unseen']
    
    # 检查指令数量是否足够
    if len(seen_instructions) < 5:
        print(f"警告: 任务 {task_name} 的seen指令少于5个 ({len(seen_instructions)})")
        return
    
    if len(unseen_instructions) < 5:
        print(f"警告: 任务 {task_name} 的unseen指令少于5个 ({len(unseen_instructions)})")
        return
    
    # 生成100个episode文件
    for episode_num in range(100):
        output_path = output_base_path / f"episode{episode_num}.json"
        generate_episode_file(seen_instructions, unseen_instructions, output_path, episode_num)

def main():
    # 配置路径
    data_root_path = "/new_data/data_robotwin"
    source_dir = ""
    
    # 获取所有任务名称
    task_names = get_task_names(data_root_path)
    
    if not task_names:
        print("未找到任务，退出程序")
        return
    
    # 为每个任务生成episode文件
    for task_name in task_names:
        print(f"\n处理任务: {task_name}")
        
        # 加载源指令数据
        source_data = load_source_instructions(task_name, source_dir)
        if source_data is None:
            continue
        
        # 设置输出路径
        output_base_path = Path(data_root_path) / task_name / "demo_randomized" / "instructions"
        
        # 生成所有episode文件
        generate_all_episodes(task_name, source_data, output_base_path)
        
        print(f"任务 {task_name} 完成，生成了100个episode文件")
    
    print(f"\n全部完成！处理了 {len(task_names)} 个任务")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性（可选）
    # random.seed(42)
    
    main()