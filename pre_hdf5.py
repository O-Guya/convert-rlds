#!/usr/bin/env python3
"""
将仿真HDF5数据转换为目标格式的转换脚本
"""

import h5py
import numpy as np
import os
import glob
from PIL import Image
import io
import argparse
from tqdm import tqdm
import json

def decode_image_from_bytes(image_bytes):
    """从字节数据解码图像"""
    try:
        # 尝试直接从字节解码
        image = Image.open(io.BytesIO(image_bytes))
        # 转换为RGB格式并调整为目标尺寸 (480, 640, 3)
        image = image.convert('RGB')
        image = image.resize((640, 480))
        return np.array(image, dtype=np.uint8)
    except Exception as e:
        print(f"图像解码失败: {e}")
        # 返回黑色图像作为fallback
        return np.zeros((480, 640, 3), dtype=np.uint8)


def convert_episode(src_path, dst_path):
    """转换单个episode文件"""
    try:
        with h5py.File(src_path, 'r') as src_file:
            # 创建输出目录
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            with h5py.File(dst_path, 'w') as dst_file:
                # 设置文件属性
                dst_file.attrs['compress'] = False
                dst_file.attrs['sim'] = False
                
                # 获取episode长度
                qpos_data = src_file['qpos'][:]
                episode_length = qpos_data.shape[0]
                
                # 1. 创建action数据集 (与qpos相同)
                action_data = qpos_data.astype(np.float32)
                dst_file.create_dataset('action', data=action_data)
                
                # 2. 创建observations组
                obs_group = dst_file.create_group('observations')
                
                # 2.1 添加qpos到observations
                obs_group.create_dataset('qpos', data=qpos_data.astype(np.float32))
                
                # 2.2 创建images组
                images_group = obs_group.create_group('images')
                
                # 相机映射：sim格式 -> 目标格式
                camera_mapping = {
                    'head_camera': 'cam1', 
                    'left_camera': 'cam2',
                    'right_camera': 'cam3'
                }
                
                # 处理每个相机的RGB数据
                for sim_camera, target_camera in camera_mapping.items():
                    if sim_camera in src_file['observation']:
                        camera_group = src_file['observation'][sim_camera]
                        if 'rgb' in camera_group:
                            rgb_data = camera_group['rgb'][:]
                            
                            # 初始化图像数组
                            images_array = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
                            
                            # 解码每一帧图像
                            for i in range(episode_length):
                                if i < len(rgb_data):
                                    image_bytes = rgb_data[i]
                                    if isinstance(image_bytes, bytes):
                                        images_array[i] = decode_image_from_bytes(image_bytes)
                                    else:
                                        # 如果是字符串，先编码为字节
                                        try:
                                            image_bytes = image_bytes.encode() if isinstance(image_bytes, str) else image_bytes
                                            images_array[i] = decode_image_from_bytes(image_bytes)
                                        except:
                                            print(f"警告: {sim_camera} 第 {i} 帧图像处理失败，使用黑色图像")
                                            images_array[i] = np.zeros((480, 640, 3), dtype=np.uint8)
                            
                            # 只有当相机有有效数据时才创建数据集
                            if target_camera in ['cam1', 'cam2', 'cam3']:  # 只保留前3个相机
                                images_group.create_dataset(target_camera, data=images_array)
                                print(f"  转换 {sim_camera} -> {target_camera}: {images_array.shape}")
                
                print(f"成功转换: {src_path} -> {dst_path}")
                
    except Exception as e:
        print(f"转换失败 {src_path}: {e}")
        # 删除部分创建的文件
        if os.path.exists(dst_path):
            os.remove(dst_path)


def convert_task(src_base_dir, dst_base_dir, task_info):
    """转换指定任务的所有episode"""
    task_name = task_info['task_name'] # name
    task_type = task_info['task_type'] # demo_randomized / demo_clean
    task_source = task_info['task_source'] # sim / real

    new_task_name = f"{task_source}_{task_name}_{task_type}"

    src_task_dir = os.path.join(src_base_dir, task_name, task_type, 'data')
    dst_task_dir = os.path.join(dst_base_dir, new_task_name)

    # 查找所有episode文件
    episode_files = glob.glob(os.path.join(src_task_dir, 'episode*.hdf5'))
    episode_files.sort()
    
    if not episode_files:
        print(f"在 {src_task_dir} 中未找到episode文件")
        return
    
    print(f"开始转换任务: {task_name}")
    print(f"找到 {len(episode_files)} 个episode文件")
    
    for src_file in tqdm(episode_files, desc=f"转换 {task_name}"):
        episode_name = os.path.basename(src_file).replace('episode', 'episode_')
        dst_file = os.path.join(dst_task_dir, episode_name)
        convert_episode(src_file, dst_file)

def load_task_info(task_json_path):
    try:
        with open(task_json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        return tasks
    except Exception as e:
        print(f"读取任务列表失败{e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='将仿真HDF5格式转换为目标格式')
    parser.add_argument('--src_dir', type=str, default='/new_data/data_robotwin',
                       help='源数据目录路径')
    parser.add_argument('--dst_dir', type=str, default='/new_data/data_robotwin_real_hdf5',
                       help='目标数据目录路径')
    parser.add_argument('--task_list', type=str, default='/new_data/dataset/openvla-scripts/task.json',
                    help='任务列表JSON文件路径')
    parser.add_argument('--task', type=str, default=None,
                    help='指定要转换的任务名称，如果不指定则转换所有任务')
    parser.add_argument('--list_tasks', action='store_true',
                    help='列出所有可用的任务')
    
    args = parser.parse_args()
    
    # 列出所有任务
    if args.list_tasks:
        tasks = load_task_info(args.task_list)
        print("可用任务：")
        for task in tasks:
            new_name = f"{task['task_source']}_{task['task_name']}_{task['task_type']}"
            print(f"task:{task['task_name']} ➡️ {new_name}")
        return

    # 创建目标目录
    os.makedirs(args.dst_dir, exist_ok=True)
    
    tasks = load_task_info(args.task_list)

    if not tasks:
        print("没有任何有效任务")
        return

    if args.task:
        # 转换指定任务
        task_info = next((t for t in tasks if t['task_name'] == args.task), None)
        if task_info:
            convert_task(args.src_dir, args.dst_dir, task_info)
        else:
            print(f"未找到任务: {args.task}")
    else:
        # 转换所有任务
        print(f"从JSON文件加载了 {len(tasks)} 个任务")
        for task_info in tasks:
            convert_task(args.src_dir, args.dst_dir, task_info)
        
        print("转换完成!")


if __name__ == '__main__':
    main()