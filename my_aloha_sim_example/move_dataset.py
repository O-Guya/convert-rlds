import os
import shutil
import re

def get_next_episode_index(target_dir):
    """获取目标目录中现有 episode 文件的最大编号 + 1"""
    episode_files = [f for f in os.listdir(target_dir) if f.startswith("episode_") and f.endswith(".hdf5")]
    indices = []
    for fname in episode_files:
        match = re.match(r"episode_(\d+)\.hdf5", fname)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 0

def copy_and_rename_episodes(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    next_idx = get_next_episode_index(dst_dir)
    
    episode_files = sorted(f for f in os.listdir(src_dir) if f.startswith("episode_") and f.endswith(".hdf5"))
    for f in episode_files:
        new_fname = f"episode_{next_idx}.hdf5"
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, new_fname))
        print(f"Copied {f} → {new_fname}")
        next_idx += 1

# 源和目标目录
base_src = "/data/dataset/datasets-openvla-oft/new_dataset/aloha_preprocess_hdf5/pick_put_banana_0604"
base_dst = "/data/dataset/datasets-openvla-oft/new_dataset_banana2/aloha_preprocess_hdf5/pick_put_banana_0616"

# 处理 train 和 val 子目录
for split in ["train", "val"]:
    src = os.path.join(base_src, split)
    dst = os.path.join(base_dst, split)  # 注意：现在加在路径末尾，不是拼接字符串
    copy_and_rename_episodes(src, dst)
