# 2025/08/20/ UPDATED
现在非常简单了，整个工作流程就是
1. python sim_to_real_hdf5.py # 将robotwin数据格式变成了可以preprocess的格式
2. bash preprocess_by_group.sh # 批量preprocess，划分数据集，便于后续rlds直接转换
3. bash rlds_all.sh # 批量执行rlds转换脚本rlds_prepare_instruction.sh

# 三合一process
# 仅限于 robobrain 打标，没有经过打标用不到这个脚本
python integrated_aloha_processor.py \
    --dataset_path /new_data/sim_to_real_hdf5 \
    --annotations_path /new_data/dataset/openvla-brain/ \
    --out_base_dir /new_data/dataset/aloha preprocessed/ \
    --percent_val 0.05 \
    --max_episodes 1

# 以下为正常转rlds过程

conda activate rlds_env

tfds build --overwrite --data_dir /new_data/dataset/openvla-rlds

python /new_data/dataset/scripts/my_aloha_scripts/my_aloha_sim_example/preprocess_split_aloha_data.py \
  --dataset_path /new_data/sim_to_real_hdf5/adjust_bottle \
  --out_base_dir /new_data/dataset/openvla_aloha_preprocessed/aloha_preprocessed_sim_adjust_bottle_ \
  --percent_val 0.05

# hdf5转rlds创建文件可以用脚本快速创建
# {task_name},{data_path}写到train/val的上层文件夹就行

cd /new_data/dataset/scripts/data-utils
bash rlds_prepare.sh {task_name} {data_path}

bash rlds_prepare.sh sim_click_bell /new_data/dataset/aloha_preprocessed_sim_click_bell/click_bell

bash rlds_prepare.sh sim_adjust_bottle_left dataset/aloha_preprocessed_sim_adjust_bottle_left/left_hand_data

# 注意！转完每个数据，记得修改instruction
# robotwin的instruction路径为：/home/Better-oft/RoboTwin/description/task_instruction

# 以上是老版本的流程（instruction自行在class里面添加写死）
# 现在配合robotwin的random instruction测试方式，增添了在robotwin数据集里面读取instruction.json的功能，可以随机生成

cd /new_data/dataset/scripts/data-utils
bash rlds_prepare_instruction.sh {task_name} {data_path}

