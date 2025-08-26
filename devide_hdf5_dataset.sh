#!/bin/bash

# 一键把所有的sim_to_real_hdf5数据集转换为ALOHA预处理格式

# 默认参数
TASK_LIST="/new_data/dataset/openvla-scripts/task.json" # 转换数据任务清单
DATASET_BASE_DIR="/new_data/data_robotwin_real_hdf5" # 格式正确的hdf5
INSTRUCTION_BASE_DIR="/new_data/data_robotwin" # robotwin数据内生成好的instruction
OUTPUT_BASE_DIR="../openvla-aloha-preprocessed" # 输出数据文件夹
PERCENT_VAL="0.05"

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -t, --task-list FILE    任务列表JSON文件路径 (默认: $TASK_LIST)"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                              # 使用默认设置"
    echo "  $0 -t /path/to/tasks.json      # 指定任务列表文件"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--task-list)
            TASK_LIST="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查任务列表文件
if [[ ! -f "$TASK_LIST" ]]; then
    echo "错误: 任务列表文件 $TASK_LIST 不存在"
    exit 1
fi

echo "开始处理任务列表: $TASK_LIST"
echo "========================================="

# 从JSON文件中提取任务信息的函数
extract_tasks() {
    # 使用grep和sed从JSON中提取task信息
    grep -o '"task_name":[^,}]*' "$TASK_LIST" | sed 's/"task_name":[[:space:]]*"//' | sed 's/"//' > /tmp/task_names.tmp
    grep -o '"task_type":[^,}]*' "$TASK_LIST" | sed 's/"task_type":[[:space:]]*"//' | sed 's/"//' > /tmp/task_types.tmp
    grep -o '"task_source":[^,}]*' "$TASK_LIST" | sed 's/"task_source":[[:space:]]*"//' | sed 's/"//' > /tmp/task_sources.tmp
}

# 提取任务信息
extract_tasks

# 读取任务数量
task_count=$(wc -l < /tmp/task_names.tmp)

# 处理每个任务
for ((i=1; i<=task_count; i++)); do
    task_name=$(sed -n "${i}p" /tmp/task_names.tmp)
    task_type=$(sed -n "${i}p" /tmp/task_types.tmp)
    task_source=$(sed -n "${i}p" /tmp/task_sources.tmp)
    
    # 构建路径
    dataset_path="$DATASET_BASE_DIR/${task_source}_${task_name}_${task_type}"
    instruction_path="$INSTRUCTION_BASE_DIR/${task_name}/${task_type}/instructions"
    output_name="aloha_preprocessed_${task_source}_${task_name}_${task_type}"
    output_path="$OUTPUT_BASE_DIR/$output_name"
    
    echo "Processing task: ${task_source}_${task_name}_${task_type}"
    
    # 检查路径是否存在
    if [[ ! -d "$dataset_path" ]]; then
        echo "  跳过: 数据集路径不存在 $dataset_path"
        continue
    fi
    
    if [[ ! -d "$instruction_path" ]]; then
        echo "  跳过: 指令路径不存在 $instruction_path"
        continue
    fi
    
    # 执行预处理命令
    if python ./my_aloha_sim_example/preprocess_split_aloha_data.py \
        --dataset_path "$dataset_path" \
        --instruction_path "$instruction_path" \
        --out_base_dir "$output_path" \
        --percent_val "$PERCENT_VAL"; then
        echo "  ✓ 完成: ${task_source}_${task_name}_${task_type}"
    else
        echo "  ✗ 失败: ${task_source}_${task_name}_${task_type}"
    fi
done

# 清理临时文件
rm -f /tmp/task_names.tmp /tmp/task_types.tmp /tmp/task_sources.tmp

echo "========================================="
echo "处理完成!"