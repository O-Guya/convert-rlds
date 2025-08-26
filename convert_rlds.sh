#!/bin/bash

# 一次处理一个任务的RLDS构建脚本
# 功能: 读取task.json -> 生成脚本 -> 构建TFDS -> 删除脚本

# 默认参数
TASK_LIST="/new_data/dataset/openvla-scripts/task.json" # task定义文件，用于明确task名称和性质
OUTPUT_BASE_DIR="/new_data/dataset/openvla_aloha_preprocessed" # 划分好数据集的位置
SCRIPT_BASE_DIR="/new_data/dataset/openvla-scripts/" # my_aloha_sim_example文件夹 的父文件夹
RLDS_DATA_DIR="/new_data/dataset/openvla-rlds-new" # 生成的rlds数据大地址
PROXY_IP="192.168.1.148" # 代理IP

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
    echo -e "${RED}错误: 任务列表文件 $TASK_LIST 不存在${NC}"
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}开始逐个构建RLDS数据集${NC}"
echo -e "${BLUE}任务列表: $TASK_LIST${NC}"
echo -e "${BLUE}============================================${NC}"

# 从JSON文件中提取任务信息的函数
extract_tasks() {
    grep -o '"task_name":[^,}]*' "$TASK_LIST" | sed 's/"task_name":[[:space:]]*"//' | sed 's/"//' > /tmp/task_names.tmp
    grep -o '"task_type":[^,}]*' "$TASK_LIST" | sed 's/"task_type":[[:space:]]*"//' | sed 's/"//' > /tmp/task_types.tmp
    grep -o '"task_source":[^,}]*' "$TASK_LIST" | sed 's/"task_source":[[:space:]]*"//' | sed 's/"//' > /tmp/task_sources.tmp
}

# 清理函数
cleanup() {
    rm -f /tmp/task_names.tmp /tmp/task_types.tmp /tmp/task_sources.tmp
}

# 处理单个任务的函数
process_single_task() {
    local task_name="$1"
    local task_type="$2"
    local task_source="$3"
    
    local full_task_name="${task_source}_${task_name}_${task_type}"
    local preprocessed_name="aloha_preprocessed_$full_task_name"
    local preprocessed_path="$OUTPUT_BASE_DIR/$preprocessed_name"
    
    echo -e "${BLUE}==================== 开始处理: $full_task_name ====================${NC}"
    
    # 检查预处理数据是否存在
    if [[ ! -d "$preprocessed_path" ]]; then
        echo -e "${RED}  跳过: 预处理数据路径不存在 $preprocessed_path${NC}"
        return 1
    fi
    
    # 步骤1: 生成RLDS脚本
    echo -e "${YELLOW}步骤1: 生成RLDS脚本${NC}"
    local source_dir="$SCRIPT_BASE_DIR/my_aloha_sim_example"
    local target_dir="$SCRIPT_BASE_DIR/tmp_scripts/my_aloha_$full_task_name"
    
    # 创建目标文件夹
    if [[ -d "$target_dir" ]]; then
        echo -e "  删除已存在的目录: $target_dir"
        rm -rf "$target_dir"
    fi
    mkdir -p "$target_dir"
    
    # 复制源文件
    cp -r "$source_dir"/* "$target_dir/"
    echo -e "  ✓ 复制脚本文件"
    
    # 重命名文件
    local old_builder="$target_dir/my_aloha_sim_example_dataset_builder.py"
    local new_builder="$target_dir/my_aloha_${full_task_name}_dataset_builder.py"
    mv "$old_builder" "$new_builder"
    echo -e "  ✓ 重命名构建器文件"
    
    # 修改文件内容
    sed -i "s|'train': glob\.glob('[^']*')|'train': glob.glob(f'$preprocessed_path/$full_task_name/train/episode_*.hdf5')|g" "$new_builder"
    sed -i "s|'val': glob\.glob('[^']*')|'val': glob.glob(f'$preprocessed_path/$full_task_name/val/episode_*.hdf5')|g" "$new_builder"
    sed -i "s/class my_aloha_sim_example/class my_aloha_${preprocessed_name}/g" "$new_builder"
    sed -i "s/my_aloha_sim_example/my_aloha_${full_task_name}/g" "$new_builder"
    sed -i "s/192\.168\.1\.91/$PROXY_IP/g" "$new_builder"
    echo -e "  ✓ 修改脚本内容"
    
    # 步骤2: 构建TFDS数据集
    echo -e "${YELLOW}步骤2: 构建TFDS数据集${NC}"
    cd "$target_dir"
    echo -e "  当前目录: $(pwd)"
    echo -e "  执行: tfds build --overwrite --data_dir $RLDS_DATA_DIR"
    
    if tfds build --overwrite --data_dir "$RLDS_DATA_DIR"; then
        echo -e "${GREEN}  ✓ TFDS构建成功${NC}"
        cd - > /dev/null
        
        # 步骤3: 删除生成的脚本
        echo -e "${YELLOW}步骤3: 清理临时脚本${NC}"
        rm -rf "$target_dir"
        echo -e "  ✓ 删除脚本目录: $target_dir"
        
        echo -e "${GREEN}✅ 任务 $full_task_name 处理完成！${NC}"
        return 0
    else
        echo -e "${RED}  ✗ TFDS构建失败${NC}"
        cd - > /dev/null
        
        # 构建失败时保留脚本用于调试
        echo -e "${YELLOW}  保留脚本目录用于调试: $target_dir${NC}"
        return 1
    fi
}

# 主处理逻辑
extract_tasks
task_count=$(wc -l < /tmp/task_names.tmp)

echo -e "${BLUE}找到 $task_count 个任务${NC}"

total_tasks=0
success_count=0
fail_count=0

# 处理每个任务
for ((i=1; i<=task_count; i++)); do
    task_name=$(sed -n "${i}p" /tmp/task_names.tmp)
    task_type=$(sed -n "${i}p" /tmp/task_types.tmp)
    task_source=$(sed -n "${i}p" /tmp/task_sources.tmp)
    
    total_tasks=$((total_tasks + 1))
    
    echo ""
    echo -e "${BLUE}[$i/$task_count] 处理任务: ${task_source}_${task_name}_${task_type}${NC}"
    
    if process_single_task "$task_name" "$task_type" "$task_source"; then
        success_count=$((success_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
    
    echo -e "${BLUE}==================== 完成: ${task_source}_${task_name}_${task_type} ====================${NC}"
done

# 清理临时文件
cleanup

# 输出最终统计
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}处理完成统计${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}总任务数: $total_tasks${NC}"
echo -e "${GREEN}成功处理: $success_count${NC}"
echo -e "${RED}处理失败: $fail_count${NC}"
echo -e "${BLUE}============================================${NC}"

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}🎉 所有任务处理成功！${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  部分任务处理失败，请检查日志${NC}"
    exit 1
fi