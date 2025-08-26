脚本在/new_data/dataset/openvla-scripts下
# 1. 统一基本hdf5格式
检查现有的hdf5格式，如果与preprocess程序期望的hdf5格式相同，则跳过这个转换步骤。
这里默认使用robotwin生成的仿真数据集进行转换。
期望格式如下：
```shell
HDF5文件: /new_data/sim_to_real_hdf5/open_laptop/episode_0.hdf5
============================================================
文件属性:
  @compress: False
  @sim: False

Group: /
  @compress: False
  @sim: False
  └─ action
    Dataset: /action
      Shape: (278, 14)
      Dtype: float32

  └─ observations
    Group: /observations
      └─ images
        Group: /observations/images
          └─ cam1
            Dataset: /observations/images/cam1
              Shape: (278, 480, 640, 3)
              Dtype: uint8

          └─ cam2
            Dataset: /observations/images/cam2
              Shape: (278, 480, 640, 3)
              Dtype: uint8

          └─ cam3
            Dataset: /observations/images/cam3
              Shape: (278, 480, 640, 3)
              Dtype: uint8


      └─ qpos
        Dataset: /observations/qpos
          Shape: (278, 14)
          Dtype: float32



============================================================
分析完成
```
运行位于openvla-scripts/pre_hdf5.py可以完成转换：
```shell
python pre_hdf5.py \
     --src_dir /new_data/data_robotwin \
     --dst_dir /new_data/sim_to_real_hdf5 \
     --task_list /new_data/dataset/scripts/data-utils/openvla-dataset/task.json \
     --task adjust_bottle # 有则转换单个任务，无则默认批量任务
```
在openvla-scripts/task.json里面存储了有待处理的数据信息，格式如下：
```json
    {
    "task_name":"adjust_bottle",
    "task_type":"demo_randomized",
    "task_source":"sim"
    },
```
# 2. preprocess过程划分训练/测试集
进入preprocess过程，主要是想要将hdf5数据集按照rlds的需要进行重新整理，并且分割出需要的train和eval集，将hdf5放进命名正确（双臂任务命名需要带上aloha）的文件夹下。
由于robotwin生成数据时的instruction缺乏azure api，我们自行将robotwin的所有instruction整理成了新的json文件集合。preprocess过程中，我们将会将这些instruction随机加入生成的hdf5数据集中。
运行以下脚本即可批量划分数据集：
```shell
bash devide_hdf5_dataset.sh
```
转换数据路径写在shell脚本内部，可以手动修改：
```shell
TASK_LIST="/new_data/dataset/openvla-scripts/task.json" # 转换数据任务清单
DATASET_BASE_DIR="/new_data/data_robotwin_real_hdf5" # 格式正确的hdf5
INSTRUCTION_BASE_DIR="/new_data/data_robotwin" # robotwin数据内生成好的instruction
OUTPUT_BASE_DIR="../openvla-aloha-preprocessed" # 输出数据文件夹
PERCENT_VAL="0.05"
```
# 3. 批量执行rlds build程序
## 3.1. rlds需要额外配置环境
参考openvla官方给出的方法：https://github.com/kpertsch/rlds_dataset_builder#
将rlds_dataset_builder克隆到本地之后：
```shell
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```
即配置完毕。

## 3.2. 运行convert_rlds.sh，即可批量执行转换rlds工作。
```shell
bash conver_rlds.sh
```
脚本执行的工作：
1. 根据范例my_aloha_sim_example文件夹内的文件，复制、修改，生成对应任务的类和程序于tmp_scripts文件夹下。
2. 定位到新文件夹内，执行tfds build --overwrite --data_dir $RLDS_DATA_DIR

脚本可修改参数如下：
```shell
TASK_LIST="/new_data/dataset/openvla-scripts/task.json" # task定义文件，用于明确需要转换的task名称和性质
OUTPUT_BASE_DIR="/new_data/dataset/openvla_aloha_preprocessed" # 划分好数据集的位置
SCRIPT_BASE_DIR="/new_data/dataset/openvla-scripts/" # my_aloha_sim_example文件夹 的父文件夹
RLDS_DATA_DIR="/new_data/dataset/openvla-rlds-new" # 生成的rlds数据大地址
PROXY_IP="192.168.1.148" # 代理IP
```
