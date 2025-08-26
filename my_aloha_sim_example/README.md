
## aloha preprocess
```
python /data/dataset/datasets-openvla-oft/scripts/aloha_to_rlds/my_aloha_sim_adjust_bottle/preprocess_split_aloha_data.py \
  --dataset_path /new_data/sim_to_real_hdf5/classified_data/adjust_bottle/left_hand_data \
  --out_base_dir /new_data/dataset/aloha_preprocessed_sim_adjust_bottle_left \
  --percent_val 0.05

```
## 再把h5转换为rlds

切换到rlds_env环境，`conda activate rlds_env`

在`/data/dataset/datasets-openvla-oft/scripts/aloha_to_rlds/my_aloha_picking_banana_new`中运行

`tfds build --overwrite --data_dir /new_data/dataset/openvla-rlds`

ps:
出现连不上google问题，请设置代理
```
os.environ['http_proxy'] = 'http://192.168.1.82:7897'
os.environ['https_proxy'] = 'http://192.168.1.82:7897'
```