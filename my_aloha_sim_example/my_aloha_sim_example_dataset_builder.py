from typing import Iterator, Tuple, Any

import os
# os.environ["TFDS_DATA_DIR"] = "/data/dataset/datasets-openvla-oft/aloha_preprocess_hdf5/1"  # 替换为你的本地路径
# os.environ["TFDS_DISABLE_GCS"] = "1"  # 禁用 GCS 下载
# os.environ['http_proxy'] = 'http://192.168.1.50:7897'
# os.environ['https_proxy'] = 'http://192.168.1.50:7897'
import h5py
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append('.')
from .conversion_utils import MultiThreadedDatasetBuilder

os.environ['http_proxy'] = 'http://192.168.1.148:7897'
os.environ['https_proxy'] = 'http://192.168.1.148:7897'

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path):
        # Load raw data
        with h5py.File(episode_path, "r") as F:
            # breakpoint()
            actions = F['action'][:]                          # (T, 14)
            states = F['observations']['qpos'][:]             # (T, 14)
            images_cam1 = F['observations']['images']['cam1'][:]  #left # (T, 256, 256, 3)
            images_cam2 = F['observations']['images']['cam2'][:]         #top (T, 256, 256, 3)
            images_cam3 = F['observations']['images']['cam3'][:]  #right (T, 256, 256, 3)

            language_instruction = F['instruction'][()].decode('utf-8')
            # reward = F['frames']["reward"][:]
            # language_instruction = "Pick up the orange and put it in the drawer."

        # Assemble episode: here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
               'observation': {
                    'cam1': images_cam1[i],#cam_left_wrist
                    'cam2': images_cam2[i],#cam_high
                    'cam3': images_cam3[i],#cam_right_wrist
                    'state': np.asarray(states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                # 'reward': np.asarray(reward[i], dtype=np.float32),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': language_instruction,
            })

        # Create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # If you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class my_aloha_sim_example(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'cam1': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Camera 1 RGB observation.',
                        ),
                        'cam2': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Camera 2 RGB observation.',
                        ),
                        'cam3': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Camera 3 RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot state, consists of 14 joint angles (7 per arm including grippers).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot action, consists of 14 joint commands (7 per arm including grippers).',
                    ),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            'train': glob.glob('/new_data/dataset/openvla_aloha_preprocessed/openvla_aloha_preprocessed/aloha_preprocessed_sim_example/example/train/episode_*.hdf5'),
            'val': glob.glob('/new_data/dataset/openvla_aloha_preprocessed/openvla_aloha_preprocessed/aloha_preprocessed_sim_example/example/val/episode_*.hdf5'),
        }