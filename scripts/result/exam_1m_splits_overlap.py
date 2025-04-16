import io
import os

import cv2
import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm
import gc
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.util import update_checkpoint_param_names
import matplotlib.pyplot as plt
import random


def get_image_file_name_list(hdf5_file, split_name):
    hdf5_group = hdf5_file[split_name]
    image_file_name_list = [i.decode("utf-8") for i in hdf5_group["image_file"]]

    return image_file_name_list

def check_image_name_lists_of_key_splits(hdf5_file, split_name, no_split_and_seen_train_file_name):
    hdf5_group = hdf5_file[split_name]
    image_file_name_list = [i.decode("utf-8") for i in hdf5_group["image_file"]]

    # check if there are overlapping images
    overlap = set(image_file_name_list).intersection(set(no_split_and_seen_train_file_name))
    overlap = list(overlap)
    return overlap

@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    # Load some images from the hdf5 file
    if args.model_config.dataset == "bioscan_5m":
        path_to_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    else:
        path_to_hdf5 = args.bioscan_data.path_to_hdf5_data

    hdf5_file = h5py.File(path_to_hdf5, "r", libver="latest")

    print(f"All splits: {list(hdf5_file.keys())}")
    print(f"All kind of data: {list(hdf5_file['all_keys'].keys())}")

    # keys_file_name = get_image_file_name_list(hdf5_file, 'all_keys')
    no_split_and_seen_train_file_name = get_image_file_name_list(hdf5_file, 'no_split_and_seen_train')

    for split_name in ['all_keys', 'no_split', 'no_split_and_seen_train', 'seen_keys', 'single_species', 'test_seen', 'test_unseen', 'test_unseen_keys', 'train_seen', 'val_seen', 'val_unseen', 'val_unseen_keys']:
        if split_name == 'no_split_and_seen_train' or split_name == 'no_split' or split_name == 'train_seen':
            continue
        print(f"Checking {split_name}...")
        overlap = check_image_name_lists_of_key_splits(hdf5_file, split_name, no_split_and_seen_train_file_name)
        print(f"Number of overlapping images in {split_name}: {len(overlap)}")


if __name__ == "__main__":
    main()