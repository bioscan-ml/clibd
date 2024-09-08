import h5py
import io
import json
import os
import csv
import random
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
import faiss
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import silhouette_samples
from umap import UMAP
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from bioscanclip.epoch.inference_epoch import get_feature_and_label
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import Table, categorical_cmap

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"
All_TYPE_OF_FEATURES_OF_QUERY = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    "averaged_feature",
    "concatenated_feature",
]
All_TYPE_OF_FEATURES_OF_KEY = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    "averaged_feature",
    "concatenated_feature",
    "all_key_features",
]
LEVELS = ["order", "family", "genus", "species"]

# Reference and modified fromhttps://github.com/jacobgil/vit-explain/blob/main/vit_grad_rollout.py
def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def encode_all_image(image_list, model, transform, device):
    for image in image_list:
        image = transform(image).unsqueeze(0).to(device)
        image_output = F.normalize(model(image), p=2, dim=-1)

def get_some_images_from_hdf5(hdf5_group, n=100):
    image_list = []
    for idx in range(n):
        image_enc_padded = hdf5_group["image"][idx].astype(np.uint8)
        enc_length = hdf5_group["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc))
        image_list.append(curr_image)
    return image_list

def get_image_encoder(model, device):
    image_encoder = model.image_encoder
    image_encoder.eval()
    image_encoder.to(device)
    return image_encoder

def get_and_save_vit_explaination(image_list, grad_rollout, transform, device):
    for idx, image in enumerate(image_list):
        image.show()
        image = transform(image).unsqueeze(0).to(device)
        mask = grad_rollout(image)
        mask_255 = (mask * 255).astype(np.uint8)

        image = Image.fromarray(mask_255)
        image.show()

        plt.axis("off")
        plt.savefig(f"vit_explaination_{idx}.png")
        plt.close()
        exit()

@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load some images from the hdf5 file
    if args.model_config.dataset == "bioscan_5m":
        path_to_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    else:
        path_to_hdf5 = args.bioscan_data.path_to_hdf5_data

    # Init transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=256, antialias=True),
            transforms.CenterCrop(224),
        ]
    )

    # Open the hdf5 file
    hdf5_file = h5py.File(path_to_hdf5, "r", libver="latest")
    # For now just use train_seen data
    hdf5_group = hdf5_file["train_seen"]

    # Load some images from the hdf5 file
    image_list = get_some_images_from_hdf5(hdf5_group)

    print("Initialize model...")
    model = load_clip_model(args, device)
    model.eval()
    # Get the image encoder
    image_encoder = get_image_encoder(model, device)
    for block in image_encoder.lora_vit.blocks:
        block.attn.fused_attn = False

    # Extract feature and representation visualization for one of the image.
    # encode_all_image(image_list, image_encoder, transform, device)
    grad_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9)
    get_and_save_vit_explaination(image_list, grad_rollout, transform, device)



    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)



if __name__ == "__main__":
    main()