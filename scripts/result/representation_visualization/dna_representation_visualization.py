import io
import os
import cv2
import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import gc
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.model.dna_encoder import get_sequence_pipeline
from bioscanclip.util.dataset import tokenize_dna_sequence
from PIL import Image, ImageDraw, ImageFont
import random

# Reference and modified fromhttps://github.com/jacobgil/vit-explain


def get_attention_mask(dna_list, dna_tokens_list, attn_rollout, device, layer_idx=None,
                                  folder_name="representation_visualization/before_contrastive_learning"):
    os.makedirs(folder_name, exist_ok=True)
    mask_list = []

    for idx, tokens in tqdm(enumerate(dna_tokens_list), total=len(dna_tokens_list)):
        with torch.no_grad():
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            mask = attn_rollout(tokens, layer_idx=layer_idx)
            # normalize mask
            mask = mask / np.max(mask)
            mask_list.append(mask)
    attn_rollout.remove_hooks()
    torch.cuda.empty_cache()
    gc.collect()
    return mask_list

def rollout(attentions, discard_ratio, head_fusion="max", layer_idx=None):

    result = torch.eye(attentions[0].size(-1))

    if layer_idx is None:
        all_attentions = attentions[1:]
    else:
        all_attentions = attentions[layer_idx:layer_idx + 1]
        # all_attentions = attentions[0:layer_idx]+attentions[layer_idx + 1:]

    with torch.no_grad():
        for attention in all_attentions:
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
    mask = mask.cpu().numpy()
    mask = mask / np.max(mask)
    return mask

class BarcodeBERTAttentionRollout:
    def __init__(self, model, attention_layer_name='attention.self.dropout', head_fusion="min",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.hooks = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                print(f"Registering hook for {name}")
                hook = module.register_forward_hook(self.get_attention)
                self.hooks.append(hook)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu().detach())

    def __call__(self, input_tensor, layer_idx=None):
        self.attentions.clear()
        with torch.no_grad():
            output = self.model(input_tensor)
        result = rollout(self.attentions, self.discard_ratio, self.head_fusion, layer_idx)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def __del__(self):
        self.remove_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_some_dna_from_hdf5(hdf5_group, n=None):
    dna_list = []
    total = len(hdf5_group["barcode"])
    if n is None or n > total:
        n = total

    indices = random.sample(range(total), n)

    for idx in indices:
        dna_barcode = hdf5_group["barcode"][idx].decode("utf-8")
        dna_list.append(dna_barcode)

    sequence_pipeline = get_sequence_pipeline()
    unprocessed_dna_barcode = np.array(dna_list)
    dna_tokens_list = tokenize_dna_sequence(sequence_pipeline, unprocessed_dna_barcode)
    return dna_list, dna_tokens_list

def draw_dna_barcode(dna_barcode_list, mask_list, save_path):

    font = ImageFont.load_default()
    bbox = font.getbbox("A")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    line_spacing = 5

    margin_top = 10
    margin_bottom = 10
    margin_left = 10
    img_width = 4200
    img_height = margin_top + margin_bottom + len(dna_barcode_list) * (text_height + line_spacing)

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    y = margin_top

    for i, line in enumerate(dna_barcode_list):
        tokens = [line[j:j+5] for j in range(0, len(line), 5)]
        token_masks = mask_list[i]
        x = margin_left
        for token, att in zip(tokens, token_masks):
            token_width = text_width * len(token)
            r = 255
            g = int(255 * (1 - att))
            b = int(255 * (1 - att))
            bg_color = (r, g, b)
            draw.rectangle([x, y, x + token_width, y + text_height], fill=bg_color)
            draw.text((x, y), token, fill=(0, 0, 0), font=font)
            x += token_width
        y += text_height + line_spacing

    os.makedirs(save_path, exist_ok=True)
    img.show()
    img.save(os.path.join(save_path, "sample_dna_barcode.png"))

@hydra.main(config_path="../../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    save_path = os.path.join(args.project_root_path, "dna_representation_visualization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load some dna from the hdf5 file
    if args.model_config.dataset == "bioscan_5m":
        path_to_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    else:
        path_to_hdf5 = args.bioscan_data.path_to_hdf5_data
    # Open the hdf5 file

    with  h5py.File(path_to_hdf5, "r", libver="latest") as hdf5_file:
        # For now just use train_seen data
        hdf5_group = hdf5_file["train_seen"]
        # Load some dna barcode and processed tokens.
        dna_list, dna_tokens_list = get_some_dna_from_hdf5(hdf5_group, n=50)


    print("Initialize model...")
    model = load_clip_model(args, device)
    model.eval()

    # Get the dna encoder
    dna_encoder = model.dna_encoder
    dna_encoder.eval()
    dna_encoder.to(device)

    # print(dna_encoder)
    print(dna_encoder)

    # draw_dna_barcode(dna_list, dna_tokens_list, save_path)

    for single_layer in dna_encoder.base_dna_encoder.bert.encoder.layer:
        single_layer.attention.self.fused_attn = False
        # single_layer.attention.fused_attn = False

    print("Before contrastive learning")
    attn_rollout = BarcodeBERTAttentionRollout(dna_encoder, attention_layer_name='attention.self.dropout',
                                               head_fusion="max", )
    attention_mask_list = get_attention_mask(dna_list, dna_tokens_list, attn_rollout, device)

    draw_dna_barcode(dna_list, attention_mask_list, save_path)

    print("After contrastive learning")
    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)

    model.eval()

    dna_encoder = model.dna_encoder
    dna_encoder.eval()
    dna_encoder.to(device)
    for single_layer in dna_encoder.base_dna_encoder.bert.encoder.layer:
        single_layer.attention.self.fused_attn = False
        # single_layer.attention.fused_attn = False
    attn_rollout = BarcodeBERTAttentionRollout(dna_encoder, attention_layer_name='attention.self.dropout',
                                               head_fusion="max", )
    attention_mask_list = get_attention_mask(dna_list, dna_tokens_list, attn_rollout, device)
    draw_dna_barcode(dna_list, attention_mask_list, save_path)


if __name__ == "__main__":
    main()