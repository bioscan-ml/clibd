import gc
import os
import random

import h5py
import hydra
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import DictConfig
from tqdm import tqdm

from bioscanclip.model.dna_encoder import get_sequence_pipeline
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import tokenize_dna_sequence
from bioscanclip.util.util import update_checkpoint_param_names



# Reference and modified fromhttps://github.com/jacobgil/vit-explain


def get_attention_mask(top_order_barcodes, top_order_tokens, attn_rollout, device, layer_idx=None,
                       folder_name="representation_visualization/before_contrastive_learning"):
    os.makedirs(folder_name, exist_ok=True)

    top_order_attention_masks = {}

    for order, tokens_list in top_order_tokens.items():
        order_masks = []
        for tokens in tqdm(tokens_list, desc=f"Processing order {order}", total=len(tokens_list)):
            with torch.no_grad():
                tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
                mask = attn_rollout(tokens_tensor, layer_idx=layer_idx)
                mask = mask / np.max(mask)
                order_masks.append(mask)
        top_order_attention_masks[order] = order_masks

    attn_rollout.remove_hooks()
    torch.cuda.empty_cache()
    gc.collect()

    return top_order_attention_masks


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


def get_top5_orders_dna_barcodes(hdf5_group, n_order=5, n_sample=10):
    orders_data = hdf5_group["order"][...]
    barcodes_data = hdf5_group["barcode"][...]

    orders = [o.decode("utf-8") if isinstance(o, bytes) else o for o in orders_data]
    barcodes = [b.decode("utf-8") if isinstance(b, bytes) else b for b in barcodes_data]

    order_to_indices = {}
    for idx, order in enumerate(orders):
        order_to_indices.setdefault(order, []).append(idx)

    sorted_orders = sorted(order_to_indices.keys(), key=lambda key: len(order_to_indices[key]), reverse=True)
    top_orders = sorted_orders[:n_order]

    top_order_barcodes = {}
    top_order_tokens = {}

    sequence_pipeline = get_sequence_pipeline()

    for order in top_orders:
        indices = order_to_indices[order]
        sample_size = min(n_sample, len(indices))
        selected_indices = random.sample(indices, sample_size)
        selected_barcodes = [barcodes[i] for i in selected_indices]
        top_order_barcodes[order] = selected_barcodes
        unprocessed_dna_barcode = np.array(selected_barcodes)
        tokens = tokenize_dna_sequence(sequence_pipeline, unprocessed_dna_barcode)
        top_order_tokens[order] = tokens

    return top_order_barcodes, top_order_tokens


def draw_dna_barcode(top_order_barcodes, top_order_attention_masks, save_path):

    font = ImageFont.load_default()
    bbox = font.getbbox("A")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    line_spacing = 5

    try:
        order_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except IOError:
        order_font = ImageFont.load_default()

    margin_top = 10
    margin_bottom = 10
    margin_left = 10

    name_col_width = 200
    barcode_area_width = 3800
    total_width = margin_left + name_col_width + barcode_area_width + margin_left

    total_rows = 0
    for order in top_order_barcodes:
        total_rows += len(top_order_barcodes[order])
    n_groups = len(top_order_barcodes)
    separator_height = 5

    total_height = (margin_top + margin_bottom +
                    total_rows * (text_height + line_spacing) +
                    (n_groups - 1) * separator_height)

    img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(img)

    y = margin_top

    for order in top_order_barcodes:
        barcodes_list = top_order_barcodes[order]
        masks_list = top_order_attention_masks[order]
        group_start_y = y

        for i, barcode in enumerate(barcodes_list):
            tokens = [barcode[j:j + 5] for j in range(0, len(barcode), 5)]
            token_masks = masks_list[i]

            x = margin_left + name_col_width
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

        group_end_y = y - line_spacing
        center_y = (group_start_y + group_end_y) // 2

        order_text = order
        order_bbox = order_font.getbbox(order_text)
        order_text_width = order_bbox[2] - order_bbox[0]
        order_text_height = order_bbox[3] - order_bbox[1]
        order_x = margin_left + (name_col_width - order_text_width) // 2
        order_y = center_y - order_text_height // 2
        draw.text((order_x, order_y), order_text, fill=(0, 0, 0), font=order_font)

        if y + separator_height < total_height:
            draw.rectangle([margin_left, y, total_width - margin_left, y + separator_height],
                           fill=(0, 0, 0))
            y += separator_height

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
        top_order_barcodes, top_order_tokens = get_top5_orders_dna_barcodes(hdf5_group, n_order=5, n_sample=10)

    print("Initialize model...")
    model = load_clip_model(args, device)
    model.eval()

    # Get the dna encoder
    dna_encoder = model.dna_encoder
    dna_encoder.eval()
    dna_encoder.to(device)

    # print(dna_encoder)
    print(dna_encoder)

    for single_layer in dna_encoder.base_dna_encoder.bert.encoder.layer:
        single_layer.attention.self.fused_attn = False
        # single_layer.attention.fused_attn = False

    print("Before contrastive learning")
    attn_rollout = BarcodeBERTAttentionRollout(dna_encoder, attention_layer_name='attention.self.dropout',
                                               head_fusion="max", )
    top_order_attention_masks = get_attention_mask(top_order_barcodes, top_order_tokens, attn_rollout, device)

    draw_dna_barcode(top_order_barcodes, top_order_attention_masks, save_path)

    print("After contrastive learning")
    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        checkpoint = update_checkpoint_param_names(checkpoint)
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
    top_order_attention_masks = get_attention_mask(top_order_barcodes, top_order_tokens, attn_rollout, device)
    draw_dna_barcode(top_order_barcodes, top_order_attention_masks, save_path)


if __name__ == "__main__":
    main()
