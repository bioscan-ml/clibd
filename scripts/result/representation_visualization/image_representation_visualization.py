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


# Reference and modified fromhttps://github.com/jacobgil/vit-explain
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
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="min",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.hooks = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
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


def encode_all_image(image_list, model, transform, device):
    for image in image_list:
        image = transform(image).unsqueeze(0).to(device)
        image_output = F.normalize(model(image), p=2, dim=-1)


def get_some_images_from_hdf5(hdf5_group, n=None):
    image_list = []
    if n is None: n = len(hdf5_group["image"])
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


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_and_save_vit_explaination(image_list, attn_rollout, transform, device, layer_idx=None,
                                  folder_name="representation_visualization/before_contrastive_learning"):
    os.makedirs(folder_name, exist_ok=True)
    for idx, image in tqdm(enumerate(image_list), total=len(image_list)):
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(device)
            mask = attn_rollout(image_tensor, layer_idx=layer_idx)

        np_img = np.array(image)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        image_with_mask = show_mask_on_image(np_img, mask)

        # print(type(np_img), type(mask), type(image_with_mask))

        cv2.imwrite(f"{folder_name}/vit_explaination_image_with_mask_{idx}.png", image_with_mask)
    attn_rollout.remove_hooks()
    # torch.cuda.empty_cache()
    # gc.collect()

@hydra.main(config_path="../../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    save_path = os.path.join(args.project_root_path, "image_representation_visualization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load some images from the hdf5 file
    if args.model_config.dataset == "bioscan_5m":
        path_to_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    else:
        path_to_hdf5 = args.bioscan_data.path_to_hdf5_data

    # Init transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Open the hdf5 file
    with  h5py.File(path_to_hdf5, "r", libver="latest") as hdf5_file:
        # For now just use train_seen data
        hdf5_group = hdf5_file["train_seen"]
        # Load some images from the hdf5 file
        image_list = get_some_images_from_hdf5(hdf5_group, n=100)

    # Save these images into a folder call "representation_visualization/original_images"
    original_imgage_folder = os.path.join(save_path, "original_images")
    if not os.path.exists(original_imgage_folder):
        os.makedirs(os.path.join(save_path, "original_images"), exist_ok=True)
        for idx, image in enumerate(image_list):
            image.save(os.path.join(save_path, f"original_images/image_{idx}.png"))

    print("Initialize model...")
    model = load_clip_model(args, device)
    model.eval()
    # Get the image encoder
    image_encoder = get_image_encoder(model, device)
    for block in image_encoder.base_image_encoder.blocks:
        block.attn.fused_attn = False

    print("Before contrastive learning")
    attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
    get_and_save_vit_explaination(image_list, attn_rollout, transform, device,
                                  folder_name=os.path.join(save_path, "before_contrastive_learning"))



    print("After contrastive learning")
    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)

    model.eval()
    # Get the image encoder
    image_encoder = get_image_encoder(model, device)
    for block in image_encoder.base_image_encoder.blocks:
        block.attn.fused_attn = False

    attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
    get_and_save_vit_explaination(image_list, attn_rollout, transform, device,
                                  folder_name=os.path.join(save_path, "after_contrastive_learning"))

    for layer_idx in range(12):
        print(f"Layer {layer_idx}")

        attn_rollout = VITAttentionRollout(image_encoder, discard_ratio=0.9, head_fusion="max")
        get_and_save_vit_explaination(image_list, attn_rollout, transform, device, layer_idx=layer_idx,
                                      folder_name=os.path.join(save_path, f"single_layer/{layer_idx}"))

    print(f"Done, images saved to {save_path}")


if __name__ == "__main__":
    main()