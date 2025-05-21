import copy
import os

import hydra
import numpy as np
import scipy.io as sio
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from tqdm import tqdm

from bioscanclip.util.dataset import load_insect_dataloader
from bioscanclip.util.util import get_features_and_label
from bioscanclip.model.simple_clip import load_clip_model


def main_process(rank: int, world_size: int, args):

    args.model_config.batch_size = 200

    # current_datetime = datetime.datetime.now()
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    args = copy.deepcopy(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Construct dataloader...")
    all_dataloader = load_insect_dataloader(
        args, world_size=world_size, rank=rank, load_all_in_one=True)

    print("Initialize model...")
    model = load_clip_model(args)

    # Using checkpoint that contrastive fine-tuned on INSECT dataset or not
    # checkpoint = torch.load(args.model_config.insect_ckpt_path, map_location='cuda:0')
    # use_ckpt_fine_tuned_on_INSECT_dataset = False
    # if hasattr(args.model_config, 'use_ckpt_fine_tuned_on_INSECT_dataset'):
    #     use_ckpt_fine_tuned_on_INSECT_dataset = args.model_config.use_ckpt_fine_tuned_on_INSECT_dataset
    # if use_ckpt_fine_tuned_on_INSECT_dataset:
    #     checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')
    # else:
    #     checkpoint = torch.load(args.model_config.pretrained_ckpt_path, map_location='cuda:0')

    # For now, just use model without fine-tuned on INSECT dataset
    use_ckpt_fine_tuned_on_INSECT_dataset = True
    if use_ckpt_fine_tuned_on_INSECT_dataset:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')


    if hasattr(args.model_config, 'load_ckpt') and args.model_config.load_ckpt is False:
        pass
        print("Do not load checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("Load checkpoint")
    model = model.to(device)

    if use_ckpt_fine_tuned_on_INSECT_dataset:
        folder = os.path.join(args.project_root_path, f"extracted_embedding/INSECT/{args.model_config.model_output_name}/finetuned_on_INSECT")
        os.makedirs(folder, exist_ok=True)
        dna_embed_path = os.path.join(folder, "dna_embedding_from_bioscan_clip.csv")
        image_embed_path = os.path.join(folder, "image_embedding_from_bioscan_clip.csv")
    else:
        folder = os.path.join(args.project_root_path, f"extracted_embedding/INSECT/{args.model_config.model_output_name}/trained_on_BIOSCAN_1M")
        os.makedirs(folder, exist_ok=True)
        dna_embed_path = os.path.join(folder, "dna_embedding_from_bioscan_clip_no_fine_tuned_on_INSECT.csv")
        image_embed_path = os.path.join(folder, "image_embedding_from_bioscan_clip_no_fine_tuned_on_INSECT.csv")

    att_splits_dict = sio.loadmat(args.insect_data.path_to_res_101_mat)
    labels = att_splits_dict["labels"].squeeze() - 1
    all_label = np.unique(labels)
    all_label.sort()
    dict_for_feature = get_features_and_label(all_dataloader, model, device, for_key_set=False)
    dna_feature = dict_for_feature["encoded_dna_feature"]


    dict_emb = {}

    pbar = tqdm(enumerate(labels), total=len(labels))
    for i, label in pbar:
        pbar.set_description("Extracting features: ")
        curr_feature = dna_feature[i]
        if str(label) not in dict_emb.keys():
            dict_emb[str(label)] = []
        dict_emb[str(label)].append(curr_feature)
    class_embed = []
    for i in all_label:
        class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
    class_embed = np.array(class_embed, dtype=object)
    class_embed = class_embed.T.squeeze()
    print(class_embed.shape)
    np.savetxt(dna_embed_path, class_embed,
               delimiter=",")

    # save_image_embedding to “image_embedding_from_bioscan_clip.csv”
    image_feature = dict_for_feature["encoded_image_feature"]
    image_feature = image_feature.astype(np.float32)
    image_feature = image_feature.T
    print(image_feature.shape)
    np.savetxt(image_embed_path, image_feature, delimiter=",")
    print(dna_embed_path)
    print(image_embed_path)


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()