import copy
import json
import os

import hydra
import numpy as np
import scipy.io as sio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from tqdm import tqdm

from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.util import get_features_and_label, inference_and_print_result
from bioscanclip.util.dataset import load_insect_dataloader


def save_prediction(pred_list, gt_list, json_path):
    data = {
        "gt_labels": gt_list,
        "pred_labels": pred_list
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)


def ddp_setup(rank: int, world_size: int, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def construct_key_dict(list_of_dict):
    key_dict = {}

    for curr_dict in list_of_dict:
        for a_kind_of_feature_or_label in curr_dict.keys():
            if a_kind_of_feature_or_label == "all_key_features" or a_kind_of_feature_or_label == "all_key_features_label":
                key_dict[a_kind_of_feature_or_label] = None
                continue

            if a_kind_of_feature_or_label not in key_dict.keys():
                key_dict[a_kind_of_feature_or_label] = curr_dict[a_kind_of_feature_or_label]
            else:
                if isinstance(curr_dict[a_kind_of_feature_or_label], list):
                    key_dict[a_kind_of_feature_or_label] = key_dict[a_kind_of_feature_or_label] + curr_dict[
                        a_kind_of_feature_or_label]
                else:
                    key_dict[a_kind_of_feature_or_label] = np.concatenate(
                        (key_dict[a_kind_of_feature_or_label], curr_dict[a_kind_of_feature_or_label]), axis=0)

    return key_dict


def eval_phase(model, device, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader,
               insect_test_unseen_dataloader, k_list,
               species_to_drop=None):
    insect_train_dict = get_features_and_label(
        insect_train_dataloader_for_key, model, device)
    insect_val_dict = get_features_and_label(
        insect_val_dataloader, model, device)
    insect_test_seen_dict = get_features_and_label(
        insect_test_seen_dataloader, model, device)
    insect_test_unseen_dict = get_features_and_label(
        insect_test_unseen_dataloader, model, device)

    keys_dict = construct_key_dict([insect_train_dict, insect_val_dict, insect_test_seen_dict, insect_test_unseen_dict])

    acc_dict, _, pred_dict = inference_and_print_result(keys_dict, insect_test_seen_dict, insect_test_unseen_dict,
                                                        small_species_list=None, k_list=k_list)

    return acc_dict, pred_dict


def convert_acc_dict_to_wandb_dict(acc_dict):
    dict_for_wandb = {}
    acc_dict = acc_dict['encoded_image_feature']['encoded_image_feature']

    # For now, we just put query: image and key:image acc to wandb

    for split, split_dict in acc_dict.items():
        for type_of_acc, type_of_acc_dict in split_dict.items():
            for k, k_dict in type_of_acc_dict.items():
                for level, curr_acc in split_dict.items():
                    dict_for_wandb[f"{split} {type_of_acc} top-{k} {level} level"] = curr_acc

    return dict_for_wandb


def main_process(rank: int, world_size: int, args):
    # # Set up for debug, delete when you see it!

    args.debug_flag = False
    # # Set up for debug, delete when you see it!

    # special set up for train on INSECT dataset
    args.model_config.batch_size = 200
    args.model_config.epochs = 80

    if args.debug_flag:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

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
    checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')

    model.load_state_dict(checkpoint)
    model = model.to(device)
    os.makedirs("embeddings_no_fine_tune_on_INSECT")
    # save_dna_embedding to “dna_embedding_from_bioscan_clip.csv”
    dna_embed_path = "embeddings_no_fine_tune_on_INSECT/dna_embedding_from_bioscan_clip.csv"


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
    np.savetxt( dna_embed_path, class_embed,
               delimiter=",")



    # save_image_embedding to “image_embedding_from_bioscan_clip.csv”
    image_embed_path = "embeddings_no_fine_tune_on_INSECT/image_embedding_from_bioscan_clip.csv"
    image_feature = dict_for_feature["encoded_image_feature"]
    image_feature = image_feature.astype(np.float32)
    image_feature = image_feature.T
    print(image_feature.shape)
    np.savetxt(image_embed_path, image_feature, delimiter=",")









@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
