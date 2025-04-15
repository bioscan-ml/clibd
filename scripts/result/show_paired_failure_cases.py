import io
import json
import os
import random
from collections import Counter, defaultdict

import h5py
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import torch
from PIL import Image
from omegaconf import DictConfig
from sklearn.metrics import silhouette_samples, confusion_matrix
from umap import UMAP
from tqdm import tqdm
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import (
    inference_and_print_result,
    get_features_and_label,
    find_closest_match,
    create_id_index_map,
    load_image_from_hdf5_with_id_as_input,
    All_TYPE_OF_FEATURES_OF_KEY,
)
import seaborn as sns

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"

QUERY_AND_KEY_WE_CARE_ABOUT = [
    ("encoded_image_feature", "encoded_image_feature"),
    ("encoded_dna_feature", "encoded_dna_feature"),
    ("encoded_image_feature", "encoded_dna_feature"),
]

STRING_MAP_FOR_PLOT = {"encoded_image_feature": "image", "encoded_dna_feature": "DNA", "encoded_language_feature": "text"}

def save_paired_failure_case_for_curr_dict():
    pass

def save_paired_failure_cases(
    args,
    keys_dict,
    seen_dict,
    unseen_dict,
    number_of_failure_cases=10,
):


    seen_gt_label_list = seen_dict["label_list"]
    unseen_gt_label_list = unseen_dict["label_list"]
    keys_label = keys_dict["label_list"]
    # setup directory
    folder_path = os.path.join(
        args.project_root_path,
        "failure_cases"
    )
    print("Creating id map...")
    id_to_split_and_position = create_id_index_map(args)
    print("Id map created.")
    os.makedirs(folder_path, exist_ok=True)
    for query_feature_type, key_feature_type in QUERY_AND_KEY_WE_CARE_ABOUT:
        failure_cases = []
        curr_seen_feature = seen_dict[query_feature_type]
        curr_unseen_feature = unseen_dict[query_feature_type]
        curr_keys_feature = keys_dict[key_feature_type]
        seen_output_dict = find_closest_match(
            curr_seen_feature, curr_keys_feature, keys_label, with_similarity=False, max_k=1
        )
        unseen_output_dict = find_closest_match(
            curr_unseen_feature, curr_keys_feature, keys_label, max_k=1
        )

        seen_pred_list = seen_output_dict["pred_list"]
        unseen_pred_list = unseen_output_dict["pred_list"]

        seen_pred_index_list = seen_output_dict["indices"]
        unseen_pred_index_list = unseen_output_dict["indices"]
        os.makedirs(os.path.join(folder_path, 'seen', f"{query_feature_type}_to_{key_feature_type}"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'unseen', f"{query_feature_type}_to_{key_feature_type}"), exist_ok=True)
        for idx in tqdm(range(len(seen_gt_label_list))):
            seen_gt_label, seen_pred_label = seen_gt_label_list[idx], seen_pred_list[idx]
            seen_gt_species = seen_gt_label['species']
            seen_pred_species = seen_pred_label['species'][0]

            if seen_gt_species != seen_pred_species:
                # load the images of the gt and pred
                gt_image_id = seen_dict["processed_id_list"][idx]
                pred_image_id = keys_dict["processed_id_list"][seen_pred_index_list[idx][0]]

                gt_image = load_image_from_hdf5_with_id_as_input(args, gt_image_id, id_to_split_and_position)
                pred_image = load_image_from_hdf5_with_id_as_input(args, pred_image_id, id_to_split_and_position)

                # show the images
                # Only show image, without axis
                fig, ax = plt.subplots(1, 2)
                plt.axis('off')
                ax[0].imshow(gt_image)
                ax[0].set_title(f"GT: {seen_gt_species}")
                plt.axis('off')
                ax[1].imshow(pred_image)
                ax[1].set_title(f"Pred: {seen_pred_species}")
                fig.suptitle(f"Query: {STRING_MAP_FOR_PLOT[query_feature_type]} Key: {STRING_MAP_FOR_PLOT[key_feature_type]}")
                plt.axis('off')
                plt.tight_layout()

                plt.savefig(os.path.join(folder_path, 'seen', f"{query_feature_type}_to_{key_feature_type}", f"{gt_image_id.split('.')[0]}_{pred_image_id.split('.')[0]}.png"))

        for idx in tqdm(range(len(unseen_gt_label_list))):
            unseen_gt_label, unseen_pred_label = unseen_gt_label_list[idx], unseen_pred_list[idx]
            unseen_gt_species = unseen_gt_label['species']
            unseen_pred_species = unseen_pred_label['species']

            if unseen_gt_species != unseen_pred_species:
                # load the images of the gt and pred
                gt_image_id = unseen_dict["processed_id_list"][idx]
                pred_image_id = keys_dict["processed_id_list"][unseen_pred_index_list[idx][0]]

                gt_image = load_image_from_hdf5_with_id_as_input(args, gt_image_id, id_to_split_and_position)
                pred_image = load_image_from_hdf5_with_id_as_input(args, pred_image_id, id_to_split_and_position)

                # show the images
                # Only show image, without axis
                fig, ax = plt.subplots(1, 2)
                plt.axis('off')
                ax[0].imshow(gt_image)
                ax[0].set_title(f"GT: {unseen_gt_species}")
                plt.axis('off')
                ax[1].imshow(pred_image)
                ax[1].set_title(f"Pred: {unseen_pred_species}")
                plt.axis('off')
                fig.suptitle(f"Query: {STRING_MAP_FOR_PLOT[query_feature_type]} Key: {STRING_MAP_FOR_PLOT[key_feature_type]}")

                plt.savefig(os.path.join(folder_path, 'unseen', f"{query_feature_type}_to_{key_feature_type}", f"{gt_image_id.split('.')[0]}_{pred_image_id.split('.')[0]}.png"))

@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")
    folder_for_saving = os.path.join(
        args.project_root_path, "extracted_embedding", args.model_config.dataset, args.model_config.model_output_name
    )
    os.makedirs(folder_for_saving, exist_ok=True)
    labels_path = os.path.join(folder_for_saving, f"labels_{args.inference_and_eval_setting.eval_on}.json")
    processed_id_path = os.path.join(folder_for_saving, f"processed_id_{args.inference_and_eval_setting.eval_on}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted_features_path = os.path.join(
        folder_for_saving, f"extracted_feature_from_{args.inference_and_eval_setting.eval_on}_split.hdf5"
    )

    if os.path.exists(extracted_features_path) and os.path.exists(labels_path) and args.load_inference:
        print("Loading embeddings from file...")

        with h5py.File(extracted_features_path, "r") as hdf5_file:
            seen_dict = {}
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["seen"].keys():
                    seen_dict[type_of_feature] = hdf5_file["seen"][type_of_feature][:]

            unseen_dict = {}
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["unseen"].keys():
                    unseen_dict[type_of_feature] = hdf5_file["unseen"][type_of_feature][:]
            keys_dict = {}
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["key"].keys():
                    keys_dict[type_of_feature] = hdf5_file["key"][type_of_feature][:]

        with open(labels_path, "r") as json_file:
            total_dict = json.load(json_file)
        seen_dict["label_list"] = total_dict["seen_gt_dict"]
        unseen_dict["label_list"] = total_dict["unseen_gt_dict"]
        keys_dict["label_list"] = total_dict["key_gt_dict"]
        keys_dict["all_key_features_label"] = (
                total_dict["key_gt_dict"] + total_dict["key_gt_dict"] + total_dict["key_gt_dict"]
        )

        with open(processed_id_path, "r") as json_file:
            id_dict = json.load(json_file)
        seen_dict["processed_id_list"] = id_dict["seen_id_list"]
        unseen_dict["processed_id_list"] = id_dict["unseen_id_list"]
        keys_dict["processed_id_list"] = id_dict["key_id_list"]
        keys_dict["all_processed_id_list"] = id_dict["key_id_list"] + id_dict["key_id_list"] + id_dict["key_id_list"]

    else:
        # initialize model
        print("Initialize model...")

        model = load_clip_model(args, device)

        if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
            pass
        else:
            checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
            model.load_state_dict(checkpoint)

        # Load data
        # args.model_config.batch_size = 24

        if args.inference_and_eval_setting.eval_on == "val":
            (
                _,
                seen_dataloader,
                unseen_dataloader,
                _,
                _,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        elif args.inference_and_eval_setting.eval_on == "test":
            (
                _,
                _,
                _,
                seen_dataloader,
                unseen_dataloader,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        else:
            raise ValueError(
                "Invalid value for eval_on, specify by 'python inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl_ver_0_1_2.yaml' inference_and_eval_setting.eval_on=test/val'"
            )
        for_open_clip = False

        if hasattr(args.model_config, "for_open_clip"):
            for_open_clip = args.model_config.for_open_clip

        keys_dict = get_features_and_label(
            all_keys_dataloader, model, device, for_key_set=True, for_open_clip=for_open_clip
        )

        seen_dict = get_features_and_label(seen_dataloader, model, device, for_open_clip=for_open_clip)

        unseen_dict = get_features_and_label(unseen_dataloader, model, device, for_open_clip=for_open_clip)

        if args.save_inference and not (os.path.exists(extracted_features_path) and os.path.exists(labels_path)):
            new_file = h5py.File(extracted_features_path, "w")
            name_of_splits = ["seen", "unseen", "key"]
            split_dicts = [seen_dict, unseen_dict, keys_dict]
            for split_name, split in zip(name_of_splits, split_dicts):
                group = new_file.create_group(split_name)
                for embedding_type in All_TYPE_OF_FEATURES_OF_KEY:
                    if embedding_type in split.keys():
                        try:
                            group.create_dataset(embedding_type, data=split[embedding_type])
                            print(f"Created dataset for {embedding_type}")
                        except:
                            print(f"Error in creating dataset for {embedding_type}")
                        # group.create_dataset(embedding_type, data=split[embedding_type])
            new_file.close()
            total_dict = {
                "seen_gt_dict": seen_dict["label_list"],
                "unseen_gt_dict": unseen_dict["label_list"],
                "key_gt_dict": keys_dict["label_list"],
            }
            with open(labels_path, "w") as json_file:
                json.dump(total_dict, json_file, indent=4)

            id_dict = {
                "seen_id_list": seen_dict["file_name_list"],
                "unseen_id_list": unseen_dict["file_name_list"],
                "key_id_list": keys_dict["file_name_list"],
            }
            with open(processed_id_path, "w") as json_file:
                json.dump(id_dict, json_file, indent=4)

    acc_dict, per_class_acc, pred_dict = inference_and_print_result(
        keys_dict,
        seen_dict,
        unseen_dict,
        args,
        small_species_list=None,
        k_list=args.inference_and_eval_setting.k_list,
    )

    per_claSS_acc_path = os.path.join(
        folder_for_saving, f"per_class_acc_{args.inference_and_eval_setting.eval_on}.json"
    )
    with open(per_claSS_acc_path, "w") as json_file:
        json.dump(per_class_acc, json_file, indent=4)

    acc_dict_path = os.path.join(folder_for_saving, f"acc_dict_{args.inference_and_eval_setting.eval_on}.json")
    with open(acc_dict_path, "w") as json_file:
        json.dump(acc_dict, json_file, indent=4)

    try:
        seen_keys_dataloader
        val_unseen_keys_dataloader
        test_unseen_keys_dataloader
    except:
        if args.inference_and_eval_setting.eval_on == "val":
            (
                _,
                seen_dataloader,
                unseen_dataloader,
                _,
                _,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        elif args.inference_and_eval_setting.eval_on == "test":
            (
                _,
                _,
                _,
                seen_dataloader,
                unseen_dataloader,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        else:
            raise ValueError(
                "Invalid value for eval_on, specify by 'python inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl_ver_0_1_2.yaml' inference_and_eval_setting.eval_on=test/val'"
            )

    save_paired_failure_cases(
        args,
        keys_dict,
        seen_dict,
        unseen_dict,
    )


if __name__ == "__main__":
    main()
