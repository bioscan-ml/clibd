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

from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import (
    categorical_cmap,
    inference_and_print_result,
    get_features_and_label,
    make_prediction,
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

def plot_single_confusion_matrix(cm, classes, title, split, query, key, taxonomic_level):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    official_title = title + f" {split} {STRING_MAP_FOR_PLOT[query]} to {STRING_MAP_FOR_PLOT[key]} in {taxonomic_level} level"
    plt.title(official_title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # plt.xticks(rotation=45, ha="right")
    # plt.yticks(rotation=45, va="center")
    plt.tight_layout()

    folder_path = os.path.join("/local-scratch2/projects/bioscan-clip", f"confusion_matrix")
    os.makedirs(folder_path, exist_ok=True)
    official_title = official_title.replace(" ", "_")
    plt.savefig(os.path.join(folder_path, f"{official_title}.png"))


def plot_confusion_matrix_with_largest_10_classes(y_pred, y_true, class_labels, split, query, key, taxonomic_level):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    class_counts = np.diag(cm)
    most_common_indices = np.argsort(-class_counts)[0:10]
    most_common_labels = [class_labels[idx] for idx in most_common_indices]
    sub_cm_common = cm_normalized[np.ix_(most_common_indices, most_common_indices)]

    plot_single_confusion_matrix(sub_cm_common, most_common_labels, "Most Common Classes", split, query, key, taxonomic_level)


def plot_confusion_matrix_with_10_most_confused_classes(y_pred, y_true, class_labels, split, query, key,
                                                        taxonomic_level):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    cm_masked = np.copy(cm_normalized)
    np.fill_diagonal(cm_masked, -np.inf)

    largest_indices = np.dstack(np.unravel_index(np.argsort(-cm_masked.ravel()), cm_masked.shape))[0]

    unique_indices = set()
    for i, j in largest_indices:
        unique_indices.update([i, j])
        if len(unique_indices) >= 10:
            break

    unique_indices = sorted(list(unique_indices))[:10]

    sub_cm = cm_normalized[np.ix_(unique_indices, unique_indices)]
    sub_class_labels = [class_labels[idx] for idx in unique_indices]

    plot_single_confusion_matrix(sub_cm, sub_class_labels, f"Most Confused Classes at {taxonomic_level}", split, query,
                                 key, taxonomic_level)


def plot_confusion_matrix(pred_dict):
    for split in ['seen', 'unseen']:
        gt_list = pred_dict[f"{split}_gt_label"]
        for query, key in QUERY_AND_KEY_WE_CARE_ABOUT:
            pred_list = pred_dict[query][key][f"curr_{split}_pred_list"]
            for taxonomic_level in ['order', 'family', 'genus', 'species']:
                curr_taxonomic_level_pred_list_top_1 = [pred[taxonomic_level][0] for pred in pred_list]
                curr_taxonomic_level_gt_list = [gt[taxonomic_level] for gt in gt_list]
                unique_classes = list(set(curr_taxonomic_level_gt_list))

                plot_confusion_matrix_with_largest_10_classes(
                    curr_taxonomic_level_pred_list_top_1, curr_taxonomic_level_gt_list, unique_classes, split, query, key, taxonomic_level)
                plot_confusion_matrix_with_10_most_confused_classes(
                    curr_taxonomic_level_pred_list_top_1, curr_taxonomic_level_gt_list, unique_classes, split, query, key, taxonomic_level)


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
                        except (ValueError, TypeError, OSError) as e:
                            print(f"Error creating dataset for {embedding_type}: {e}")
                            # continue to next embedding_type
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
    except NameError:
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
    plot_confusion_matrix(pred_dict)


if __name__ == "__main__":
    main()
