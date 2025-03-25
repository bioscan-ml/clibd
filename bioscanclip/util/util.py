import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import faiss
import random
from torch import nn
from tqdm import tqdm
from bioscanclip.epoch.inference_epoch import get_feature_and_label
import copy
from loratorch.layers import MultiheadAttention as LoRA_MultiheadAttention
from sklearn.preprocessing import normalize
import os
import csv
from omegaconf import DictConfig, OmegaConf
import h5py
from PIL import Image
import io
from itertools import product
from torchtext.vocab import vocab as build_vocab_from_dict

LEVELS = ["order", "family", "genus", "species"]
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

class EncoderWithExtraLayer(nn.Module):
    def __init__(self, encoder, new_linear_layer):
        super(EncoderWithExtraLayer, self).__init__()
        self.encoder = encoder
        self.new_linear_layer = new_linear_layer

    def get_feature(self, x):
        return self.encoder(x)

    def forward(self, x):
        outputs = self.encoder(x)
        outputs = self.new_linear_layer(outputs)
        return outputs

class Table:
    def __init__(self, headers, data):
        self.headers = headers
        self.data = data
        self.column_widths = [max(len(str(item)) for item in column) for column in zip(headers, *data)]

    def print_table(self):
        self.print_row(self.headers)
        self.print_separator()
        for row in self.data:
            self.print_row(row)

    def print_row(self, row):
        formatted_row = "|".join(f"{str(item):^{width}}" for item, width in zip(row, self.column_widths))
        print(f"|{formatted_row}|")

    def print_separator(self):
        separator = "+".join("-" * (width + 2) for width in self.column_widths)
        print(f"+{separator}+")


class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))


class KmerTokenizer(object):
    def __init__(self, k, vocabulary_mapper, stride=1, padding=False, max_len=660):
        self.k = k
        self.stride = stride
        self.padding = padding
        self.max_len = max_len
        self.vocabulary_mapper = vocabulary_mapper

    def __call__(self, dna_sequence, offset=0):
        tokens = []
        att_mask = [1] * (self.max_len // self.stride)
        x = dna_sequence[offset:]
        if self.padding:
            if len(x) > self.max_len:
                x = x[: self.max_len]
            else:
                att_mask[len(x) // self.stride :] = [0] * (len(att_mask) - len(x) // self.stride)
                x = x + "N" * (self.max_len - len(x))
        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i : i + self.k]
            tokens.append(k_mer)

        tokens = torch.tensor(self.vocabulary_mapper(tokens), dtype=torch.int64)
        att_mask = torch.tensor(att_mask, dtype=torch.int32)

        return tokens, att_mask


def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def load_bert_model(bert_model, path_to_ckpt):
    state_dict = torch.load(path_to_ckpt, map_location=torch.device("cpu"))
    state_dict = remove_extra_pre_fix(state_dict)
    bert_model.load_state_dict(state_dict)


def print_result(
    args,
    seen_micro_acc_dict,
    seen_macro_acc_dict,
    unseen_micro_acc_dict,
    unseen_macro_acc_dict,
    k_list,
    dict_for_wandb=None,
):
    if dict_for_wandb is None:
        dict_for_wandb = {}
    overall_acc = []
    for k in k_list:
        print(f"When k is {k}:")
        print("\tFor seen split: ")
        for level in list(seen_micro_acc_dict[k_list[0]].keys()):
            print(f"\t\tFor {level} level:")
            print(f"\t\t\tMicro acc:\t {seen_micro_acc_dict[k][level]}")
            print(f"\t\t\tMacro acc:\t {seen_macro_acc_dict[k][level]}")
            overall_acc.append(seen_micro_acc_dict[k][level])
            overall_acc.append(seen_macro_acc_dict[k][level])
            if dict_for_wandb is not None:
                dict_for_wandb[f"micro acc of seen split in {level} level with k={k}"] = seen_micro_acc_dict[k][level]
                dict_for_wandb[f"macro acc of seen split in {level} level with k={k}"] = seen_macro_acc_dict[k][level]
        print("\tFor unseen split: ")
        for level in list(unseen_micro_acc_dict[k_list[0]].keys()):
            print(f"\t\tFor {level} level:")
            print(f"\t\t\tMicro acc:\t {unseen_micro_acc_dict[k][level]}")
            print(f"\t\t\tMacro acc:\t {unseen_macro_acc_dict[k][level]}")
            overall_acc.append(unseen_micro_acc_dict[k][level])
            overall_acc.append(unseen_macro_acc_dict[k][level])
            if dict_for_wandb is not None:
                dict_for_wandb[f"macro acc of seen split in {level} level with k={k}"] = unseen_micro_acc_dict[k][level]
                dict_for_wandb[f"macro acc of unseen split in {level} level with k={k}"] = unseen_macro_acc_dict[k][
                    level
                ]
    overall_acc = sum(overall_acc) * 1.0 / len(overall_acc)
    print("Naive overall acc: " + str(overall_acc))

    print("For copy to google sheet")

    for k in k_list:
        curr_row = ""
        for level in list(seen_micro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{seen_micro_acc_dict[k][level]:.4f}\t"
        for level in list(unseen_micro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{unseen_micro_acc_dict[k][level]:.4f}\t"
        print(curr_row)
    for k in k_list:
        curr_row = ""
        for level in list(seen_macro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{seen_macro_acc_dict[k][level]:.4f}\t"
        for level in list(unseen_macro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{unseen_macro_acc_dict[k][level]:.4f}\t"
        print(curr_row)

    return overall_acc, dict_for_wandb


def tokenize_dna_sequence(curr_dna_input, sequence_pipeline):
    curr_dna_input = tokenize_dna_sequence(sequence_pipeline, np.array(curr_dna_input))
    return curr_dna_input


def load_small_species(args):
    small_species_list = None
    if hasattr(args.bioscan_data, "path_to_small_species_list_json"):
        with open(args.bioscan_data.path_to_small_species_list_json, "r") as json_file:
            small_species_list = json.load(json_file)
    return small_species_list


def find_k_closest_records(
    input_file_name_list, input_feature_np_array, keys_file_name_list, keys_feature_np_array, k=5
):
    result_dict = {}
    index = faiss.IndexFlatIP(keys_feature_np_array.shape[-1])
    index.add(keys_feature_np_array)
    distances, indices = index.search(input_feature_np_array, k)
    for record_idx, input_file_name in enumerate(input_file_name_list):
        k_closest_records_file_name = [keys_file_name_list[key_idx] for key_idx in indices[record_idx]]
        result_dict[input_file_name_list[record_idx]] = k_closest_records_file_name
    return result_dict


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Create a colormap with a certain number of shades of colours.

    Based on https://stackoverflow.com/a/47232942/1960959

    Parameters
    ----------
    nc : int
        Number of categories.
    nsc : int
        Number of shades per category.
    cmap : str, default=tab10
        Original colormap to extend into multiple shades.
    continuous : bool, default=False
        Whether ``cmap`` is continous. Otherwise it is treated
        as categorical with adjacent colors unrelated.

    Returns
    -------
    matplotlib.colors.ListedColormap
        New cmap which alternates between ``nsc`` shades of ``nc``
        colors from ``cmap``.
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc : (i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

def load_all_seen_species_name_and_create_label_map(train_seen_dataloader):
    all_seen_species = []
    species_to_other_labels = {}

    for batch in train_seen_dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_seen_species = all_seen_species + label_batch['species']
        for curr_idx in range(len(label_batch['species'])):
            if label_batch['species'][curr_idx] not in species_to_other_labels.keys():
                species_to_other_labels[label_batch['species'][curr_idx]] = {'order': label_batch['order'][curr_idx],
                                                                             'family': label_batch['family'][curr_idx],
                                                                             'genus': label_batch['genus'][curr_idx]}

    all_seen_species = list(set(all_seen_species))
    all_seen_species.sort()

    label_to_index_dict = {}
    idx_to_all_labels = {}

    for idx, species_label in enumerate(all_seen_species):
        label_to_index_dict[species_label] = idx
        idx_to_all_labels[idx] = {'species': species_label, 'order': species_to_other_labels[species_label]['order'],
                                  'family': species_to_other_labels[species_label]['family'],
                                  'genus': species_to_other_labels[species_label]['genus']}

    return label_to_index_dict, idx_to_all_labels

def get_unique_species_for_seen(dataloader):
    all_species = []
    pbar = tqdm(dataloader)
    for batch in pbar:
        pbar.set_description("Getting unique species labels")
        _, _, _, _, _, _, label_batch = batch
        all_species = all_species + label_batch['species']

    unique_species = list(set(all_species))
    return unique_species

def add_lora_layer_to_open_clip(open_clip_model, r: int = 4, num_classes: int = 0, lora_layer=None):
    if num_classes != 768:
        raise ValueError(
            "num_classes should be 768 for OpenCLIP, may need to implement a new head for other num_classes")

    vit_model = open_clip_model.visual

    for param in vit_model.parameters():
        param.requires_grad = False

    assert r > 0
    if lora_layer is not None:
        lora_layer = lora_layer
    else:
        lora_layer = list(range(len(vit_model.transformer.resblocks)))
        block_list = enumerate(vit_model.transformer.resblocks)

    for param in vit_model.parameters():
        param.requires_grad = False

    for t_layer_i, blk in block_list:
        # If we only want few lora layer instead of all
        if t_layer_i not in lora_layer:
            continue
        blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim,
                                            num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)
    open_clip_model.visual = vit_model

    # Do the same for the language model
    language_model = open_clip_model.transformer
    for param in language_model.parameters():
        param.requires_grad = False

    assert r > 0
    if lora_layer is not None:
        lora_layer = lora_layer
    else:
        lora_layer = list(range(len(language_model.resblocks)))
        block_list = enumerate(language_model.resblocks)

    for param in language_model.parameters():
        param.requires_grad = False

    for t_layer_i, blk in block_list:
        # If we only want few lora layer instead of all
        if t_layer_i not in lora_layer:
            continue
        blk.attn = create_child_from_parent(blk.attn, LoRA_MultiheadAttention, embed_dim=blk.attn.embed_dim,
                                            num_heads=blk.attn.num_heads, enable_lora=['q', 'k', 'v'], r=r)
    open_clip_model.transformer = language_model

    return open_clip_model

def create_child_from_parent(parent_instance, child_class, **child_args):
    child_instance = child_class.__new__(child_class)
    for attr, value in vars(parent_instance).items():
        if hasattr(child_instance, attr):
            setattr(child_instance, attr, copy.deepcopy(value))
    child_class.__init__(child_instance, **child_args)
    return child_instance


# Below are help functions for inference and eval
def top_k_micro_accuracy(pred_list, gt_list, k_list=None):
    total_samples = len(pred_list)
    k_micro_acc = {}
    for k in k_list:
        if k not in k_micro_acc.keys():
            k_micro_acc[k] = {}
        for level in LEVELS:
            correct_in_curr_level = 0
            for pred_dict, gt_dict in zip(pred_list, gt_list):

                pred_labels = pred_dict[level][:k]
                gt_label = gt_dict[level]
                if gt_label in pred_labels:
                    correct_in_curr_level += 1
            k_micro_acc[k][level] = correct_in_curr_level * 1.0 / total_samples

    return k_micro_acc

def print_micro_and_macro_acc(acc_dict, k_list, args):
    header = [
        " ",
        "Seen Order",
        "Seen Family",
        "Seen Genus",
        "Seen Species",
        "Unseen Order",
        "Unseen Family",
        "Unseen Genus",
        "Unseen Species",
    ]

    # read modalities from config
    # TODO: fit complicated strategey after updateing the config
    model_config = args.model_config
    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        alignment = "None"
    else:
        alignment = "I"
        if hasattr(model_config, "dna"):
            alignment += ",D"
        if hasattr(model_config, "language"):
            alignment += ",T"

    suffix = f"({alignment})"

    rows = []
    csv_data_dict = {'encoded_image_feature': 'Image',
                     'encoded_dna_feature': 'DNA',
                     'encoded_language_feature': 'Text',
                     'averaged_feature': 'Ave' + suffix,
                     'concatenated_feature': 'Concat' + suffix,
                     'all_key_features': 'All' + suffix}
    csv_data = [["learning_strategy", "Alignment", "DNA_encoder", "Image_encoder", "Language_encoder", "Epoch",
                 "Latent_space_dim", "Query", "Key", "Metric", "Seen_Order", "Seen_Family", "Seen_Genus",
                 "Seen_Species", "Unseen_Order", "Unseen_Family", "Unseen_Genus", "Unseen_Species"]]

    def read_encoder(model_config, key):
        if hasattr(model_config, key):
            if hasattr(model_config[key], "model"):
                return model_config[key].model
            elif hasattr(model_config[key], "pre_train_model"):
                return model_config[key].pre_train_model
        else:
            return "None"

    row_for_csv_data = ['LoRA', alignment]
    row_for_csv_data.append(read_encoder(model_config, "dna"))
    row_for_csv_data.append(read_encoder(model_config, "image"))
    row_for_csv_data.append(read_encoder(model_config, "language"))
    row_for_csv_data.append(model_config.epochs)
    row_for_csv_data.append(model_config.output_dim)

    rows_for_copy_to_google_doc = []
    for query_feature_type in All_TYPE_OF_FEATURES_OF_QUERY:
        if query_feature_type not in acc_dict.keys():
            continue
        for key_feature_type in All_TYPE_OF_FEATURES_OF_KEY:
            if key_feature_type not in acc_dict[query_feature_type].keys():
                continue
            for type_of_acc in ["micro_acc", "macro_acc"]:
                for k in k_list:
                    if len(list(acc_dict[query_feature_type][key_feature_type].keys())) == 0:
                        continue
                    curr_row = [
                        f"Query_feature: {query_feature_type}||Key_feature: {key_feature_type}||{type_of_acc} top-{k}"
                    ]
                    row_for_copy_to_google_doc = ""

                    row_for_csv = row_for_csv_data.copy()
                    row_for_csv += \
                        [csv_data_dict[query_feature_type], csv_data_dict[key_feature_type],
                         type_of_acc.replace('m', 'M').replace("_", f"_Top-{k}_")]

                    for spit in ["seen", "unseen"]:
                        for level in LEVELS:
                            num = round(acc_dict[query_feature_type][key_feature_type][spit][type_of_acc][k][level], 4)

                            curr_row.append(f"\t{num}")
                            row_for_copy_to_google_doc = (
                                    row_for_copy_to_google_doc + f"{num}\t"
                            )
                            row_for_csv.append(num)
                    rows.append(curr_row)
                    rows_for_copy_to_google_doc.append(row_for_copy_to_google_doc)
                    csv_data.append(row_for_csv)
    table = Table(header, rows)
    table.print_table()

    print("For copy to google doc")
    for row in rows_for_copy_to_google_doc:
        print(row)

    if args.save_inference:
        logs_folder = os.path.join("logs")
        os.makedirs(logs_folder, exist_ok=True)

        # write accurate to json
        with open(os.path.join(logs_folder, "accuracy.json"), 'w') as fp:
            json.dump(acc_dict, fp)
        print(f"Accuracy dict saved to logs folder: {logs_folder}/accuracy.json")

        # write results to csv
        with open(os.path.join(logs_folder, "results.csv"), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerows(csv_data)
        print(f"CSV results saved to logs folder: {logs_folder}/results.csv")

        raw_csv_data = []
        for row in csv_data[1:]:
            raw_csv_data.append(row[-8:])

        with open(os.path.join(logs_folder, "raw.csv"), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerows(raw_csv_data)
        print(f"raw results saved to logs folder: {logs_folder}/raw.csv")

        # write config to json
        OmegaConf.save(args, os.path.join(logs_folder, 'config.yaml'))
        print(f"Config saved to logs folder: {logs_folder}/config.json")

def make_prediction(query_feature, keys_feature, keys_label, with_similarity=False, with_indices=False, max_k=5):
    index = faiss.IndexFlatIP(keys_feature.shape[-1])
    keys_feature = normalize(keys_feature, norm="l2", axis=1).astype(np.float32)
    query_feature = normalize(query_feature, norm="l2", axis=1).astype(np.float32)
    index.add(keys_feature)
    pred_list = []

    similarities, indices = index.search(query_feature, max_k)
    for key_indices in indices:
        k_pred_in_diff_level = {}
        for level in LEVELS:
            if level not in k_pred_in_diff_level.keys():
                k_pred_in_diff_level[level] = []
            for i in key_indices:
                try:
                    k_pred_in_diff_level[level].append(keys_label[i][level])
                except:
                    print(keys_label)
                    exit()
        pred_list.append(k_pred_in_diff_level)

    out = [pred_list]

    if with_similarity:
        out.append(similarities)

    if with_indices:
        out.append(indices)

    if len(out) == 1:
        return out[0]
    return out

def top_k_macro_accuracy(pred_list, gt_list, k_list=None):
    if k_list is None:
        k_list = [1, 3, 5]

    macro_acc_dict = {}
    per_class_acc = {}
    pred_counts = {}
    gt_counts = {}

    for k in k_list:
        macro_acc_dict[k] = {}
        per_class_acc[k] = {}
        pred_counts[k] = {}
        gt_counts[k] = {}
        for level in LEVELS:
            pred_counts[k][level] = {}
            gt_counts[k][level] = {}
            for pred, gt in zip(pred_list, gt_list):

                pred_labels = pred[level][:k]
                gt_label = gt[level]
                if gt_label not in pred_counts[k][level].keys():
                    pred_counts[k][level][gt_label] = 0
                if gt_label not in gt_counts[k][level].keys():
                    gt_counts[k][level][gt_label] = 0

                if gt_label in pred_labels:
                    pred_counts[k][level][gt_label] = pred_counts[k][level][gt_label] + 1
                gt_counts[k][level][gt_label] = gt_counts[k][level][gt_label] + 1

    for k in k_list:
        for level in LEVELS:
            sum_in_this_level = 0
            list_of_labels = list(gt_counts[k][level].keys())
            per_class_acc[k][level] = {}
            for gt_label in list_of_labels:
                sum_in_this_level = (
                    sum_in_this_level + pred_counts[k][level][gt_label] * 1.0 / gt_counts[k][level][gt_label]
                )
                per_class_acc[k][level][gt_label] = (
                    pred_counts[k][level][gt_label] * 1.0 / gt_counts[k][level][gt_label]
                )
            macro_acc_dict[k][level] = sum_in_this_level / len(list_of_labels)

    return macro_acc_dict, per_class_acc

def inference_and_print_result(keys_dict, seen_dict, unseen_dict, args, small_species_list=None, k_list=None):
    acc_dict = {}
    per_class_acc = {}
    if k_list is None:
        k_list = [1, 3, 5]

    max_k = k_list[-1]

    seen_gt_label = seen_dict["label_list"]
    unseen_gt_label = unseen_dict["label_list"]
    keys_label = keys_dict["label_list"]
    try:
        pred_dict = {
            "seen_id": seen_dict["processed_id_list"],
            "seen_gt_label": seen_gt_label,
            "unseen_id": unseen_dict["processed_id_list"],
            "unseen_gt_label": unseen_gt_label,
        }
    except:
        pred_dict = {
            "seen_id": seen_dict["file_name_list"],
            "seen_gt_label": seen_gt_label,
            "unseen_id": unseen_dict["file_name_list"],
            "unseen_gt_label": unseen_gt_label,
        }

    for query_feature_type in All_TYPE_OF_FEATURES_OF_QUERY:
        if query_feature_type not in seen_dict.keys():
            continue
        acc_dict[query_feature_type] = {}
        per_class_acc[query_feature_type] = {}
        pred_dict[query_feature_type] = {}

        for key_feature_type in All_TYPE_OF_FEATURES_OF_KEY:
            if key_feature_type not in keys_dict.keys():
                continue
            acc_dict[query_feature_type][key_feature_type] = {}
            per_class_acc[query_feature_type][key_feature_type] = {}
            pred_dict[query_feature_type][key_feature_type] = {}

            curr_seen_feature = seen_dict[query_feature_type]
            curr_unseen_feature = unseen_dict[query_feature_type]

            curr_keys_feature = keys_dict[key_feature_type]
            if curr_keys_feature is None:
                continue
            if key_feature_type == "all_key_features":
                keys_label = keys_dict["all_key_features_label"]

            if (
                curr_keys_feature is None
                or curr_seen_feature is None
                or curr_unseen_feature is None
                or curr_keys_feature.shape[-1] != curr_seen_feature.shape[-1]
                or curr_keys_feature.shape[-1] != curr_unseen_feature.shape[-1]
            ):
                continue

            curr_seen_pred_list = make_prediction(
                curr_seen_feature, curr_keys_feature, keys_label, with_similarity=False, max_k=max_k
            )
            curr_unseen_pred_list = make_prediction(
                curr_unseen_feature, curr_keys_feature, keys_label, max_k=max_k
            )

            pred_dict[query_feature_type][key_feature_type] = {
                "curr_seen_pred_list": curr_seen_pred_list,
                "curr_unseen_pred_list": curr_unseen_pred_list,
            }

            acc_dict[query_feature_type][key_feature_type]["seen"] = {}
            acc_dict[query_feature_type][key_feature_type]["unseen"] = {}
            acc_dict[query_feature_type][key_feature_type]["seen"]["micro_acc"] = top_k_micro_accuracy(
                curr_seen_pred_list, seen_gt_label, k_list=k_list
            )
            acc_dict[query_feature_type][key_feature_type]["unseen"]["micro_acc"] = top_k_micro_accuracy(
                curr_unseen_pred_list, unseen_gt_label, k_list=k_list
            )

            seen_macro_acc, seen_per_class_acc = top_k_macro_accuracy(
                curr_seen_pred_list, seen_gt_label, k_list=k_list
            )

            unseen_macro_acc, unseen_per_class_acc = top_k_macro_accuracy(
                curr_unseen_pred_list, unseen_gt_label, k_list=k_list
            )

            per_class_acc[query_feature_type][key_feature_type]["seen"] = seen_per_class_acc
            per_class_acc[query_feature_type][key_feature_type]["unseen"] = unseen_per_class_acc

            acc_dict[query_feature_type][key_feature_type]["seen"]["macro_acc"] = seen_macro_acc
            acc_dict[query_feature_type][key_feature_type]["unseen"]["macro_acc"] = unseen_macro_acc

    print_micro_and_macro_acc(acc_dict, k_list, args)

    return acc_dict, per_class_acc, pred_dict

def get_features_and_label(dataloader, model, device, for_key_set=False, for_open_clip=False):
    model.eval()
    file_name_list, encoded_image_feature, encoded_dna_feature, encoded_language_feature, label_list = get_feature_and_label(
        dataloader, model, device, multi_gpu=False, for_open_clip=for_open_clip
    )
    averaged_feature = None
    concatenated_feature = None
    all_key_features = None
    all_key_features_label = None
    if encoded_dna_feature is not None and encoded_image_feature is not None:
        averaged_feature = np.mean([encoded_image_feature, encoded_dna_feature], axis=0)
        concatenated_feature = np.concatenate((encoded_image_feature, encoded_dna_feature), axis=1)

    dictionary_of_split = {
        "file_name_list": file_name_list,
        "encoded_dna_feature": encoded_dna_feature,
        "encoded_image_feature": encoded_image_feature,
        "encoded_language_feature": encoded_language_feature,
        "averaged_feature": averaged_feature,
        "concatenated_feature": concatenated_feature,
        "label_list": label_list,
    }

    if (
        for_key_set
        and encoded_image_feature is not None
        and encoded_dna_feature is not None
        and encoded_language_feature is not None
    ):
        for curr_feature in [encoded_image_feature, encoded_dna_feature, encoded_language_feature]:
            if all_key_features is None:
                all_key_features = curr_feature
                all_key_features_label = label_list
            else:
                all_key_features = np.concatenate((all_key_features, curr_feature), axis=0)
                all_key_features_label = all_key_features_label + label_list

    dictionary_of_split["all_key_features"] = all_key_features
    dictionary_of_split["all_key_features_label"] = all_key_features_label

    return dictionary_of_split

def get_all_unique_species_from_dataloader(dataloader):
    all_species = []

    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch["species"]
    all_species = list(set(all_species))
    return all_species

def scale_learning_rate(lr, batch_size, base_batch_size=500, world_size=1):
    total_batch_size = batch_size * world_size
    lr = lr * total_batch_size / base_batch_size
    return lr


def find_closest_match(query_feature, keys_feature, keys_label, with_similarity=False, with_indices=False, max_k=5):
    index = faiss.IndexFlatIP(keys_feature.shape[-1])
    keys_feature = normalize(keys_feature, norm="l2", axis=1).astype(np.float32)
    query_feature = normalize(query_feature, norm="l2", axis=1).astype(np.float32)
    index.add(keys_feature)
    pred_list = []

    similarities, indices = index.search(query_feature, max_k)
    for key_indices in indices:
        k_pred_in_diff_level = {}
        for level in LEVELS:
            if level not in k_pred_in_diff_level.keys():
                k_pred_in_diff_level[level] = []
            for i in key_indices:
                try:
                    k_pred_in_diff_level[level].append(keys_label[i][level])
                except:
                    print(keys_label)
                    exit()
        pred_list.append(k_pred_in_diff_level)

    out = [pred_list]
    output_dict = {
        "pred_list": pred_list,
        "similarities": similarities,
        "indices": indices
    }
    return output_dict

def create_id_index_map(args):
    if args.model_config.dataset == "bioscan_1m" and os.path.exists(args.bioscan_data.path_to_id_to_position_mapping):
        with open(args.bioscan_data.path_to_id_to_position_mapping, 'r') as file:
            id_index_map = json.load(file)
        return id_index_map
    elif args.model_config.dataset == "bioscan_5m" and os.path.exists(args.bioscan_5m_data.path_to_id_to_position_mapping):
        with open(args.bioscan_5m_data.path_to_id_to_position_mapping, 'r') as file:
            id_index_map = json.load(file)
        return id_index_map
    else:
        if args.model_config.dataset == "bioscan_1m":
            hdf5_path = args.bioscan_data.path_to_hdf5_data
        elif args.model_config.dataset == "bioscan_5m":
            hdf5_path = args.bioscan_5m_data.path_to_hdf5_data

        id_index_map = {}
        with h5py.File(hdf5_path, 'r') as file:
            for group_name in file.keys():
                curr_group = file[group_name]
                if args.model_config.dataset == "bioscan_5m":
                    ids = [item.decode("utf-8") for item in curr_group["processid"]]
                elif args.model_config.dataset == "bioscan_1m":
                    ids = [item.decode("utf-8") for item in curr_group["image_file"]]
                for index, id in enumerate(ids):
                    id_index_map[id] = {'group': group_name, 'index': index}

        # save the id_index_map
        if args.model_config.dataset == "bioscan_1m":
            with open(args.bioscan_data.path_to_id_to_position_mapping, 'w') as file:
                json.dump(id_index_map, file)
        elif args.model_config.dataset == "bioscan_5m":
            with open(args.bioscan_5m_data.path_to_id_to_position_mapping, 'w') as file:
                json.dump(id_index_map, file)

        return id_index_map

def load_image_from_hdf5_with_id_as_input(args, data_id, id_to_split_and_position):
    if args.model_config.dataset == "bioscan_1m":
        hdf5_path = args.bioscan_data.path_to_hdf5_data
    elif args.model_config.dataset == "bioscan_5m":
        hdf5_path = args.bioscan_5m_data.path_to_hdf5_data

    with h5py.File(hdf5_path, 'r') as file:
        group = id_to_split_and_position[data_id]['group']
        idx = id_to_split_and_position[data_id]['index']
        data = file[group]
        enc_length = data["image_mask"][idx]
        image_enc_padded = data["image"][idx].astype(np.uint8)
        image_enc = image_enc_padded[:enc_length]
        image = Image.open(io.BytesIO(image_enc))
        return image.resize((256, 256))

def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("module.", "")] = value
    return new_state_dict



def load_kmer_tokenizer(args, k=4):
    base_pairs = "ACGT"
    # TODO: check if 'tokenize_n_nucleotide' in args.
    tokenize_n_nucleotide = False
    special_tokens = ["[MASK]", "[UNK]"] 
    UNK_TOKEN = "[UNK]"
    stride = 1
    max_len = 660
    
    k_mer = k
    kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=k_mer)]

    if tokenize_n_nucleotide:
        prediction_kmers = []
        other_kmers = []
        for kmer in kmers:
            if "N" in kmer:
                other_kmers.append(kmer)
            else:
                prediction_kmers.append(kmer)

        kmers = prediction_kmers + other_kmers

    kmer_dict = dict.fromkeys(kmers, 1)
    vocab = build_vocab_from_dict(kmer_dict, specials=special_tokens)
    vocab.set_default_index(vocab[UNK_TOKEN])
    vocab_size = len(vocab)
    tokenizer = KmerTokenizer(
        k_mer, vocab, stride=stride, padding=True, max_len=max_len
    )

    return tokenizer

class TensorResizeLongEdge(object):
    def __init__(self, long_edge_size, interpolation_mode='bilinear'):
        self.long_edge_size = long_edge_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, tensor):
        c, h, w = tensor.shape
        scale = self.long_edge_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        tensor = tensor.unsqueeze(0)
        resized_tensor = F.interpolate(tensor, size=(new_h, new_w), mode=self.interpolation_mode, align_corners=False)
        return resized_tensor.squeeze(0)

    def __repr__(self):
        return f'{self.__class__.__name__}(long_edge_size={self.long_edge_size}, interpolation_mode={self.interpolation_mode})'

class PadTo224Tensor(object):
    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, tensor):
        c, h, w = tensor.shape

        if h > self.target_size or w > self.target_size:
            raise ValueError("input tensor is larger than target size")

        pad_h = self.target_size - h
        pad_w = self.target_size - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=self.fill)
        return padded_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(target_size={self.target_size}, fill={self.fill})"