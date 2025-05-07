from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def convert_label_dict_to_list_of_dict(label_batch):
    order = label_batch['order']

    family = label_batch['family']
    genus = label_batch['genus']
    species = label_batch['species']

    list_of_dict = [
        {'order': o, 'family': f, 'genus': g, 'species': s}
        for o, f, g, s in zip(order, family, genus, species)
    ]

    return list_of_dict

def show_confusion_metrix(ground_truth_labels, predicted_labels, path_to_save=None, labels=None, normalize=True):
    plt.figure(figsize=(12, 12))
    if labels is None:
        labels = list(set(ground_truth_labels))
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels, labels=labels)
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False,xticklabels=labels,
                yticklabels=labels)
    plt.xticks(rotation=30)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")

    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()


def get_feature_and_label(dataloader, model, device, for_open_clip=False, multi_gpu=False):
    """
    Extracts features and labels from the dataloader using the given model.
    Tokenizes DNA sequences using AutoTokenizer from "bioscan-ml/BarcodeBERT".
    """
    encoded_image_feature_list = []
    encoded_dna_feature_list = []
    encoded_text_feature_list = []
    label_list = []
    file_name_list =[]

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    model.eval()
    with torch.no_grad():
        for step, batch in pbar:
            pbar.set_description(f"Encoding features")
            processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            if for_open_clip:
                language_input_batch = input_ids
            else:
                language_input_batch = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                                    'attention_mask': attention_mask.to(device)}

            # Forward pass through model
            image_output, dna_output, language_output, logit_scale, logit_bias = model(
                image_input_batch.to(device),
                dna_input_batch,
                language_input_batch
            )

            # Normalizing and storing outputs
            if image_output is not None:
                encoded_image_feature_list.extend(F.normalize(image_output, dim=-1).cpu().tolist())
            if dna_output is not None:
                encoded_dna_feature_list.extend(F.normalize(dna_output, dim=-1).cpu().tolist())
            if language_output is not None:
                encoded_text_feature_list.extend(F.normalize(language_output, dim=-1).cpu().tolist())

            label_list.extend(convert_label_dict_to_list_of_dict(label_batch))
            file_name_list.extend(list(processid_batch))

    if len(encoded_image_feature_list) == 0:
        encoded_image_feature_list = None
    else:
        encoded_image_feature_list = np.array(encoded_image_feature_list)
    if len(encoded_dna_feature_list) == 0:
        encoded_dna_feature_list = None
    else:
        encoded_dna_feature_list = np.array(encoded_dna_feature_list)
    if len(encoded_text_feature_list) == 0:
        encoded_text_feature_list = None
    else:
        encoded_text_feature_list = np.array(encoded_text_feature_list)

    return file_name_list, encoded_image_feature_list, encoded_dna_feature_list, encoded_text_feature_list, label_list
