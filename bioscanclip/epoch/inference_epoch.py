from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer


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

    tokenizer = AutoTokenizer.from_pretrained("bioscan-ml/BarcodeBERT", trust_remote_code=True)  # Load tokenizer

    with torch.no_grad():
        for step, batch in pbar:
            pbar.set_description(f"Encoding features")
            processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            if for_open_clip:
                language_input = input_ids
            else:
                language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                                  'attention_mask': attention_mask.to(device)}

            # Tokenizing DNA sequences
            tokenized_dna_sequences = []
            attention_masks = []
            for dna_seq in dna_input_batch:
                tokenized_output = tokenizer(dna_seq, padding='max_length', truncation=True, max_length=133, return_tensors="pt")
                input_seq = tokenized_output["input_ids"]
                attention_mask = tokenized_output["attention_mask"]
                tokenized_dna_sequences.append(input_seq)
                attention_masks.append(attention_mask)

            # Convert DNA tokenized sequences into tensors
            dna_input_batch = torch.stack(tokenized_dna_sequences).squeeze(1).to(device)
            attention_masks = torch.stack(attention_masks).squeeze(1).to(device)

            # Forward pass through model
            image_output, dna_output, language_output, logit_scale, logit_bias = model(
                image_input_batch.to(device),
                (dna_input_batch, attention_masks),  # Passing tokenized DNA sequences
                language_input
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

    # Convert lists to numpy arrays
    encoded_image_feature_list = None if len(encoded_image_feature_list) == 0 else np.array(encoded_image_feature_list)
    encoded_dna_feature_list = None if len(encoded_dna_feature_list) == 0 else np.array(encoded_dna_feature_list)
    encoded_text_feature_list = None if len(encoded_text_feature_list) == 0 else np.array(encoded_text_feature_list)

    return file_name_list, encoded_image_feature_list, encoded_dna_feature_list, encoded_text_feature_list, label_list
