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


def iter_feature_and_label_batches(
        dataloader,
        model,
        device,
        for_open_clip=False,
        multi_gpu=False,
        tokenizer=None,
        show_progress=True,
        encode_language=True,
):
    """逐 batch 产出归一化后的模型 embedding。

    这是 :func:`get_feature_and_label` 的流式版本，用于 embedding 无法安全地
    全部保存在内存中的大型数据集。它保持原 CLIBD 的 batch 输入格式、DNA
    tokenizer、model forward 和 L2 normalization 行为，同时允许调用方跳过
    本任务不需要的 language embedding。
    """
    del multi_gpu  # Kept in the signature for parity with get_feature_and_label.
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "bioscan-ml/BarcodeBERT", trust_remote_code=True
        )

    iterator = enumerate(dataloader)
    if show_progress:
        iterator = tqdm(iterator, total=len(dataloader))

    model.eval()
    with torch.no_grad():
        for step, batch in iterator:
            if show_progress:
                iterator.set_description("Encoding features")
            (
                processid_batch,
                image_input_batch,
                dna_input_batch,
                input_ids,
                token_type_ids,
                attention_mask,
                label_batch,
            ) = batch

            if not encode_language:
                language_input = None
            elif for_open_clip:
                language_input = input_ids
            else:
                language_input = {
                    "input_ids": input_ids.to(device),
                    "token_type_ids": token_type_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                }

            if isinstance(dna_input_batch, torch.Tensor):
                dna_input_batch = dna_input_batch.to(device)
            else:
                tokenized_dna_sequences = []
                for dna_seq in dna_input_batch:
                    tokenized_output = tokenizer(
                        dna_seq,
                        padding="max_length",
                        truncation=True,
                        max_length=133,
                        return_tensors="pt",
                    )
                    tokenized_dna_sequences.append(tokenized_output["input_ids"])
                dna_input_batch = (
                    torch.stack(tokenized_dna_sequences).squeeze(1).to(device)
                )

            image_output, dna_output, language_output, logit_scale, logit_bias = model(
                image_input_batch.to(device),
                dna_input_batch,
                language_input,
            )

            def normalized_numpy(output):
                """将一个可选模型输出做 L2 归一化并转为 CPU NumPy array。"""
                if output is None:
                    return None
                return F.normalize(output, dim=-1).cpu().numpy()

            yield {
                "step": step,
                "file_name_list": list(processid_batch),
                "encoded_image_feature": normalized_numpy(image_output),
                "encoded_dna_feature": normalized_numpy(dna_output),
                "encoded_text_feature": normalized_numpy(language_output),
                "label_list": convert_label_dict_to_list_of_dict(label_batch),
            }


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

    tokenizer = AutoTokenizer.from_pretrained("bioscan-ml/BarcodeBERT", trust_remote_code=True)  # Load tokenizer
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    model.eval()
    with torch.no_grad():
        for step, batch in pbar:
            pbar.set_description(f"Encoding features")
            processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            if for_open_clip:
                language_input = input_ids
            else:
                language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                                  'attention_mask': attention_mask.to(device)}

            if isinstance(dna_input_batch, torch.Tensor):
                dna_input_batch = dna_input_batch.to(device)
            else:
            # Tokenizing DNA sequences
                tokenized_dna_sequences = []
                for dna_seq in dna_input_batch:
                    tokenized_output = tokenizer(dna_seq, padding='max_length', truncation=True, max_length=133, return_tensors="pt")
                    input_seq = tokenized_output["input_ids"]
                    tokenized_dna_sequences.append(input_seq)
                # Convert DNA tokenized sequences into tensors
                dna_input_batch = torch.stack(tokenized_dna_sequences).squeeze(1).to(device)

            # Forward pass through model
            image_output, dna_output, language_output, logit_scale, logit_bias = model(
                image_input_batch.to(device),
                dna_input_batch,  # Passing tokenized DNA sequences
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
