import os

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from bioscanclip.epoch.inference_epoch import convert_label_dict_to_list_of_dict
from bioscanclip.util.dataset import load_dataloader_for_everything_in_5m
from bioscanclip.model.simple_clip import initialize_model_and_load_from_checkpoint

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"

LEVELS = ["order", "family", "genus", "species"]


def convert_labels_to_four_list(list_of_dict):
    order_list = []
    family_list = []
    genus_list = []
    species_list = []
    for a_dict in list_of_dict:
        order_list.append(a_dict["order"])
        family_list.append(a_dict["family"])
        genus_list.append(a_dict["genus"])
        species_list.append(a_dict["species"])
    return order_list, family_list, genus_list, species_list


def write_feature_to_hdf5_file(embedding_dict, hdf5_file, feature_length=768):
    labels = convert_labels_to_four_list(embedding_dict["label_list"])
    label_names = ["order_list", "family_list", "genus_list", "species_list"]
    encoded_names = ["encoded_image_feature", "encoded_dna_feature", "encoded_language_feature"]

    # write file name list
    file_name_list = np.array([s.encode('utf-8') for s in embedding_dict["file_name_list"]])
    if "file_name_list" in hdf5_file:
        original_len = hdf5_file["file_name_list"].shape[0]
        hdf5_file["file_name_list"].resize((original_len + len(file_name_list),))
        hdf5_file["file_name_list"][original_len:] = file_name_list
    else:
        hdf5_file.create_dataset(
            "file_name_list",
            data=file_name_list,
            maxshape=(None,),
            compression='gzip',
            compression_opts=5,
            chunks=True
        )

    for label_name, label_data in zip(label_names, labels):
        label_data = np.array([s.encode('utf-8') for s in label_data])
        if label_name in hdf5_file:
            original_len = hdf5_file[label_name].shape[0]
            hdf5_file[label_name].resize((original_len + len(label_data),))
            hdf5_file[label_name][original_len:] = label_data
        else:
            hdf5_file.create_dataset(
                label_name,
                data=label_data,
                maxshape=(None,),
                compression='gzip',
                compression_opts=5,
                chunks=True
            )

    for encoded_name in encoded_names:
        feature_data = np.array(embedding_dict[encoded_name], dtype='float32')
        if encoded_name in hdf5_file:
            original_len = hdf5_file[encoded_name].shape[0]
            hdf5_file[encoded_name].resize((original_len + feature_data.shape[0], 768))
            hdf5_file[encoded_name][original_len:] = feature_data
        else:
            hdf5_file.create_dataset(
                encoded_name,
                data=feature_data,
                maxshape=(None, feature_length),
                compression='gzip',
                compression_opts=5,
                chunks=(100, feature_length)
            )

def create_feature_dict(file_name_list, encoded_image_feature_list, encoded_dna_feature_list, encoded_text_feature_list,
                        label_list):
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
    embedding_dict = {
        "file_name_list": file_name_list,
        "encoded_dna_feature": encoded_dna_feature_list,
        "encoded_image_feature": encoded_image_feature_list,
        "encoded_language_feature": encoded_text_feature_list,
        "label_list": label_list,
    }
    return embedding_dict


def wirte_feature_to_hdf5_file(dataloader, model, device, hdf5_file, for_open_clip=False, multi_gpu=False,
                               limit_iter_for_write=300):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    model.eval()
    count = 0
    file_name_list = []
    encoded_image_feature_list = []
    encoded_dna_feature_list = []
    encoded_text_feature_list = []
    label_list = []

    with torch.no_grad():
        for step, batch in pbar:

            count += 1
            pbar.set_description(f"Encoding features")
            processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            if for_open_clip:
                language_input = input_ids
            else:
                language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                                  'attention_mask': attention_mask.to(device)}

            image_output, dna_output, language_output, logit_scale, logit_bias = model(image_input_batch.to(device),
                                                                                       dna_input_batch.to(device),
                                                                                       language_input)
            if image_output is not None:
                encoded_image_feature_list = encoded_image_feature_list + F.normalize(image_output,
                                                                                      dim=-1).cpu().tolist()
            if dna_output is not None:
                encoded_dna_feature_list = encoded_dna_feature_list + F.normalize(dna_output, dim=-1).cpu().tolist()
            if language_output is not None:
                encoded_text_feature_list = encoded_text_feature_list + F.normalize(language_output,
                                                                                    dim=-1).cpu().tolist()
            label_list = label_list + convert_label_dict_to_list_of_dict(label_batch)
            file_name_list = file_name_list + list(processid_batch)

            # write to hdf5 file is current is the last iteration
            if count >= limit_iter_for_write:
                embedding_dict = create_feature_dict(file_name_list, encoded_image_feature_list,
                                                     encoded_dna_feature_list, encoded_text_feature_list, label_list)
                write_feature_to_hdf5_file(embedding_dict, hdf5_file, feature_length=768)
                count = 0

                del file_name_list
                del encoded_image_feature_list
                del encoded_dna_feature_list
                del encoded_text_feature_list
                del label_list

                encoded_image_feature_list = []
                encoded_dna_feature_list = []
                encoded_text_feature_list = []
                label_list = []
                file_name_list = []


        if len(encoded_image_feature_list) > 0:
            embedding_dict = create_feature_dict(file_name_list, encoded_image_feature_list, encoded_dna_feature_list,
                                                 encoded_text_feature_list, label_list)
            write_feature_to_hdf5_file(embedding_dict, hdf5_file, feature_length=768)


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join(args.project_root_path,
                                     "new_extracted_embedding", args.model_config.dataset,
                                     args.model_config.model_output_name
                                     )
    os.makedirs(folder_for_saving, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = initialize_model_and_load_from_checkpoint(args)
    model = model.to(device)
    model.eval()

    print("Start processing dataloader...")

    pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader, other_heldout_dataloader = load_dataloader_for_everything_in_5m(
        args)
    dataloaders_that_need_to_be_process = [other_heldout_dataloader, pre_train_dataloader, seen_val_dataloader,
                                           unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader,
                                           all_keys_dataloader]

    print("Done processing dataloader!")

    print("Start extracting features...")

    extracted_features_path = os.path.join(folder_for_saving,
                                           f"extracted_features_for_all_5m_data.hdf5")

    feature_length = args.model_config.output_dim

    if os.path.exists(extracted_features_path):
        os.remove(extracted_features_path)
    model.eval()
    with h5py.File(extracted_features_path, "a") as hdf5_file:
        for index, dataloader in enumerate(dataloaders_that_need_to_be_process):
            print(f"Processing dataloader {index + 1}/{len(dataloaders_that_need_to_be_process)}")
            wirte_feature_to_hdf5_file(
                dataloader, model, device, hdf5_file, multi_gpu=False, for_open_clip=False
            )

    print("Done!")
    print(f"Saved to {extracted_features_path}")


if __name__ == "__main__":
    main()
