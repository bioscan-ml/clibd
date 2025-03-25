import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import h5py

from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_dataloader_for_everything_in_5m, load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import get_features_and_label

from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from bioscanclip.epoch.inference_epoch import convert_label_dict_to_list_of_dict
import torch

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


def extract_feature(dataloader, model, device, for_open_clip=False, multi_gpu=False,
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

            print(file_name_list[0])
            print(encoded_image_feature_list[0])
            exit()


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    print("Initialize model...")
    model = load_clip_model(args, device)
    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)

    print("Model loaded!")
    print("Start processing dataloader...")

    (_,
    seen_dataloader,
    unseen_dataloader,
    _,
    _,
    seen_keys_dataloader,
    val_unseen_keys_dataloader,
    test_unseen_keys_dataloader,
    all_keys_dataloader,

    ) = load_bioscan_dataloader_all_small_splits(args)


    print("Done processing dataloader!")

    print("Start extracting features...")

    feature_length = args.model_config.output_dim

    model.eval()

    extract_feature(
        seen_dataloader, model, device, multi_gpu=False, for_open_clip=False
    )



if __name__ == "__main__":
    main()
#
# GMSPA11989-21
# [-0.03652520105242729, -0.08753364533185959, 0.01263172086328268, -0.018361177295446396,
# GMSPA11989-21
# [-0.036525171250104904, -0.08753366023302078, 0.01263170875608921, -0.018361205235123634,