import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import h5py

from bioscanclip.epoch.inference_epoch import get_feature_and_label
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import get_features_and_label, initialize_model_and_load_from_checkpoint

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"
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

@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join( args.project_root_path,
        "new_extracted_embedding", args.model_config.dataset, args.model_config.model_output_name
    )
    os.makedirs(folder_for_saving, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = initialize_model_and_load_from_checkpoint(args)
    model = model.to(device)
    model.eval()

    # Load data
    args.model_config.batch_size = 24
    train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
        args)
    if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
        dataloaders_that_need_to_be_process = {"seen_val_dataloader": seen_val_dataloader,
                                               "unseen_val_dataloader": unseen_val_dataloader,
                                               "seen_test_dataloader": seen_test_dataloader,
                                               "unseen_test_dataloader": unseen_test_dataloader,
                                               "seen_keys_dataloader": seen_keys_dataloader,
                                               "unseen_keys_dataloader": val_unseen_keys_dataloader,
                                               "all_keys_dataloader": all_keys_dataloader}
    else:
        dataloaders_that_need_to_be_process = {"train_seen_dataloader": train_seen_dataloader,
                                               "seen_val_dataloader": seen_val_dataloader,
                                               "unseen_val_dataloader": unseen_val_dataloader,
                                               "seen_test_dataloader": seen_test_dataloader,
                                               "unseen_test_dataloader": unseen_test_dataloader,
                                               "seen_keys_dataloader": seen_keys_dataloader,
                                               "val_unseen_keys_dataloader": val_unseen_keys_dataloader,
                                               "test_unseen_keys_dataloader": test_unseen_keys_dataloader,
                                               "all_keys_dataloader": all_keys_dataloader}

    for key, dataloader in dataloaders_that_need_to_be_process.items():
        print(f"Processing {key}...")
        split = key.split('_')[:-1]
        split_name = "_".join(split)
        extracted_features_path = os.path.join(folder_for_saving,
                                               f"extracted_features_of_{split_name}.hdf5")
        labels_path = os.path.join(folder_for_saving, f"labels_{key}.npy")
        if args.save_inference and not (os.path.exists(extracted_features_path) and os.path.exists(labels_path)):

            embedding_dict = get_features_and_label(dataloader, model, device)


            new_file = h5py.File(extracted_features_path, "w")

            order_list, family_list, genus_list, species_list = convert_labels_to_four_list(embedding_dict["label_list"])

            order_list = [s.encode('utf-8') for s in order_list]
            new_file.create_dataset("order_list", data=order_list)

            family_list = [s.encode('utf-8') for s in family_list]
            new_file.create_dataset("family_list", data=family_list)

            genus_list = [s.encode('utf-8') for s in genus_list]
            new_file.create_dataset("genus_list", data=genus_list)

            species_list = [s.encode('utf-8') for s in species_list]
            new_file.create_dataset("species_list", data=species_list)


            id_list = [s.encode('utf-8') for s in embedding_dict["file_name_list"]]
            if args.model_config.dataset == "bioscan_5m":
                new_file.create_dataset("processid", data=id_list)
            else:
                new_file.create_dataset("file_name", data=id_list)

            new_file.create_dataset("encoded_image_feature", data=embedding_dict["encoded_image_feature"])
            new_file.create_dataset("encoded_dna_feature", data=embedding_dict["encoded_dna_feature"])
            if hasattr(args.model_config, "language"):
                new_file.create_dataset("encoded_language_feature", data=embedding_dict["encoded_language_feature"])

if __name__ == "__main__":
    main()
