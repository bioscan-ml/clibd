import os
import random
from collections import defaultdict

import h5py
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

"""
This script is using to generate a special one fifth pre-train data for the BIOSCAN-5M's pre-train split.
Note, any config is ok for this script, as it only need to access the data.
"""


def special_round_to_avoid_zero(number):
    if number < 1:
        return 1
    else:
        return int(round(number))


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:

    random.seed(42)

    path_to_5m_hdf5 = args.bioscan_5m_data.path_to_hdf5_data
    bioscan_5m_hdf5_file = h5py.File(path_to_5m_hdf5, "r")
    pre_train_split_5m = bioscan_5m_hdf5_file["no_split_and_seen_train"]

    path_to_1m_hdf5 = args.bioscan_data.path_to_hdf5_data
    bioscan_1m_hdf5_file = h5py.File(path_to_1m_hdf5, "r")
    pre_train_split_1m = bioscan_1m_hdf5_file["no_split_and_seen_train"]

    species_list_1m = [item.decode("utf-8") for item in pre_train_split_1m["species"][:]]
    species_list_5m = [item.decode("utf-8") for item in pre_train_split_5m["species"][:]]

    ratio_we_want_to_keep = len(species_list_1m) / len(species_list_5m)

    del pre_train_split_1m
    del species_list_1m
    bioscan_1m_hdf5_file.close()

    idx_without_species_label = []
    idx_with_species_label_and_we_want_to_keep = []
    species_to_their_index = defaultdict(list)
    pbar = tqdm(enumerate(species_list_5m), total=len(species_list_5m), desc="Processing species list")
    for idx, species in pbar:
        if species == "not_classified":
            idx_without_species_label.append(idx)
        else:
            species_to_their_index[species].append(idx)

    idx_we_want_to_keep = []

    idx_without_species_label_and_we_want_to_keep = random.sample(idx_without_species_label,
                                              special_round_to_avoid_zero(len(idx_without_species_label)
                                                                          * ratio_we_want_to_keep))

    idx_we_want_to_keep =  idx_we_want_to_keep + idx_without_species_label_and_we_want_to_keep
    pbar = tqdm(species_to_their_index.items(), total=len(species_to_their_index), desc="Processing species to their index")
    for species, idx_list in pbar:
        idx_we_want_to_keep = idx_we_want_to_keep + random.sample(idx_list,
                                                                  special_round_to_avoid_zero(len(idx_list)
                                                                                              * ratio_we_want_to_keep))
    # create a special split and save to a new hdf5 file based on the idx_we_want_to_keep
    special_split_hdf5_path = os.path.join(args.bioscan_5m_data.dir, "special_pre_train_5m.hdf5")
    # remove the file if it exists
    if os.path.exists(special_split_hdf5_path):
        os.remove(special_split_hdf5_path)
    new_file = h5py.File(special_split_hdf5_path, "w")
    new_file.create_group("no_split_and_seen_train")
    for key in bioscan_5m_hdf5_file["no_split_and_seen_train"].keys():
        new_data_list = []
        pbar = tqdm(idx_we_want_to_keep, total=len(idx_we_want_to_keep), desc=f"Processing {key}")
        for idx in pbar:
            new_data_list.append(bioscan_5m_hdf5_file["no_split_and_seen_train"][key][idx])
        new_file["no_split_and_seen_train"].create_dataset(key, data=new_data_list)
    new_file.close()
    bioscan_5m_hdf5_file.close()

if __name__ == "__main__":
    main()
