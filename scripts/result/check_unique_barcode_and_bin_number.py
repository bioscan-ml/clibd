import os

import h5py
import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    # load path to bioscan-1m hdf5
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    path_to_5m = args.bioscan_5m_data.path_to_hdf5_data
    path_to_1m_sub_set_of_5m = args.bioscan_5m_data.path_to_smaller_hdf5_data

    bioscan_5m_tsv_path = args.bioscan_5m_data.path_to_tsv_data
    bioscan_5m_tsv = pd.read_csv(bioscan_5m_tsv_path, sep="\,")
    # print columns of the tsv

    # create a dictionary. Key: sampleid, value: dna_bin
    sample_id_to_dna_bin = {}
    # load sampleid and dna_bin as two lists
    sample_id_list = bioscan_5m_tsv['sampleid'].tolist()
    dna_bin_list = bioscan_5m_tsv['dna_bin'].tolist()
    # create a dictionary
    sample_id_to_dna_bin = dict(zip(sample_id_list, dna_bin_list))
    # close the tsv file
    bioscan_5m_tsv = None

    print("For BIOSCAN-5M subset")
    # load bioscan-5m subset hdf5
    with h5py.File(path_to_1m_sub_set_of_5m, "r") as f:
        pre_train_group = f['no_split_and_seen_train']
        barcode_from_pre_train = [item.decode("utf-8") for item in pre_train_group['barcode'][:]]
        unique_barcode_from_pre_train = set(barcode_from_pre_train)
        print("The number of unique barcode in pre-train: ", len(unique_barcode_from_pre_train))
        # get sample_id for the pre-train
        sample_id_from_pre_train = [item.decode("utf-8") for item in pre_train_group['sampleid'][:]]
        # with the list of sample_id, get the list of dna_bin
        dna_bin_from_pre_train = [sample_id_to_dna_bin[sample_id] for sample_id in sample_id_from_pre_train]
        unique_dna_bin_from_pre_train = set(dna_bin_from_pre_train)
        print("The number of unique dna_bin in pre-train: ", len(unique_dna_bin_from_pre_train))

        species_from_pre_train = [item.decode("utf-8") for item in pre_train_group['species'][:]]
        unique_species_from_pre_train = set(species_from_pre_train)
        print("The number of unique species in pre-train: ", len(unique_species_from_pre_train))


    print("For BIOSCAN-5M")
    with h5py.File(path_to_5m, "r") as f:
        pre_train_group = f['no_split_and_seen_train']
        barcode_from_pre_train = [item.decode("utf-8") for item in pre_train_group['barcode'][:]]
        unique_barcode_from_pre_train = set(barcode_from_pre_train)
        print("The number of unique barcode in pre-train: ", len(unique_barcode_from_pre_train))
        # get sample_id for the pre-train
        sample_id_from_pre_train = [item.decode("utf-8") for item in pre_train_group['sampleid'][:]]
        # with the list of sample_id, get the list of dna_bin
        dna_bin_from_pre_train = [sample_id_to_dna_bin[sample_id] for sample_id in sample_id_from_pre_train]
        unique_dna_bin_from_pre_train = set(dna_bin_from_pre_train)
        print("The number of unique dna_bin in pre-train: ", len(unique_dna_bin_from_pre_train))
        species_from_pre_train = [item.decode("utf-8") for item in pre_train_group['species'][:]]
        unique_species_from_pre_train = set(species_from_pre_train)
        print("The number of unique species in pre-train: ", len(unique_species_from_pre_train))





if __name__ == "__main__":
    main()
