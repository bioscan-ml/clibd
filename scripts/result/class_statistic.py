import os

import h5py
import hydra
from omegaconf import DictConfig


def show_pretrain_data_seen_unseen_overlap(taxon_level, hdf5_file):
    seen_key_split_list = [item.decode("utf-8") for item in hdf5_file['seen_keys'][taxon_level][:]]
    unique_classes_in_seen_keys = set(seen_key_split_list)

    val_unseen_keys_split_list = [item.decode("utf-8") for item in hdf5_file['val_unseen_keys'][taxon_level][:]]
    test_unseen_keys_split_list = [item.decode("utf-8") for item in hdf5_file['test_unseen_keys'][taxon_level][:]]
    unique_classes_in_unseen_keys = set(val_unseen_keys_split_list + test_unseen_keys_split_list)
    print(f"taxon_level: {taxon_level}")
    # print(len(unique_classes_in_seen_keys))
    # print(len(unique_classes_in_unseen_keys))
    # print(f"the overlap between seen and unseen classes: {len(unique_classes_in_seen_keys.intersection(unique_classes_in_unseen_keys))}")

    classes_in_no_split = [item.decode("utf-8") for item in hdf5_file['no_split'][taxon_level][:]]
    unique_classes_in_no_split = set(classes_in_no_split)
    seen_classes_in_no_split = unique_classes_in_seen_keys.intersection(unique_classes_in_no_split)
    unseen_classes_in_no_split = unique_classes_in_unseen_keys.intersection(unique_classes_in_no_split)
    print(f'Unique {taxon_level} in no_split: {len(unique_classes_in_no_split)}')
    print(f'Seen {taxon_level} in no_split: {len(seen_classes_in_no_split)}')
    print(f'Unseen {taxon_level} in no_split: {len(unseen_classes_in_no_split)}')
    overlap_classes_in_no_split = seen_classes_in_no_split.intersection(unseen_classes_in_no_split)
    print(f'Overlap {taxon_level} in no_split: {len(overlap_classes_in_no_split)}')
    print()


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    # load path to bioscan-1m hdf5
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    path_to_bioscan_1m = args.bioscan_data.path_to_hdf5_data
    # load bioscan-1m hdf5

    with h5py.File(path_to_bioscan_1m, "r") as f:
        keys = list(f.keys())
        print(f"keys: {keys}")
        for taxon_level in ["order", "family", "genus", "species"]:
            show_pretrain_data_seen_unseen_overlap(taxon_level, f)




if __name__ == "__main__":
    main()
