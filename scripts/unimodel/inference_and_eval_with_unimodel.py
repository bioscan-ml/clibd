import os
import torch.multiprocessing as mp
import h5py
import hydra
import torch
from omegaconf import DictConfig

from bioscanclip.model.simple_clip import load_vit_for_simclr_training, wrap_vit_into_simple_clip
from bioscanclip.util.dataset import DatasetForSimCLRStyleTraining, prepare, load_bioscan_dataloader_all_small_splits
from bioscanclip.util.simclr import SimCLR
from bioscanclip.util.util import set_seed, get_features_and_label, All_TYPE_OF_FEATURES_OF_KEY, inference_and_print_result, remove_module_from_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP
import json

def calculate_micro_accuracy(all_pred_labels, all_gt_labels):
    levels = ['order', 'family', 'genus', 'species']
    correct_counts = {level: 0 for level in levels}
    total_counts = {level: 0 for level in levels}

    for pred, gt in zip(all_pred_labels, all_gt_labels):
        for level in levels:
            if pred[level] == gt[level]:
                correct_counts[level] += 1
            total_counts[level] += 1

    micro_accuracies = {}
    for level in levels:
        micro_accuracies[level] = correct_counts[level] / total_counts[level] if total_counts[level] else 0

    return micro_accuracies


def main_process(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_for_saving = os.path.join(
        args.project_root_path, "extracted_embedding", args.model_config.dataset, args.model_config.model_output_name
    )
    os.makedirs(folder_for_saving, exist_ok=True)
    labels_path = os.path.join(folder_for_saving, f"labels_{args.inference_and_eval_setting.eval_on}.json")
    processed_id_path = os.path.join(folder_for_saving, f"processed_id_{args.inference_and_eval_setting.eval_on}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted_features_path = os.path.join(
        folder_for_saving, f"extracted_feature_from_{args.inference_and_eval_setting.eval_on}_split.hdf5"
    )


    if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
        hdf5_inputs_path = (args.bioscan_5m_data.path_to_smaller_hdf5_data
                            if hasattr(args.model_config,
                                       "train_with_small_subset") and args.model_config.train_with_small_subset
                            else args.bioscan_5m_data.path_to_hdf5_data)

    else:
        hdf5_inputs_path = args.bioscan_data.path_to_hdf5_data

    model = load_vit_for_simclr_training(args, device=None)
    model = model.to(device)
    from collections import OrderedDict

    checkpoint = torch.load(args.model_config.ckpt_path)
    state_dict = checkpoint['state_dict']

    new_state_dict = remove_module_from_state_dict(state_dict)
    # for k, v in state_dict.items():
    #     if k.startswith('module.'):
    #         new_key = k[7:]
    #     else:
    #         new_key = k
    #     new_state_dict[new_key] = v


    model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(args.model_config.ckpt_path)['state_dict'].module)

    # Wrap vit into SimpleCLIP
    model = wrap_vit_into_simple_clip(args, model, device=device)

    print("Model loaded from %s" % args.model_config.ckpt_path)

    if args.inference_and_eval_setting.eval_on == "val":
        (
            _,
            seen_dataloader,
            unseen_dataloader,
            _,
            _,
            seen_keys_dataloader,
            val_unseen_keys_dataloader,
            test_unseen_keys_dataloader,
            all_keys_dataloader,
        ) = load_bioscan_dataloader_all_small_splits(args)
    elif args.inference_and_eval_setting.eval_on == "test":
        (
            _,
            _,
            _,
            seen_dataloader,
            unseen_dataloader,
            seen_keys_dataloader,
            val_unseen_keys_dataloader,
            test_unseen_keys_dataloader,
            all_keys_dataloader,
        ) = load_bioscan_dataloader_all_small_splits(args)
    else:
        raise ValueError("Invalid eval_on setting")

    keys_dict = get_features_and_label(
        all_keys_dataloader, model, device, for_key_set=True, for_open_clip=False
    )

    seen_dict = get_features_and_label(seen_dataloader, model, device, for_open_clip=False)

    unseen_dict = get_features_and_label(unseen_dataloader, model, device, for_open_clip=False)

    if args.save_inference and not (
            os.path.exists(extracted_features_path)
            and os.path.exists(labels_path)
    ):
        # Open (or create) the HDF5 file for writing
        with h5py.File(extracted_features_path, "w") as new_file:
            name_of_splits = ["seen", "unseen", "key"]
            split_dicts = [seen_dict, unseen_dict, keys_dict]

            for split_name, split in zip(name_of_splits, split_dicts):
                group = new_file.create_group(split_name)
                for embedding_type in All_TYPE_OF_FEATURES_OF_KEY:
                    if embedding_type in split:
                        try:
                            # Attempt to create the dataset
                            group.create_dataset(embedding_type, data=split[embedding_type])
                            print(f"Created dataset for {embedding_type}")
                        except (ValueError, TypeError, OSError) as e:
                            # Only catch dataset-creation errors
                            print(f"Error creating dataset for {embedding_type}: {e}")
        new_file.close()
        total_dict = {
            "seen_gt_dict": seen_dict["label_list"],
            "unseen_gt_dict": unseen_dict["label_list"],
            "key_gt_dict": keys_dict["label_list"],
        }
        with open(labels_path, "w") as json_file:
            json.dump(total_dict, json_file, indent=4)
        id_dict = {
            "seen_id_list": seen_dict["file_name_list"],
            "unseen_id_list": unseen_dict["file_name_list"],
            "key_id_list": keys_dict["file_name_list"],
        }
        with open(processed_id_path, "w") as json_file:
            json.dump(id_dict, json_file, indent=4)

    acc_dict, per_class_acc, pred_dict = inference_and_print_result(
        keys_dict,
        seen_dict,
        unseen_dict,
        args,
        small_species_list=None,
        k_list=args.inference_and_eval_setting.k_list,
    )
    path_to_acc_dict = os.path.join(folder_for_saving, f"acc_dict_{args.inference_and_eval_setting.eval_on}.json")
    with open(path_to_acc_dict, "w") as json_file:
        json.dump(acc_dict, json_file, indent=4)
    print(f"Saved acc_dict to {path_to_acc_dict}")





@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    torch.cuda.empty_cache()
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

    default_seed = args.default_seed
    if hasattr(args.model_config, 'default_seed'):
        default_seed = args.model_config.default_seed

    if hasattr(args.model_config, 'random_seed') and args.model_config.random_seed:
        seed = set_seed();
        string = "random seed"
    else:
        seed = set_seed(seed=int(default_seed));
        string = "default seed"
    print("The module is run with %s: %d" % (string, seed))
    main_process(args)

if __name__ == '__main__':
    main()
