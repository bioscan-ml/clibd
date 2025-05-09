import json
import json
import os

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import silhouette_samples
from tqdm import tqdm

from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import get_features_and_label, \
    All_TYPE_OF_FEATURES_OF_KEY
from bioscanclip.model.simple_clip import load_clip_model

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"

QUERY_AND_KEY_FEATURES = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
]


def avg_list(l):
    return sum(l) * 1.0 / len(l)


def calculate_silhouette_score(args, image_features, labels):
    for level in ["order", "family", "genus", "species"]:
        gt_list = [labels[1][i][level] for i in range(len(labels[1]))]
        silhouette_score = silhouette_samples(image_features, gt_list)
        print(f"The silhouette score for {level} level is : {avg_list(silhouette_score)}")


def check_for_acc_about_correct_predict_seen_or_unseen(final_pred_list, species_list):
    for k in [1, 3, 5]:
        correct = 0
        total = 0
        for record in final_pred_list:
            top_k_species = record["species"]
            curr_top_k_pred = top_k_species[:k]
            for single_pred in curr_top_k_pred:
                if single_pred in species_list:
                    correct = correct + 1
                    break
            total = total + 1

        print(f"for k = {k}: {correct * 1.0 / total}")


def get_similarity_for_different_combination_of_modalities(
        keys_dict,
        seen_dict,
        unseen_dict,
        args,
):
    # build a dict for key_dict with species name as key, and their rest information as value
    key_dict_with_species_as_key = {}
    for idx, file_name in enumerate(keys_dict['processed_id_list']):
        curr_species = keys_dict['label_list'][idx]['species']
        if curr_species not in key_dict_with_species_as_key:
            key_dict_with_species_as_key[curr_species] = []
        curr_instance = {}
        for type_of_info in keys_dict.keys():
            curr_instance[type_of_info] = keys_dict[type_of_info][idx]
        key_dict_with_species_as_key[curr_species].append(curr_instance)

    # Using file_name as key, record other information and distance to the key with the same species
    similarity_dict = {}
    for seen_or_unseen in ["seen", "unseen"]:
        if seen_or_unseen == "seen":
            query_dict = seen_dict
        else:
            query_dict = unseen_dict
        similarity_dict[seen_or_unseen] = {}
        pbar = tqdm(enumerate(query_dict['processed_id_list']), total=len(query_dict['processed_id_list']))
        for idx, file_name in pbar:
            pbar.set_description(f"Processing {seen_or_unseen} data")
            # Store the information of the current instance
            curr_instance_with_distance_information = {}
            for type_of_info in query_dict.keys():
                curr_instance_with_distance_information[type_of_info] = query_dict[type_of_info][idx]
            curr_instance_with_distance_information['smallest_distance'] = {}
            curr_species = query_dict['label_list'][idx]['species']
            key_list_with_same_species = key_dict_with_species_as_key[curr_species]

            for query in QUERY_AND_KEY_FEATURES:
                for key in QUERY_AND_KEY_FEATURES:
                    query_feature = curr_instance_with_distance_information[query]
                    # fine smallest distance
                    smallest_distance = None

                    for key_instance in key_list_with_same_species:
                        key_feature = key_instance[key]
                        distance = np.linalg.norm(query_feature - key_feature)
                        if smallest_distance is None or distance < smallest_distance:
                            smallest_distance = distance
                    curr_instance_with_distance_information['smallest_distance'][f"{query}_{key}"] = smallest_distance
            similarity_dict[seen_or_unseen][file_name] = curr_instance_with_distance_information

    list_of_query_info = []

    for split in ['seen', 'unseen']:
        for curr_query_file_name in similarity_dict[split].keys():
            curr_query_instance = similarity_dict[split][curr_query_file_name]
            curr_query_info = {}
            curr_query_info['file_name'] = curr_query_file_name
            curr_query_info['order'] = curr_query_instance['label_list']['order']
            curr_query_info['family'] = curr_query_instance['label_list']['family']
            curr_query_info['genus'] = curr_query_instance['label_list']['genus']
            curr_query_info['species'] = curr_query_instance['label_list']['species']
            curr_query_info['distance_for_image_to_image'] = curr_query_instance['smallest_distance'][
                'encoded_image_feature_encoded_image_feature']
            curr_query_info['distance_for_dna_to_dna'] = curr_query_instance['smallest_distance'][
                'encoded_dna_feature_encoded_dna_feature']
            curr_query_info['distance_for_image_to_dna'] = curr_query_instance['smallest_distance'][
                'encoded_image_feature_encoded_dna_feature']
            curr_query_info['distance_for_dna_to_image'] = curr_query_instance['smallest_distance'][
                'encoded_dna_feature_encoded_image_feature']
            curr_query_info['distance_for_image_to_language'] = curr_query_instance['smallest_distance'][
                'encoded_dna_feature_encoded_language_feature']
            curr_query_info['split'] = args.inference_and_eval_setting.eval_on + "_" + split
            list_of_query_info.append(curr_query_info)
    df = pd.DataFrame(list_of_query_info)
    folder_path = os.path.join(args.project_root_path, "distribution_of_similarities")
    df.to_csv(os.path.join(folder_path,
                           f'{args.inference_and_eval_setting.eval_on}_query_info_with_distance_to_nearest_key_with_same_species.csv'),
              index=False)
    print(f"Saved the query info with distance to nearest key with same species at {folder_path}")

    # For df, get the column call encoded_image_feature_encoded_image_feature as a list, then find the max and min
    distance_for_image_to_image_in_list = df['distance_for_image_to_image'].tolist()
    distance_for_dna_to_dna_in_list = df['distance_for_dna_to_dna'].tolist()
    distance_for_image_to_dna_in_list = df['distance_for_image_to_dna'].tolist()
    distance_for_image_to_language_in_list = df['distance_for_image_to_language'].tolist()

    bins = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

    plt.figure(figsize=(8, 4))

    bin_centers = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))

    plt.hist([distance_for_image_to_image_in_list, distance_for_dna_to_dna_in_list, distance_for_image_to_dna_in_list, distance_for_image_to_language_in_list],
             bins=bins, label=['Image-to-Image', 'DNA-to-DNA', 'Image-to-DNA', 'Image-to-Text'], alpha=0.7, rwidth=0.85)

    plt.legend()

    plt.title('Distance Distribution Comparison')
    plt.xlabel('Distance')
    plt.ylabel('Frequencies')

    plt.yscale('log')

    plt.xticks(bin_centers, [f"{left:.3f}~{right:.3f}" for left, right in zip(bins[:-1], bins[1:])])

    plt.subplots_adjust(top=0.93, bottom=0.120, left=0.0075, right=0.950)

    plt.show()


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join(args.project_root_path,
                                     "extracted_embedding", args.model_config.dataset,
                                     args.model_config.model_output_name
                                     )
    os.makedirs(folder_for_saving, exist_ok=True)

    labels_path = os.path.join(folder_for_saving, f"labels_{args.inference_and_eval_setting.eval_on}.json")
    processed_id_path = os.path.join(folder_for_saving, f"processed_id_{args.inference_and_eval_setting.eval_on}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted_features_path = os.path.join(folder_for_saving,
                                           f"extracted_feature_from_{args.inference_and_eval_setting.eval_on}_split.hdf5")
    args.load_inference = True
    if os.path.exists(extracted_features_path) and os.path.exists(labels_path) and args.load_inference:
        print("Loading embeddings from file...")

        with h5py.File(extracted_features_path, 'r') as hdf5_file:
            seen_dict = {

            }
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["seen"].keys():
                    seen_dict[type_of_feature] = hdf5_file["seen"][type_of_feature][:]

            unseen_dict = {
            }
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["unseen"].keys():
                    unseen_dict[type_of_feature] = hdf5_file["unseen"][type_of_feature][:]
            keys_dict = {
            }
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["key"].keys():
                    keys_dict[type_of_feature] = hdf5_file["key"][type_of_feature][:]

        with open(labels_path, "r") as json_file:
            total_dict = json.load(json_file)
        seen_dict["label_list"] = total_dict["seen_gt_dict"]
        unseen_dict["label_list"] = total_dict["unseen_gt_dict"]
        keys_dict["label_list"] = total_dict["key_gt_dict"]
        keys_dict["all_key_features_label"] = total_dict["key_gt_dict"] + total_dict["key_gt_dict"] + total_dict[
            "key_gt_dict"]

        with open(processed_id_path, "r") as json_file:
            id_dict = json.load(json_file)
        seen_dict["processed_id_list"] = id_dict['seen_id_list']
        unseen_dict["processed_id_list"] = id_dict['unseen_id_list']
        keys_dict["processed_id_list"] = id_dict['key_id_list']
        keys_dict["all_processed_id_list"] = id_dict['key_id_list'] + id_dict['key_id_list'] + id_dict['key_id_list']
    else:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize model
        print("Initialize model...")
        model = load_clip_model(args, device)
        if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
            pass
        else:
            checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
            model.load_state_dict(checkpoint)
        model.eval()
        folder_for_saving = os.path.join(args.project_root_path,
                                         "distribution_of_similarities", args.inference_and_eval_setting.eval_on,
                                         args.model_config.dataset,
                                         args.model_config.model_output_name
                                         )
        os.makedirs(folder_for_saving, exist_ok=True)
        if args.inference_and_eval_setting.eval_on == "val":
            _, seen_dataloader, unseen_dataloader, _, _, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
                args)
        elif args.inference_and_eval_setting.eval_on == "test":
            _, _, _, seen_dataloader, unseen_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
                args)
        else:
            raise ValueError(
                "Invalid value for eval_on, specify by 'python inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl_ver_0_1_2.yaml' inference_and_eval_setting.eval_on=test/val'")
        for_open_clip = False
        if hasattr(args.model_config, "for_open_clip"):
            for_open_clip = args.model_config.for_open_clip
        keys_dict = get_features_and_label(all_keys_dataloader, model, device, for_key_set=True,
                                           for_open_clip=for_open_clip)
        seen_dict = get_features_and_label(seen_dataloader, model, device, for_open_clip=for_open_clip)
        unseen_dict = get_features_and_label(unseen_dataloader, model, device, for_open_clip=for_open_clip)

    get_similarity_for_different_combination_of_modalities(
        keys_dict,
        seen_dict,
        unseen_dict,
        args,
    )


if __name__ == "__main__":
    main()
