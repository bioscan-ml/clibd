import copy
import os
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_with_train_seen_and_separate_keys, load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import top_k_micro_accuracy, top_k_macro_accuracy, make_prediction, \
    initialize_model_and_load_from_checkpoint
from bioscanclip.epoch.inference_epoch import get_feature_and_label
import numpy as np

K_LIST = None
"""
Add classification head at the end of the image encoder for classify the seen split
"""


def inference_with_original_image_encoder_and_dna_encoder(original_model, seen_query_dataloader,
                                                          unseen_query_dataloader,
                                                          key_dataloaders, device, key_type='dna'):
    # Get query feature
    _, seen_query_image_feature, _, _, gt_label_for_seen_query = get_feature_and_label(seen_query_dataloader,
                                                                                 original_model, device,
                                                                                 multi_gpu=False)
    _, unseen_query_image_feature, _, _, gt_label_for_unseen_query = get_feature_and_label(
        unseen_query_dataloader, original_model, device, multi_gpu=False)

    # Get key feature
    all_key_feature = []
    all_key_labels = []
    for curr_key_data_loader in key_dataloaders:

        file_name_list, encoded_image_feature, encoded_dna_feature, encoded_language_feature, curr_key_labels = get_feature_and_label(curr_key_data_loader,
                                                                     original_model, device,
                                                                     multi_gpu=False)
        if key_type == 'image':
            curr_key_feature = encoded_image_feature
        elif key_type == 'dna':
            curr_key_feature = encoded_dna_feature
        else:
            raise ValueError("key_type must be either 'image' or 'dna'.")

        all_key_feature.append(curr_key_feature)
        all_key_labels = all_key_labels + curr_key_labels

    # Maybe in the future we can just use val unseen
    all_key_feature = np.concatenate(all_key_feature,
                                     axis=0)
    all_key_labels = all_key_labels

    seen_pred_labels, seen_pred_similarity = make_prediction(seen_query_image_feature, all_key_feature,
                                                             all_key_labels, with_similarity=True,
                                                             max_k=5)
    unseen_pred_labels, unseen_pred_similarity = make_prediction(unseen_query_image_feature,
                                                                 all_key_feature,
                                                                 all_key_labels, with_similarity=True,
                                                                 max_k=5)

    return seen_pred_labels, seen_pred_similarity, gt_label_for_seen_query, unseen_pred_labels, unseen_pred_similarity, gt_label_for_unseen_query


def decide_prediction_with_threshold(args, pred_labels_from_image_classifier,
                                     confidence_score_or_similarity,
                                     pred_labels_from_search, threshold):
    final_pred_labels = []
    for idx_of_record in range(len(pred_labels_from_image_classifier)):
        curr_k_pred_from_image_classifier = pred_labels_from_image_classifier[idx_of_record]
        curr_k_confidence_score = confidence_score_or_similarity[idx_of_record]
        curr_k_pred_from_search = pred_labels_from_search[idx_of_record]

        curr_final_k_pred = {}
        for kth in range(len(curr_k_confidence_score)):
            curr_confidence_score = curr_k_confidence_score[kth]
            if curr_confidence_score > threshold:

                for level in curr_k_pred_from_image_classifier.keys():
                    if level not in curr_final_k_pred.keys():
                        curr_final_k_pred[level] = []
                    curr_final_k_pred[level].append(curr_k_pred_from_image_classifier[level][kth])
            else:
                for level in curr_k_pred_from_search.keys():
                    if level not in curr_final_k_pred.keys():
                        curr_final_k_pred[level] = []
                    curr_final_k_pred[level].append(curr_k_pred_from_search[level][kth])

        final_pred_labels.append(curr_final_k_pred)
    return final_pred_labels


def get_final_pred_and_acc(args, pred_labels_from_search_with_seen_keys,
                           similarity_from_search_with_seen_keys,
                           pred_labels_from_search_with_unseen_keys,
                           gt_labels, best_threshold=None):
    final_pred_labels = decide_prediction_with_threshold(args, pred_labels_from_search_with_seen_keys,
                                                         similarity_from_search_with_seen_keys,
                                                         pred_labels_from_search_with_unseen_keys, best_threshold)
    micro_acc = top_k_micro_accuracy(
        final_pred_labels, gt_labels, k_list=args.inference_and_eval_setting.k_list)
    macro_acc, per_class_acc = top_k_macro_accuracy(
        final_pred_labels, gt_labels, k_list=args.inference_and_eval_setting.k_list)

    output_dict = {"final_pred_labels": final_pred_labels, "gt_labels": gt_labels, "best_threshold": best_threshold,
                   "micro_acc": micro_acc, "macro_acc": macro_acc, "per_class_acc": per_class_acc}

    return output_dict


def make_final_pred(args, pred_labels_from_search_with_seen_keys, similarity_from_search_with_seen_keys,
                    pred_labels_from_search_with_unseen_keys, gt_labels, threshold):
    if len(pred_labels_from_search_with_seen_keys) != len(similarity_from_search_with_seen_keys) != len(
            pred_labels_from_search_with_unseen_keys):
        print(f"pred_labels_from_search_with_seen_keys: {len(pred_labels_from_search_with_seen_keys)}")
        print(f"similarity_from_search_with_seen_keys: {len(similarity_from_search_with_seen_keys)}")
        print(f"pred_labels_from_search_with_unseen_keys: {len(pred_labels_from_search_with_unseen_keys)}")
        exit()

    final_pred_labels = decide_prediction_with_threshold(args, pred_labels_from_search_with_seen_keys,
                                                         similarity_from_search_with_seen_keys,
                                                         pred_labels_from_search_with_unseen_keys, threshold)

    return final_pred_labels, gt_labels


def harmonic_mean(l):
    s = 0
    for i in l:
        if i == 0:
            return 0
        s = s + 1 / i

    return len(l) / s


def search_threshold_with_harmonic_mean(args, all_split_data, num_intervals=1000):
    thresholds = np.linspace(0, 1, num_intervals)
    best_threshold = None
    max_score = float('-inf')
    pbar = tqdm(thresholds)
    for threshold in pbar:
        acc_list = []
        for split in all_split_data:
            final_pred_labels, gt_labels = make_final_pred(
                args,
                split['pred_labels_from_search_with_seen_keys'],
                split['pred_similarity_from_search_with_seen_keys'],
                split['pred_labels_from_search_with_unseen_keys'],
                split['gt_label'],
                threshold=threshold
            )
            micro_acc = top_k_micro_accuracy(
                final_pred_labels, gt_labels, k_list=args.inference_and_eval_setting.k_list)
            acc_list.append(micro_acc[1]['species'])
        harmonic_mean_over_acc = harmonic_mean(acc_list)
        if harmonic_mean_over_acc > max_score:
            max_score = harmonic_mean_over_acc
            best_threshold = threshold
        pbar.set_description(
            f"Curr best harmonic_mean_over_acc: {max_score} || Curr best threshold: {best_threshold} || Curr harmonic_mean_over_acc: {harmonic_mean_over_acc} || Curr threshold: {threshold}")

    return best_threshold


def get_all_unique_species_from_dataloader(dataloader):
    all_species = []

    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch['species']
    all_species = list(set(all_species))
    return all_species


def method_1_inference_and_eval_for_seen_and_unseen(args, original_model, seen_query_dataloader,
                                                    unseen_query_dataloader, seen_keys_dataloader,
                                                    val_unseen_keys_dataloader,
                                                    test_unseen_keys_dataloader, device, searched_threshold=None):
    seen_key_dataloaders = [seen_keys_dataloader]
    unseen_key_dataloaders = [val_unseen_keys_dataloader, test_unseen_keys_dataloader]

    seen_pred_labels_from_search_with_seen_keys, seen_pred_similarity_from_search_with_seen_keys, gt_label_for_seen_query, unseen_pred_labels_from_search_with_seen_keys, unseen_pred_similarity_from_search_with_seen_keys, gt_label_for_unseen_query = inference_with_original_image_encoder_and_dna_encoder(
        original_model,
        seen_query_dataloader,
        unseen_query_dataloader,
        seen_key_dataloaders,
        device=device,
        key_type='image',
    )

    print()

    seen_pred_labels_from_search_with_unseen_keys, _, _, unseen_pred_labels_from_search_with_unseen_keys, _, _ = inference_with_original_image_encoder_and_dna_encoder(
        original_model,
        seen_query_dataloader,
        unseen_query_dataloader,
        unseen_key_dataloaders,
        device=device,
        key_type='dna'
    )

    seen_pred_similarity_from_search_with_seen_keys = seen_pred_similarity_from_search_with_seen_keys.tolist()
    unseen_pred_similarity_from_search_with_seen_keys = unseen_pred_similarity_from_search_with_seen_keys.tolist()

    # all_pred_labels_from_search_with_seen_keys = seen_pred_labels_from_search_with_seen_keys + unseen_pred_labels_from_search_with_seen_keys
    # all_similarity_from_search_with_seen_keys = seen_pred_similarity_from_search_with_seen_keys + unseen_pred_similarity_from_search_with_seen_keys
    # all_pred_labels_from_search_with_unseen_keys = seen_pred_labels_from_search_with_unseen_keys + unseen_pred_labels_from_search_with_unseen_keys
    # all_gt_labels = gt_label_for_seen_query + gt_label_for_unseen_query

    seen_query_data = {'pred_labels_from_search_with_seen_keys': seen_pred_labels_from_search_with_seen_keys,
                       'pred_labels_from_search_with_unseen_keys': seen_pred_labels_from_search_with_unseen_keys,
                       'pred_similarity_from_search_with_seen_keys': seen_pred_similarity_from_search_with_seen_keys
        , "gt_label": gt_label_for_seen_query}

    unseen_query_data = {
        'pred_labels_from_search_with_seen_keys': unseen_pred_labels_from_search_with_seen_keys,
        'pred_labels_from_search_with_unseen_keys': unseen_pred_labels_from_search_with_unseen_keys,
        'pred_similarity_from_search_with_seen_keys': unseen_pred_similarity_from_search_with_seen_keys
        , "gt_label": gt_label_for_unseen_query}

    print("Searching best threshold.")

    if searched_threshold is None:
        best_threshold = search_threshold_with_harmonic_mean(args,
                                                             [seen_query_data,
                                                              unseen_query_data])
    else:
        best_threshold = searched_threshold

    seen_output_dict = get_final_pred_and_acc(args,
                                              seen_pred_labels_from_search_with_seen_keys,
                                              seen_pred_similarity_from_search_with_seen_keys,
                                              seen_pred_labels_from_search_with_unseen_keys,
                                              gt_label_for_seen_query,
                                              best_threshold=best_threshold)

    unseen_output_dict = get_final_pred_and_acc(args,
                                                unseen_pred_labels_from_search_with_seen_keys,
                                                unseen_pred_similarity_from_search_with_seen_keys,
                                                unseen_pred_labels_from_search_with_unseen_keys,
                                                gt_label_for_unseen_query,
                                                best_threshold=best_threshold)

    return seen_output_dict, unseen_output_dict


def print_acc_for_google_doc(seen_output_dict, unseen_output_dict, K_LIST=None):
    if K_LIST is None:
        K_LIST = [1, 3, 5]
    acc_dict = {'seen': seen_output_dict, 'unseen': unseen_output_dict}

    for type_of_acc in ['micro_acc', 'macro_acc']:
        for k in K_LIST:
            curr_row = ""
            acc_dict_for_harmonic_mean = {}
            for data_split in ['seen', 'unseen']:
                for level in ['order', 'family', 'genus', 'species']:
                    currue = acc_dict[data_split][type_of_acc][k][level]
                    curr_row = curr_row + " " + str(round(currue, 4))
                    if level not in acc_dict_for_harmonic_mean.keys():
                        acc_dict_for_harmonic_mean[level] = []
                    acc_dict_for_harmonic_mean[level].append(currue)
            for level in ['order', 'family', 'genus', 'species']:
                harmonic_mean_over_seen_and_unseen = harmonic_mean(acc_dict_for_harmonic_mean[level])
                curr_row = curr_row + " " + str(round(harmonic_mean_over_seen_and_unseen, 4))

            print(curr_row)

def check_for_acc_about_correct_predict_seen_or_unseen(final_pred_list, species_list):

    for k in [1, 3, 5]:
        correct = 0
        total = 0
        for record in final_pred_list:
            top_k_species = record['species']
            curr_top_k_pred = top_k_species[:k]
            for single_pred in curr_top_k_pred:
                if single_pred in species_list:
                    correct = correct + 1
                    break
            total = total + 1

        print(f"for k = {k}: {correct*1.0/total}")



@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    if args.debug_flag:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    args = copy.deepcopy(args)

    K_LIST = args.inference_and_eval_setting.k_list

    # Custom batch size
    args.model_config.batch_size = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Construct dataloader...")

    # Come up a separate dataloader function.
    train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader = load_bioscan_dataloader_with_train_seen_and_separate_keys(
        args, for_pretrain=False)

    original_model = initialize_model_and_load_from_checkpoint(args)
    original_model = original_model.to(device)
    original_model.eval()

    # Save the best_threshold.

    seen_val_output_dict, unseen_val_output_dict = method_1_inference_and_eval_for_seen_and_unseen(args, original_model,
                                                                                                   seen_val_dataloader,
                                                                                                   unseen_val_dataloader,
                                                                                                   seen_keys_dataloader,
                                                                                                   val_unseen_keys_dataloader,
                                                                                                   test_unseen_keys_dataloader,
                                                                                                   device)

    print_acc_for_google_doc(seen_val_output_dict, unseen_val_output_dict)

    seen_val_final_pred = seen_val_output_dict['final_pred_labels']
    unseen_val_final_pred = unseen_val_output_dict['final_pred_labels']

    all_unique_seen_species = get_all_unique_species_from_dataloader(seen_keys_dataloader)
    all_unique_val_seen_species = get_all_unique_species_from_dataloader(val_unseen_keys_dataloader)
    all_unique_test_seen_species = get_all_unique_species_from_dataloader(test_unseen_keys_dataloader)

    print("For seen")
    check_for_acc_about_correct_predict_seen_or_unseen(seen_val_final_pred, all_unique_seen_species)
    print("For unseen")
    check_for_acc_about_correct_predict_seen_or_unseen(unseen_val_final_pred,
                                                       all_unique_val_seen_species + all_unique_test_seen_species)

    # For test splits
    best_threshold = seen_val_output_dict['best_threshold']
    if args.inference_and_eval_setting.eval_on == "val":

        _, seen_dataloader, unseen_dataloader, _, _, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
            args)
    elif args.inference_and_eval_setting.eval_on == "test":
        _, _, _, seen_dataloader, unseen_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
            args)

    seen_output_dict, unseen_output_dict = method_1_inference_and_eval_for_seen_and_unseen(args,
                                                                                                     original_model,
                                                                                                     seen_dataloader,
                                                                                                     unseen_dataloader,
                                                                                                     seen_keys_dataloader,
                                                                                                     val_unseen_keys_dataloader,
                                                                                                     test_unseen_keys_dataloader,
                                                                                                     device,
                                                                                                     searched_threshold=best_threshold)

    print_acc_for_google_doc(seen_output_dict, unseen_output_dict)

    seen_final_pred = seen_output_dict['final_pred_labels']
    unseen_final_pred = unseen_output_dict['final_pred_labels']

    print("For seen")
    check_for_acc_about_correct_predict_seen_or_unseen(seen_final_pred, all_unique_seen_species)
    print("For unseen")
    check_for_acc_about_correct_predict_seen_or_unseen(unseen_final_pred,
                                                       all_unique_val_seen_species + all_unique_test_seen_species)


if __name__ == '__main__':
    main()
