import copy
import os
import torch.nn as nn
import torch.optim as optim
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_with_train_seen_and_separate_keys, \
    load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import top_k_micro_accuracy, top_k_macro_accuracy, make_prediction
from bioscanclip.epoch.inference_epoch import get_feature_and_label
import numpy as np
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import OneCycleLR

K_LIST = None
"""
Add classification head at the end of the image encoder for classify the seen split
"""


class ViTWIthExtraLayer(nn.Module):
    def __init__(self, vit_model, new_linear_layer):
        super(ViTWIthExtraLayer, self).__init__()
        self.vit = vit_model
        self.new_linear_layer = new_linear_layer

    def get_feature(self, x):
        return self.vit(x)

    def forward(self, x):
        outputs = self.vit(x)
        outputs = self.new_linear_layer(outputs)
        return outputs


def inference_with_fine_tuned_image_encoder(image_encoder, dataloader, species_level_label_to_index_dict,
                                            idx_to_all_labels, device):
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        all_confidence_score = []
        all_indices = []
        gt_labels = []
        pred_labels = []
        all_levels = None

        for step, batch in pbar:
            pbar.set_description("Inference with image classifier...")
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            # target = label_batch_to_species_idx(label_batch, species_level_label_to_index_dict)
            # target = target.to(device)
            image_input_batch = image_input_batch.to(device)
            output = image_encoder(image_input_batch)
            output_after_softmax = F.softmax(output, dim=-1)

            # max_values_and_indices = torch.max(output_after_softmax, dim=-1)
            # all_confidence_score = all_confidence_score + max_values_and_indices.values.tolist()

            topk_values, topk_indices = torch.topk(output_after_softmax, k=5, dim=1, largest=True, sorted=True)
            topk_values = topk_values.tolist()
            topk_indices = topk_indices.tolist()

            all_confidence_score = all_confidence_score + topk_values
            all_indices = all_indices + topk_indices

            for idx in range(len(label_batch['species'])):
                gt_labels.append({'order': label_batch['order'][idx], 'family': label_batch['family'][idx],
                                  'genus': label_batch['genus'][idx], 'species': label_batch['species'][idx]})

    for top_5_indices_for_curr_pred in all_indices:
        curr_pred_in_multi_levels = {}
        for idx in top_5_indices_for_curr_pred:
            curr_k_pred = idx_to_all_labels[idx]
            for level in curr_k_pred.keys():
                if level not in curr_pred_in_multi_levels:
                    curr_pred_in_multi_levels[level] = []
                curr_pred_in_multi_levels[level].append(curr_k_pred[level])
        pred_labels.append(curr_pred_in_multi_levels)

    return all_confidence_score, pred_labels, gt_labels


def decide_prediction_with_threshold(args, pred_labels_from_a,
                                     pred_confidence_from_a,
                                     pred_labels_from_b, threshold):
    final_pred_labels = []
    for idx_of_record in range(len(pred_labels_from_a)):
        curr_k_pred_from_a = pred_labels_from_a[idx_of_record]
        curr_k_confidence_score_from_a = pred_confidence_from_a[idx_of_record]
        curr_k_pred_from_b = pred_labels_from_b[idx_of_record]

        curr_final_k_pred = {}

        for kth in range(len(curr_k_confidence_score_from_a)):
            curr_confidence_score = curr_k_confidence_score_from_a[kth]

            if curr_confidence_score > threshold:
                for level in curr_k_pred_from_a.keys():
                    if level not in curr_final_k_pred.keys():
                        curr_final_k_pred[level] = []
                    curr_final_k_pred[level].append(curr_k_pred_from_a[level][kth])
            else:
                for level in curr_k_pred_from_b.keys():
                    if level not in curr_final_k_pred.keys():
                        curr_final_k_pred[level] = []
                    curr_final_k_pred[level].append(curr_k_pred_from_b[level][kth])

        final_pred_labels.append(curr_final_k_pred)
    return final_pred_labels


def make_final_pred(args, pred_labels_from_a, pred_confidence_from_a,
                    pred_labels_from_b, gt_labels, threshold):
    if len(pred_labels_from_a) != len(pred_confidence_from_a) != len(
            pred_labels_from_b):
        print(f"pred_labels_from_a: {len(pred_labels_from_a)}")
        print(f"pred_confidence_from_a: {len(pred_confidence_from_a)}")
        print(f"pred_labels_from_b: {len(pred_labels_from_b)}")
        exit()

    final_pred_labels = decide_prediction_with_threshold(args, pred_labels_from_a,
                                                         pred_confidence_from_a,
                                                         pred_labels_from_b, threshold)

    return final_pred_labels, gt_labels


def inference_with_original_image_encoder_and_dna_encoder(original_model, seen_dataloader, unseen_dataloader,
                                                          val_unseen_keys_dataloader, test_unseen_keys_dataloader,
                                                          device):
    # Get query feature
    _, seen_query_image_feature, _, _, gt_label_for_seen_query = get_feature_and_label(seen_dataloader,
                                                                                       original_model, device,
                                                                                       multi_gpu=False)

    _, unseen_query_image_feature, _, _, gt_label_for_unseen_query = get_feature_and_label(
        unseen_dataloader, original_model, device, multi_gpu=False)

    _, _, unseen_val_keys_dna_feature, _, unseen_val_keys_labels = get_feature_and_label(val_unseen_keys_dataloader,
                                                                                         original_model, device,
                                                                                         multi_gpu=False)

    _, _, unseen_test_keys_dna_feature, _, unseen_test_keys_labels = get_feature_and_label(test_unseen_keys_dataloader,
                                                                                           original_model, device,
                                                                                           multi_gpu=False)

    # Maybe in the future we can just use val unseen
    dna_feature_of_all_unseen_species = np.concatenate((unseen_val_keys_dna_feature, unseen_test_keys_dna_feature),
                                                       axis=0)
    key_labels_of_all_unseen_species = unseen_val_keys_labels + unseen_test_keys_labels

    seen_pred_labels = make_prediction(seen_query_image_feature, dna_feature_of_all_unseen_species,
                                       key_labels_of_all_unseen_species,
                                       max_k=5)
    unseen_pred_labels = make_prediction(unseen_query_image_feature, dna_feature_of_all_unseen_species,
                                         key_labels_of_all_unseen_species,
                                         max_k=5)

    return seen_pred_labels, unseen_pred_labels


def harmonic_mean(l):
    s = 0

    for i in l:
        if i == 0:
            return 0
        s = s + 1 / i

    return len(l) / s


def search_threshold_with_harmonic_mean(args, all_split_data, num_intervals=1000):
    thresholds = np.linspace(0, 1, num_intervals + 1)
    best_threshold = None
    max_score = float('-inf')
    pbar = tqdm(thresholds)
    for threshold in pbar:
        acc_list = []
        for split in all_split_data:
            final_pred_labels, gt_labels = make_final_pred(
                args,
                split['pred_labels_from_a'],
                split['pred_confidence_from_a'],
                split['pred_labels_from_b'],
                split['gt_labels'],
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
                f"Curr best harmonic_mean_over_acc: {harmonic_mean_over_acc} || Curr best threshold: {threshold}")

    return best_threshold


def get_final_pred_and_acc(args, pred_labels_from_a,
                           pred_confidence_from_a,
                           pred_labels_from_b,
                           gt_labels, best_threshold=None):
    final_pred_labels = decide_prediction_with_threshold(args, pred_labels_from_a,
                                                         pred_confidence_from_a,
                                                         pred_labels_from_b, best_threshold)
    micro_acc = top_k_micro_accuracy(
        final_pred_labels, gt_labels, k_list=args.inference_and_eval_setting.k_list)
    macro_acc, per_class_acc = top_k_macro_accuracy(
        final_pred_labels, gt_labels, k_list=args.inference_and_eval_setting.k_list)

    output_dict = {"final_pred_labels": final_pred_labels, "gt_labels": gt_labels, "best_threshold": best_threshold,
                   "micro_acc": micro_acc, "macro_acc": macro_acc, "per_class_acc": per_class_acc}

    return output_dict


def method_2_inference_and_eval_for_seen_and_unseen(args, image_classifier, original_model, seen_dataloader,
                                                    unseen_dataloader, val_unseen_keys_dataloader,
                                                    test_unseen_keys_dataloader, species_level_label_to_index_dict,
                                                    idx_to_all_labels, device, searched_threshold=None):
    seen_all_confidence_score_from_image_classifier, seen_pred_labels_from_image_classifier, seen_gt_labels = inference_with_fine_tuned_image_encoder(
        image_classifier,
        seen_dataloader, species_level_label_to_index_dict, idx_to_all_labels, device)
    unseen_all_confidence_score_from_image_classifier, unseen_pred_labels_from_image_classifier, unseen_gt_labels = inference_with_fine_tuned_image_encoder(
        image_classifier,
        unseen_dataloader, species_level_label_to_index_dict, idx_to_all_labels, device)

    seen_pred_labels_from_search_with_unseen_keys, unseen_pred_labels_from_search_with_unseen_keys = inference_with_original_image_encoder_and_dna_encoder(
        original_model,
        seen_dataloader,
        unseen_dataloader,
        val_unseen_keys_dataloader,
        test_unseen_keys_dataloader,
        device)
    print("For val seen, search best threshold.")

    seen_query_data = {'pred_labels_from_a': seen_pred_labels_from_image_classifier,
                       'pred_confidence_from_a': seen_all_confidence_score_from_image_classifier,
                       'pred_labels_from_b': seen_pred_labels_from_search_with_unseen_keys
        , "gt_labels": seen_gt_labels}

    unseen_query_data = {'pred_labels_from_a': unseen_pred_labels_from_image_classifier,
                         'pred_confidence_from_a': unseen_all_confidence_score_from_image_classifier,
                         'pred_labels_from_b': unseen_pred_labels_from_search_with_unseen_keys
        , "gt_labels": unseen_gt_labels}

    if searched_threshold is None:
        print("Searching best threshold.")
        best_threshold = search_threshold_with_harmonic_mean(args, [seen_query_data, unseen_query_data])
    else:
        best_threshold = searched_threshold

    seen_output_dict = get_final_pred_and_acc(args, seen_pred_labels_from_image_classifier,
                                              seen_all_confidence_score_from_image_classifier,
                                              seen_pred_labels_from_search_with_unseen_keys,
                                              seen_gt_labels,
                                              best_threshold=best_threshold)

    print("For val unseen, search best threshold.")
    unseen_output_dict = get_final_pred_and_acc(args,
                                                unseen_pred_labels_from_image_classifier,
                                                unseen_all_confidence_score_from_image_classifier,
                                                unseen_pred_labels_from_search_with_unseen_keys,
                                                unseen_gt_labels,
                                                best_threshold=best_threshold)

    return seen_output_dict, unseen_output_dict


def get_all_unique_species_from_dataloader(dataloader):
    all_species = []

    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch['species']
    all_species = list(set(all_species))
    return all_species


def load_all_seen_species_name_and_create_label_map(train_seen_dataloader):
    all_seen_species = []
    species_to_other_labels = {}

    for batch in train_seen_dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_seen_species = all_seen_species + label_batch['species']
        for curr_idx in range(len(label_batch['species'])):
            if label_batch['species'][curr_idx] not in species_to_other_labels.keys():
                species_to_other_labels[label_batch['species'][curr_idx]] = {'order': label_batch['order'][curr_idx],
                                                                             'family': label_batch['family'][curr_idx],
                                                                             'genus': label_batch['genus'][curr_idx]}

    all_seen_species = list(set(all_seen_species))
    all_seen_species.sort()

    label_to_index_dict = {}
    idx_to_all_labels = {}

    for idx, species_label in enumerate(all_seen_species):
        label_to_index_dict[species_label] = idx
        idx_to_all_labels[idx] = {'species': species_label, 'order': species_to_other_labels[species_label]['order'],
                                  'family': species_to_other_labels[species_label]['family'],
                                  'genus': species_to_other_labels[species_label]['genus']}

    return label_to_index_dict, idx_to_all_labels


def label_to_index(label, label_map):
    return label_map[label]


def label_batch_to_species_idx(label_batch, species_level_label_to_index_dict):
    species_list = label_batch['species']
    target = torch.tensor([label_to_index(species, species_level_label_to_index_dict) for species in species_list])
    return target


def fine_tuning_epoch(args, model, train_dataloader, val_seen_dataloader, val_unseen_dataloader, optimizer, criterion,
                      scheduler,
                      device, species_level_label_to_index_dict):
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    epoch_loss = []
    for step, batch in pbar:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        target = label_batch_to_species_idx(label_batch, species_level_label_to_index_dict)
        target = target.to(device)

        optimizer.zero_grad()
        image_input_batch = image_input_batch.to(device)
        output = model(image_input_batch)
        loss = criterion(output, target)
        loss.backward()
        current_lr = optimizer.param_groups[0]['lr']
        if args.activate_wandb:
            wandb.log({"loss": loss.item(), "learning_rate": current_lr})
        optimizer.step()
        scheduler.step()
        pbar.set_description(f"loss: {loss.item()}")
        epoch_loss.append(loss.item())
    epoch_loss = sum(epoch_loss) * 1.0 / len(epoch_loss)

    print("Eval on seen val.")
    seen_evaluation_result = evaluate_epoch(model, val_seen_dataloader, device, species_level_label_to_index_dict)
    print("Evaluation Result:", seen_evaluation_result)
    return epoch_loss, seen_evaluation_result


def evaluate_epoch(model, dataloader, device, species_level_label_to_index_dict, k_values=None):
    if k_values is None:
        k_values = [1, 3, 5]

    model.eval()
    all_targets = []
    all_predictions = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for idx, batch in pbar:
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            target = label_batch_to_species_idx(label_batch, species_level_label_to_index_dict)
            target = target.to(device)
            image_input_batch = image_input_batch.to(device)

            output = model(image_input_batch)
            predictions = torch.argsort(output, dim=1, descending=True)[:, :max(k_values)]

            all_targets.append(target.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    topk_accuracies = {}
    for k in k_values:
        topk_predictions = all_predictions[:, :k]
        topk_correct = np.any(topk_predictions == all_targets[:, None], axis=1)
        topk_accuracy = np.mean(topk_correct)
        topk_accuracies[f"top{k}_accuracy"] = topk_accuracy

    return topk_accuracies


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
                    curr_value = acc_dict[data_split][type_of_acc][k][level]
                    curr_row = curr_row + " " + str(round(curr_value, 4))
                    if level not in acc_dict_for_harmonic_mean.keys():
                        acc_dict_for_harmonic_mean[level] = []
                    acc_dict_for_harmonic_mean[level].append(curr_value)
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
        print(f"for k = {k}: {(correct * 1.0) / total}")


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
    args.model_config.batch_size = args.model_config.fine_tuning_set.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Construct dataloader...")

    # Come up a separate dataloader function.
    train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader = load_bioscan_dataloader_with_train_seen_and_separate_keys(
        args, for_pretrain=False)

    species_level_label_to_index_dict, idx_to_all_labels = load_all_seen_species_name_and_create_label_map(
        train_seen_dataloader)

    original_model = load_clip_model(args)
    checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')
    original_model.load_state_dict(checkpoint)
    original_model = original_model.to(device)

    image_classifier = copy.deepcopy(original_model.image_encoder)
    # image_classifier.reset_classifier(num_classes=916)
    new_linear_layer = nn.Linear(args.model_config.output_dim, 916)
    image_classifier = ViTWIthExtraLayer(image_classifier, new_linear_layer)
    image_classifier = image_classifier.to(device)

    if image_classifier is not None:
        for param in image_classifier.parameters():
            param.requires_grad = True

    last_ckpt_path = os.path.join(args.model_config.fine_tuning_set.fine_tune_model_output_dir, 'last.pth')
    os.makedirs(args.model_config.fine_tuning_set.fine_tune_model_output_dir, exist_ok=True)
    print(last_ckpt_path)

    if os.path.exists(last_ckpt_path):
        print(f"Found pre-trained model in {last_ckpt_path}")
        saved_image_classifier_param = torch.load(last_ckpt_path)  # Load the saved state dictionary
        image_classifier.load_state_dict(saved_image_classifier_param)
        args.save_ckpt = False
        for param in image_classifier.parameters():
            param.requires_grad = False
    else:
        if args.activate_wandb:
            wandb.init(project=args.model_config.wandb_project_name + "fine_tune_image_classifier",
                       name=args.model_config.model_output_name)
        pbar = tqdm(list(range(args.model_config.fine_tuning_set.epochs)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(image_classifier.parameters(), lr=0.0001)

        total_steps = len(train_seen_dataloader) * args.model_config.fine_tuning_set.epochs

        scheduler = OneCycleLR(optimizer, max_lr=0.0001, total_steps=total_steps)
        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch}")
            epoch_loss, seen_evaluation_result = fine_tuning_epoch(args, image_classifier, train_seen_dataloader,
                                                                   seen_val_dataloader, unseen_val_dataloader,
                                                                   optimizer, criterion, scheduler, device,
                                                                   species_level_label_to_index_dict)
            dict_for_wandb = {'epoch_loss': epoch_loss}
            for key in seen_evaluation_result.keys():
                dict_for_wandb[key] = seen_evaluation_result[key]
            dict_for_wandb['epoch'] = epoch
            if args.activate_wandb:
                wandb.log(dict_for_wandb,
                          commit=True)
            if args.save_ckpt:
                torch.save(image_classifier.state_dict(), last_ckpt_path)
                print(f'Last ckpt: {last_ckpt_path}')

    if args.save_ckpt:
        torch.save(image_classifier.state_dict(), last_ckpt_path)
        print(f'Last ckpt: {last_ckpt_path}')

    seen_val_output_dict, unseen_val_output_dict = method_2_inference_and_eval_for_seen_and_unseen(args,
                                                                                                   image_classifier,
                                                                                                   original_model,
                                                                                                   seen_val_dataloader,
                                                                                                   unseen_val_dataloader,
                                                                                                   val_unseen_keys_dataloader,
                                                                                                   test_unseen_keys_dataloader,
                                                                                                   species_level_label_to_index_dict,
                                                                                                   idx_to_all_labels,
                                                                                                   device)

    print_acc_for_google_doc(seen_val_output_dict, unseen_val_output_dict, K_LIST=K_LIST)

    seen_val_final_pred = seen_val_output_dict['final_pred_labels']
    unseen_val_final_pred = unseen_val_output_dict['final_pred_labels']

    all_unique_seen_species = get_all_unique_species_from_dataloader(seen_keys_dataloader)
    all_unique_val_seen_species = get_all_unique_species_from_dataloader(val_unseen_keys_dataloader)
    all_unique_test_seen_species = get_all_unique_species_from_dataloader(test_unseen_keys_dataloader)

    print_acc_for_google_doc(seen_val_output_dict, unseen_val_output_dict)

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

    seen_test_output_dict, unseen_test_output_dict = method_2_inference_and_eval_for_seen_and_unseen(args,
                                                                                                     image_classifier,
                                                                                                     original_model,
                                                                                                     seen_dataloader,
                                                                                                     unseen_dataloader,
                                                                                                     val_unseen_keys_dataloader,
                                                                                                     test_unseen_keys_dataloader,
                                                                                                     species_level_label_to_index_dict,
                                                                                                     idx_to_all_labels,
                                                                                                     device,
                                                                                                     searched_threshold=best_threshold)

    print_acc_for_google_doc(seen_test_output_dict, unseen_test_output_dict)

    seen_test_final_pred = seen_test_output_dict['final_pred_labels']
    unseen_test_final_pred = unseen_test_output_dict['final_pred_labels']

    print("For seen")
    check_for_acc_about_correct_predict_seen_or_unseen(seen_test_final_pred, all_unique_seen_species)
    print("For unseen")
    check_for_acc_about_correct_predict_seen_or_unseen(unseen_test_final_pred,
                                                       all_unique_val_seen_species + all_unique_test_seen_species)


if __name__ == '__main__':
    main()
