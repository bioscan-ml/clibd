import copy
import datetime
import os

import hydra
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from bioscanclip.model.vit_with_mlp import ViTWIthExtraLayer
from bioscanclip.util.dataset_for_insect_dataset import load_insect_dataloader, load_insect_dataloader_trainval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def evaluate_epoch(model, dataloader, device, unique_species_for_seen, k_values=None):
    if k_values is None:
        k_values = [1, 3, 5]

    model.eval()
    all_targets = []
    all_predictions = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for idx, batch in pbar:
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            target = label_batch_to_species_idx(label_batch, unique_species_for_seen)
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


def label_batch_to_species_idx(label_batch, unique_species_for_seen):
    species_list = label_batch['species']
    target = torch.tensor([unique_species_for_seen.index(species) for species in species_list])
    return target


def fine_tuning_epoch(args, model, insect_train_dataloader,
                      optimizer, criterion, scheduler, unique_species_for_seen, epoch, device):
    pbar = tqdm(enumerate(insect_train_dataloader), total=len(insect_train_dataloader))
    epoch_loss = []
    len_loader = len(insect_train_dataloader)
    for step, batch in pbar:
        image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        target = label_batch_to_species_idx(label_batch, unique_species_for_seen)
        target = target.to(device)
        optimizer.zero_grad()
        image_input_batch = image_input_batch.to(device)
        output = model(image_input_batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(f"loss: {loss.item()}, lr: {scheduler.get_last_lr()[0]}")
        epoch_loss.append(loss.item())
        if args.activate_wandb:
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]
                      , "step": step + epoch * len_loader})

    epoch_loss = sum(epoch_loss) * 1.0 / len(epoch_loss)

    return epoch_loss


def get_unique_species_for_seen(dataloader):
    all_species = []
    pbar = tqdm(dataloader)
    for batch in pbar:
        pbar.set_description("Getting unique species labels")
        b, c, d, e, f, label_batch = batch
        all_species = all_species + label_batch['species']

    unique_species = list(set(all_species))
    return unique_species

def get_features(all_dataloader, image_classifier, device):
    pbar = tqdm(enumerate(all_dataloader), total=len(all_dataloader))
    all_feature = []
    with torch.no_grad():
        for step, batch in pbar:
            pbar.set_description(f"Getting image features")
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            image_input_batch = image_input_batch.to(device)
            feature = image_classifier.get_feature(image_input_batch)
            feature = F.normalize(feature, dim=-1).cpu().tolist()
            all_feature = all_feature + feature
    return all_feature



@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    # # Set up for debug, delete when you see it!
    args.debug_flag = False
    # # Set up for debug, delete when you see it!

    # special set up for train on INSECT dataset
    args.model_config.batch_size = 300
    args.model_config.epochs = 100
    args.model_config.evaluation_period = 15

    if args.debug_flag:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    args = copy.deepcopy(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Construct dataloader...")
    insect_trainval_dataloader = load_insect_dataloader_trainval(args, num_workers=8, shuffle_for_train_seen_key=False)

    insect_train_dataloader, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader, insect_test_unseen_dataloader = load_insect_dataloader(
        args, world_size=None, rank=None, shuffle_for_train_seen_key=True)

    unique_species_for_seen = get_unique_species_for_seen(insect_trainval_dataloader)

    print("Initialize model...")
    pre_trained_timm_vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    new_linear_layer = nn.Linear(args.model_config.output_dim, len(unique_species_for_seen))
    image_classifier = ViTWIthExtraLayer(pre_trained_timm_vit, new_linear_layer)
    image_classifier = image_classifier.to(device)
    for param in image_classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(image_classifier.parameters(), lr=0.00015)

    # Calculate total number of steps (iterations)
    total_steps = args.model_config.epochs * len(insect_trainval_dataloader)

    # Define a linear learning rate scheduler based on the step
    lambda_lr = lambda step: 1 - step / total_steps
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    if args.activate_wandb:
        wandb.init(project="Fine-tune ViT-B on INSECT dataset",
                   name="Fine-tune ViT-B on INSECT dataset")

    folder_path = os.path.join(args.project_root_path, args.model_output_dir,
                               "Fine_tune_ViT-B_on_INSECT_dataset", formatted_datetime)
    folder_to_save_embed = os.path.join(args.project_root_path, "embedding_from_vit_fine_tuned_on_insect",
                                        formatted_datetime)
    os.makedirs(folder_path, exist_ok=True)
    last_ckpt_path = os.path.join(folder_path, 'last.ckpt')
    OmegaConf.save(args, os.path.join(folder_path, 'config.yaml'))

    all_dataloader = load_insect_dataloader(
        args, world_size=None, rank=None, load_all_in_one=True)

    print("training...")
    pbar = tqdm(range(args.model_config.epochs))
    for epoch in pbar:
        pbar.set_description(f"Epoch: {epoch}")
        epoch_loss = fine_tuning_epoch(args, image_classifier, insect_trainval_dataloader,
                                       optimizer, criterion, scheduler, unique_species_for_seen, epoch, device)

        if epoch % args.model_config.evaluation_period == 0 or epoch - 1 == args.model_config.epochs:
            print("Eval:")
            seen_evaluation_result = evaluate_epoch(image_classifier, insect_test_seen_dataloader, device,
                                                    unique_species_for_seen)
            print("Evaluation Result:", seen_evaluation_result)

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
                # save_image_embedding to “image_embedding_from_bioscan_clip.csv”
                os.makedirs(folder_to_save_embed, exist_ok=True)
                image_embed_path = os.path.join(folder_to_save_embed,
                                                "image_embedding_from_vit_fine_tuned_on_insect.csv")
                image_feature = get_features(all_dataloader, image_classifier, device)
                image_feature = np.array(image_feature, dtype=np.float32)
                image_feature = image_feature.T
                print(image_feature.shape)
                np.savetxt(image_embed_path, image_feature, delimiter=",")
    if args.save_ckpt:
        torch.save(image_classifier.state_dict(), last_ckpt_path)
        print(f'Last ckpt: {last_ckpt_path}')
        # save_image_embedding to “image_embedding_from_bioscan_clip.csv”
        folder_to_save_embed = os.path.join(args.project_root_path, "embedding_from_vit_fine_tuned_on_insect",
                                            formatted_datetime)
        os.makedirs(folder_to_save_embed, exist_ok=True)
        image_embed_path = os.path.join(folder_to_save_embed, "image_embedding_from_vit_fine_tuned_on_insect.csv")
        image_feature = get_features(all_dataloader, image_classifier, device)
        image_feature = np.array(image_feature, dtype=np.float32)
        image_feature = image_feature.T
        print(image_feature.shape)
        np.savetxt(image_embed_path, image_feature, delimiter=",")


if __name__ == '__main__':
    main()
