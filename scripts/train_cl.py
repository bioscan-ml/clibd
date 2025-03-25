import copy
import datetime
import json
import os
import hydra
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from bioscanclip.epoch.train_epoch import train_epoch
from inference_and_eval import get_features_and_label, inference_and_print_result
from bioscanclip.model.loss_func import ContrastiveLoss, ClipLoss
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.util import set_seed
from bioscanclip.util.dataset import load_dataloader, load_insect_dataloader
from bioscanclip.util.util import scale_learning_rate


def print_when_rank_zero(message, rank=0):
    if rank is None or rank == 0:
        print(message)


def save_prediction(pred_list, gt_list, json_path):
    data = {
        "gt_labels": gt_list,
        "pred_labels": pred_list
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)


def ddp_setup(rank: int, world_size: int, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def construct_key_dict(list_of_dict):
    key_dict = {}

    for curr_dict in list_of_dict:
        for a_kind_of_feature_or_label in curr_dict.keys():
            if a_kind_of_feature_or_label == "all_key_features" or a_kind_of_feature_or_label == "all_key_features_label":
                key_dict[a_kind_of_feature_or_label] = None
                continue

            if a_kind_of_feature_or_label not in key_dict.keys():
                key_dict[a_kind_of_feature_or_label] = curr_dict[a_kind_of_feature_or_label]
            else:
                if isinstance(curr_dict[a_kind_of_feature_or_label], list):
                    key_dict[a_kind_of_feature_or_label] = key_dict[a_kind_of_feature_or_label] + curr_dict[
                        a_kind_of_feature_or_label]
                else:
                    key_dict[a_kind_of_feature_or_label] = np.concatenate(
                        (key_dict[a_kind_of_feature_or_label], curr_dict[a_kind_of_feature_or_label]), axis=0)

    return key_dict


def eval_phase(model, device, all_keys_dataloader, seen_val_dataloader, unseen_val_dataloader, k_list, args,
               species_to_drop=None, rank=None, for_open_clip=False):
    keys_dict = get_features_and_label(
        all_keys_dataloader, model, device, for_key_set=True, for_open_clip=for_open_clip)

    seen_val_dict = get_features_and_label(
        seen_val_dataloader, model, device, for_open_clip=for_open_clip)

    unseen_val_dict = get_features_and_label(
        unseen_val_dataloader, model, device, for_open_clip=for_open_clip)

    acc_dict, _, pred_dict = inference_and_print_result(keys_dict, seen_val_dict, unseen_val_dict, args=args,
                                                        small_species_list=None, k_list=k_list)
    return acc_dict, pred_dict


def eval_phase_for_insect(model, device, insect_train_dataloader_for_key, insect_val_dataloader,
                          insect_test_seen_dataloader, insect_test_unseen_dataloader, k_list, args,
                          species_to_drop=None):
    insect_train_dict = get_features_and_label(
        insect_train_dataloader_for_key, model, device)
    insect_val_dict = get_features_and_label(
        insect_val_dataloader, model, device)
    insect_test_seen_dict = get_features_and_label(
        insect_test_seen_dataloader, model, device)
    insect_test_unseen_dict = get_features_and_label(
        insect_test_unseen_dataloader, model, device)

    keys_dict = construct_key_dict([insect_train_dict, insect_val_dict, insect_test_seen_dict, insect_test_unseen_dict])

    acc_dict, _, pred_dict = inference_and_print_result(keys_dict, insect_test_seen_dict, insect_test_unseen_dict,
                                                        args=args,
                                                        small_species_list=None, k_list=k_list)

    return acc_dict, pred_dict


def convert_acc_dict_to_wandb_dict(acc_dict):
    dict_for_wandb = {}
    acc_dict = acc_dict['encoded_image_feature']['encoded_image_feature']

    # For now, we just put query: image and key:image acc to wandb
    for split, split_dict in acc_dict.items():
        for type_of_acc, type_of_acc_dict in split_dict.items():
            for k, k_dict in type_of_acc_dict.items():
                for level, curr_acc in type_of_acc_dict.items():
                    dict_for_wandb[f"Image to Image_{split} {type_of_acc} top-{k} {level} level"] = curr_acc

    return dict_for_wandb


def compute_overall_acc(acc_dict):
    overall_acc_list = []

    for query_type in acc_dict.keys():
        for key_type in acc_dict[query_type].keys():
            for seen_or_unseen in acc_dict[query_type][key_type].keys():
                for micro_and_macro in acc_dict[query_type][key_type][seen_or_unseen].keys():
                    for k in acc_dict[query_type][key_type][seen_or_unseen][micro_and_macro].keys():
                        if k == 3 or k == 5:
                            # Only care about top 1 accuracy
                            continue
                        for level in acc_dict[query_type][key_type][seen_or_unseen][micro_and_macro][k].keys():
                            try:
                                curr_acc = acc_dict[query_type][key_type][seen_or_unseen][micro_and_macro][k][level]
                                overall_acc_list.append(curr_acc)
                            except:
                                pass
    overall_acc = sum(overall_acc_list) / len(overall_acc_list)
    return overall_acc

def main_process(rank: int, world_size: int, args):
    stop_flag = torch.tensor([0], device=rank)

    if args.debug_flag or rank != 0:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    args = copy.deepcopy(args)

    with open_dict(args.model_config):
        if not hasattr(args.model_config, "for_open_clip"):
            args.model_config.for_open_clip = False

    ddp_setup(rank, world_size, str(args.model_config.port))

    # Load DATALOADER
    if rank == 0:
        print("Construct dataloader...")
    if hasattr(args.model_config, 'dataset') and args.model_config.dataset == "INSECT":
        insect_train_dataloader, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader, insect_test_unseen_dataloader = load_insect_dataloader(
            args, world_size=world_size, rank=rank)
        pre_train_dataloader = insect_train_dataloader
    else:
        pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(
            args, world_size=world_size, rank=rank)

    # optional configs
    for_open_clip = False
    if hasattr(args.model_config, 'for_open_clip') and args.model_config.for_open_clip:
        for_open_clip = True

    all_gather = False
    if hasattr(args.model_config, 'all_gather') and args.model_config.all_gather:
        all_gather = True

    fix_temperature = None
    if hasattr(args.model_config, 'fix_temperature') and args.model_config.fix_temperature:
        fix_temperature = args.model_config.fix_temperature

    enable_amp = False
    if hasattr(args.model_config, 'amp') and args.model_config.amp:
        enable_amp = True

    eval_skip_epoch = -1
    if hasattr(args.model_config, 'eval_skip_epoch') and args.model_config.eval_skip_epoch:
        eval_skip_epoch = args.model_config.eval_skip_epoch

    scaler = GradScaler(enabled=enable_amp)

    # Load MODEL
    if rank == 0:
        print("Initialize model...")
    model = load_clip_model(args, device=rank)
    if hasattr(args.model_config, 'pretrained_ckpt_path'):
        checkpoint = torch.load(args.model_config.pretrained_ckpt_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    total_steps = len(pre_train_dataloader) * args.model_config.epochs

    lr = 0.001

    if hasattr(args.model_config, 'lr_config') and hasattr(args.model_config.lr_config, 'lr'):
        lr = args.model_config.lr_config.lr
    """
    We do not scale the learning rate then write in the config. 
    Instead, we get the general learning rate from the config 
    and scale it by the number of GPUs and batch size.
    """

    lr = scale_learning_rate(lr=lr, batch_size=args.model_config.batch_size, world_size=world_size)


    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = None
    if hasattr(args.model_config, 'lr_scheduler'):
        if args.model_config.lr_scheduler == 'one_cycle':
            max_lr = 0.001
            if hasattr(args.model_config.lr_config, 'max_lr'):
                max_lr = args.model_config.lr_config.max_lr
            max_lr = scale_learning_rate(lr=max_lr, batch_size=args.model_config.batch_size, world_size=world_size)
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=False,
            )
        elif args.model_config.lr_scheduler == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif args.model_config.lr_scheduler == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif args.model_config.lr_scheduler == 'cosine':
            min_lr = 1e-9
            if hasattr(args.model_config.lr_config, 'min_lr'):
                min_lr = args.model_config.lr_config.min_lr
            min_lr = scale_learning_rate(lr=min_lr, batch_size=args.model_config.batch_size, world_size=world_size)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    bind_to = None
    if hasattr(args.model_config, 'bind_to'):
        bind_to = args.model_config.bind_to

    no_image_text_loss = False
    if hasattr(args.model_config, 'no_image_text_loss'):
        no_image_text_loss = args.model_config.no_image_text_loss

    if all_gather:
        criterion = ClipLoss(local_loss=args.model_config.loss_setup.local_loss,
                             gather_with_grad=args.model_config.loss_setup.gather_with_grad, rank=rank,
                             world_size=world_size, use_horovod=args.model_config.loss_setup.use_horovod,
                             criterion=nn.CrossEntropyLoss(), bind_to=bind_to, no_image_text_loss=no_image_text_loss)
    else:
        criterion = ContrastiveLoss(criterion=nn.CrossEntropyLoss(), logit_scale=1 / 0.07)

    if args.activate_wandb and rank == 0:
        wandb.init(project=args.model_config.wandb_project_name, name=args.model_config.model_output_name)
    if rank == 0:
        print("training...")

    k_list = [1, 3, 5]

    best_epoch = None
    best_overall_acc = None
    folder_path = os.path.join(args.project_root_path, args.model_output_dir,
                               args.model_config.model_output_name, formatted_datetime)
    os.makedirs(folder_path, exist_ok=True)

    OmegaConf.save(args, os.path.join(folder_path, 'config.yaml'))

    for epoch in range(args.model_config.epochs):
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            print(f"Process {rank} stopping at epoch {epoch} due to early stopping")
            break
        train_epoch(args.activate_wandb, args.model_config.epochs, epoch,
                                                                 pre_train_dataloader, model, optimizer,
                                                                 criterion, rank, rank=rank, scheduler=scheduler,
                                                                 for_open_clip=for_open_clip,
                                                                 fix_temperature=fix_temperature, scaler=scaler, enable_autocast=enable_amp)

        if (epoch % args.model_config.evaluation_period == 0 or epoch == args.model_config.epochs - 1) and rank == 0 and epoch > eval_skip_epoch:
            original_model = model.module if hasattr(model, 'module') else model
            if args.save_ckpt:
                print(f'Saving ckpt at epoch {epoch}')
                last_ckpt_path = os.path.join(folder_path, f'last.pth')
                torch.save(original_model.state_dict(), last_ckpt_path)
                print(f'Last ckpt: {last_ckpt_path}')

            if hasattr(args.model_config, 'dataset') and args.model_config.dataset == "INSECT":
                acc_dict, pred_dict = eval_phase(original_model, rank, insect_train_dataloader_for_key,
                                                 insect_test_seen_dataloader, insect_test_unseen_dataloader, k_list,
                                                 args=args, for_open_clip=for_open_clip, rank=rank)
            else:
                acc_dict, pred_dict = eval_phase(original_model, rank, all_keys_dataloader, seen_val_dataloader,
                                                 unseen_val_dataloader, k_list, rank=rank, args=args,
                                                 for_open_clip=for_open_clip)

            dict_for_wandb = convert_acc_dict_to_wandb_dict(acc_dict)
            dict_for_wandb['epoch'] = epoch

            overall_acc = compute_overall_acc(acc_dict)

            if best_overall_acc is None or best_overall_acc < overall_acc:
                best_epoch = epoch
                best_overall_acc = overall_acc
                if args.save_ckpt:
                    best_ckpt_path = os.path.join(folder_path, f'best.pth')

                    torch.save(original_model.state_dict(), best_ckpt_path)
                    print(f'Best ckpt: {best_ckpt_path}')
            else:
                if args.enable_early_stopping:
                    stop_flag[0] = 1
            dict_for_wandb["overall_acc"] = overall_acc
            dict_for_wandb["best_epoch"] = best_epoch
            if args.activate_wandb and rank == 0:
                wandb.log(dict_for_wandb,
                          commit=True)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            print(f"Process {rank} stopping at epoch {epoch} due to early stopping")
            break

@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    torch.cuda.empty_cache()
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')

    default_seed = args.default_seed
    if hasattr(args.model_config, 'default_seed'):
        default_seed = args.model_config.default_seed

    if hasattr(args.model_config, 'random_seed') and args.model_config.random_seed:
        seed = set_seed(); string = "random seed"
    else:
        seed = set_seed(seed=int(default_seed)); string = "default seed"
    print("The module is run with %s: %d" % (string, seed))

    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
