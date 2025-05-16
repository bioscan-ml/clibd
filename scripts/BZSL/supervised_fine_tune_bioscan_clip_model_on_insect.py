import copy
import datetime
import os

import hydra
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bioscanclip.epoch.fine_tuning_epoch import fine_tuning_epoch_image_and_dna, evaluate_epoch
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_insect_dataloader, load_insect_dataloader_trainval
from bioscanclip.util.util import EncoderWithExtraLayer, get_unique_species_for_seen
from bioscanclip.util.util import get_features_and_label


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args.activate_wandb = False
    if args.debug_flag:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")

    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    args = copy.deepcopy(args)

    K_LIST = args.inference_and_eval_setting.k_list

    # Custom batch size
    args.model_config.batch_size = args.model_config.fine_tuning_set.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Construct dataloader...")
    insect_trainval_dataloader = load_insect_dataloader_trainval(args, num_workers=8, shuffle_for_train_seen_key=False)

    insect_train_dataloader, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader, insect_test_unseen_dataloader = load_insect_dataloader(
        args, world_size=None, rank=None, shuffle_for_train_seen_key=True)

    unique_species_for_seen = get_unique_species_for_seen(insect_trainval_dataloader)

    print("Load model...")
    original_model = load_clip_model(args)
    if hasattr(args.model_config, 'ckpt_trained_with_insect_image_dna_text_path') and os.path.exists(os.path.join(args.model_config.ckpt_trained_with_insect_image_dna_text_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.model_config.ckpt_trained_with_insect_image_dna_text_path, "best.pth"), map_location='cuda:0')
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location='cuda:0')
    original_model.load_state_dict(checkpoint)
    original_model = original_model.to(device)

    # image_classifier = copy.deepcopy(original_model.image_encoder)
    image_classifier = original_model.image_encoder
    new_image_linear_layer = nn.Linear(args.model_config.output_dim, len(unique_species_for_seen))
    image_classifier = EncoderWithExtraLayer(image_classifier, new_image_linear_layer)
    image_classifier = image_classifier.to(device)

    # dna_classifier = copy.deepcopy(original_model.dna_encoder)
    dna_classifier = original_model.dna_encoder
    new_dna_linear_layer = nn.Linear(args.model_config.output_dim, len(unique_species_for_seen))
    dna_classifier = EncoderWithExtraLayer(dna_classifier, new_dna_linear_layer)
    dna_classifier = dna_classifier.to(device)

    for param in image_classifier.parameters():
        param.requires_grad = True

    for param in dna_classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    combined_params = list(image_classifier.parameters()) + list(dna_classifier.parameters())
    optimizer = optim.AdamW(combined_params, lr=0.001)

    if args.activate_wandb:
        wandb.init(project="Supervised fine-tune BSC on INSECT dataset",
                   name="Supervised fine-tune  BSC on INSECT dataset")

    folder_path = os.path.join(args.project_root_path, args.model_output_dir,
                               "supervised_fine_tune_bioscan_clip_model_on_insect", formatted_datetime)

    os.makedirs(folder_path, exist_ok=True)

    image_last_ckpt_path = os.path.join(folder_path, 'image_last.ckpt')
    dna_last_ckpt_path = os.path.join(folder_path, 'dna_last.ckpt')
    OmegaConf.save(args, os.path.join(folder_path, 'config.yaml'))

    all_dataloader = load_insect_dataloader(
        args, world_size=None, rank=None, load_all_in_one=True)

    print("training...")
    pbar = tqdm(range(args.model_config.fine_tuning_set.epochs))

    for epoch in pbar:
        pbar.set_description(f"Epoch: {epoch}")
        # image_loss = fine_tuning_epoch(args, image_classifier, insect_trainval_dataloader,
        #                                optimizer, criterion, unique_species_for_seen, epoch, device, modality="image")
        # dna_loss = fine_tuning_epoch(args, dna_classifier, insect_trainval_dataloader,
        #                              optimizer, criterion, unique_species_for_seen, epoch, device, modality="dna")
        epoch_loss = fine_tuning_epoch_image_and_dna(args, image_classifier, dna_classifier, insect_trainval_dataloader,
                                                     optimizer, criterion, unique_species_for_seen, epoch, device)
        if epoch % args.model_config.evaluation_period == 0 or epoch - 1 == args.model_config.epochs:
            print("Eval:")
            image_seen_evaluation_result = evaluate_epoch(image_classifier, insect_test_seen_dataloader, device,
                                                    unique_species_for_seen, modality="image")
            dna_seen_evaluation_result = evaluate_epoch(dna_classifier, insect_test_seen_dataloader, device,
                                                    unique_species_for_seen, modality="dna")
            print("Image Evaluation Result:", image_seen_evaluation_result)
            print("DNA Evaluation Result:", dna_seen_evaluation_result)
            dict_for_wandb = {'epoch_loss': epoch_loss}
            for key in image_seen_evaluation_result.keys():
                dict_for_wandb["image_" + key] = image_seen_evaluation_result[key]
            for key in dna_seen_evaluation_result.keys():
                dict_for_wandb["dna_" + key] = dna_seen_evaluation_result
            dict_for_wandb['epoch'] = epoch
            if args.activate_wandb:
                wandb.log(dict_for_wandb,
                          commit=True)

            if args.save_ckpt:
                torch.save(image_classifier.state_dict(), image_last_ckpt_path)
                torch.save(dna_classifier.state_dict(), dna_last_ckpt_path)
                print(f'Last image ckpt: {image_last_ckpt_path}')
                print(f'Last dna ckpt: {dna_last_ckpt_path}')
                # save_image_embedding to “image_embedding_from_bioscan_clip.csv”
                folder_to_save_embed = os.path.join(args.project_root_path, "embedding_from_bsc_fine_tuned_on_insect",
                                                    formatted_datetime)
                os.makedirs(folder_to_save_embed, exist_ok=True)
                dna_embed_path = os.path.join(folder_to_save_embed, "dna_embedding_from_bioscan_clip.csv")
                image_embed_path = os.path.join(folder_to_save_embed, "image_embedding_from_bioscan_clip.csv")

                att_splits_dict = sio.loadmat(args.insect_data.path_to_res_101_mat)
                labels = att_splits_dict["labels"].squeeze() - 1
                all_label = np.unique(labels)
                all_label.sort()
                original_model.image_encoder = image_classifier.encoder
                original_model.dna_encoder = dna_classifier.encoder

                dict_for_feature = get_features_and_label(all_dataloader, original_model, device, for_key_set=False)
                dna_feature = dict_for_feature["encoded_dna_feature"]

                dict_emb = {}

                pbar = tqdm(enumerate(labels), total=len(labels))
                for i, label in pbar:
                    pbar.set_description("Extracting features: ")
                    curr_feature = dna_feature[i]

                    if str(label) not in dict_emb.keys():
                        dict_emb[str(label)] = []
                    dict_emb[str(label)].append(curr_feature)
                class_embed = []
                for i in all_label:
                    class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
                class_embed = np.array(class_embed, dtype=object)
                class_embed = class_embed.T.squeeze()
                print(class_embed.shape)
                np.savetxt(dna_embed_path, class_embed,
                           delimiter=",")

                # save_image_embedding to “image_embedding_from_bioscan_clip.csv”
                image_feature = dict_for_feature["encoded_image_feature"]
                image_feature = image_feature.astype(np.float32)
                image_feature = image_feature.T
                print(image_feature.shape)
                np.savetxt(image_embed_path, image_feature, delimiter=",")
                print(image_embed_path)






if __name__ == '__main__':
    main()