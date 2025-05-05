from tqdm import tqdm
import wandb
import torch
import numpy as np

# TODO: these functions either need to be updated or removed.

def label_batch_to_species_idx(label_batch, unique_species_for_seen):
    species_list = label_batch['species']
    target = torch.tensor([unique_species_for_seen.index(species) for species in species_list])
    return target

def fine_tuning_epoch(args, model, insect_train_dataloader,
                      optimizer, criterion, unique_species_for_seen, epoch, device, modality="image"):
    pbar = tqdm(enumerate(insect_train_dataloader), total=len(insect_train_dataloader))
    epoch_loss = []
    len_loader = len(insect_train_dataloader)
    for step, batch in pbar:
        processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        target = label_batch_to_species_idx(label_batch, unique_species_for_seen)
        target = target.to(device)
        optimizer.zero_grad()
        if modality == "image":
            image_input_batch = image_input_batch.to(device)
            output = model(image_input_batch)
        elif modality == "dna":
            dna_input_batch = dna_input_batch.to(device)
            output = model(dna_input_batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item()}")
        epoch_loss.append(loss.item())
        if args.activate_wandb:
            wandb.log({"loss": loss.item(), "step": step + epoch * len_loader})

    epoch_loss = sum(epoch_loss) * 1.0 / len(epoch_loss)

    return epoch_loss

def evaluate_epoch(model, dataloader, device, unique_species_for_seen, k_values=None, modality="image"):
    model.eval()
    if k_values is None:
        k_values = [1, 3, 5]

    model.eval()
    all_targets = []
    all_predictions = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for idx, batch in pbar:
            processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            target = label_batch_to_species_idx(label_batch, unique_species_for_seen)
            target = target.to(device)

            if modality == "image":
                image_input_batch = image_input_batch.to(device)
                output = model(image_input_batch)
            elif modality == "dna":
                dna_input_batch = dna_input_batch.to(device)
                output = model(dna_input_batch)
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

def fine_tuning_epoch_image_and_dna(args, image_classifier, dna_classifier, insect_train_dataloader,
                      optimizer, criterion, unique_species_for_seen, epoch, device):
    image_classifier.train()
    dna_classifier.train()
    pbar = tqdm(enumerate(insect_train_dataloader), total=len(insect_train_dataloader))
    epoch_loss = []
    len_loader = len(insect_train_dataloader)
    for step, batch in pbar:
        processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        target = label_batch_to_species_idx(label_batch, unique_species_for_seen)
        target = target.to(device)
        optimizer.zero_grad()
        image_input_batch = image_input_batch.to(device)
        dna_input_batch = dna_input_batch.to(device)
        image_output = image_classifier(image_input_batch)
        dna_output = dna_classifier(dna_input_batch)
        loss = criterion(image_output, target) + criterion(dna_output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item()}")
        epoch_loss.append(loss.item())
        if args.activate_wandb:
            wandb.log({"loss": loss.item(), "step": step + epoch * len_loader})

    epoch_loss = sum(epoch_loss) * 1.0 / len(epoch_loss)

    return epoch_loss
