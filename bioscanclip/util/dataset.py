import io
import os
from typing import Any
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from bioscanclip.model.dna_encoder import get_sequence_pipeline
from torch.utils.data.distributed import DistributedSampler
import json
import time
from transformers import AutoTokenizer
from bioscanclip.model.language_encoder import load_pre_trained_bert
import open_clip

from bioscanclip.util.util import load_kmer_tokenizer, TensorResizeLongEdge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_label_ids(input_labels):
    label_to_id = {}

    for label in input_labels:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

    label_ids = np.array([label_to_id[label] for label in input_labels])
    return label_ids, label_to_id


def tokenize_dna_sequence(pipeline, dna_input):
    list_of_output = []
    for i in dna_input:
        list_of_output.append(pipeline(i))
    return list_of_output


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0, shuffle=False):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=sampler, drop_last=True
    )

    return dataloader


def get_array_of_label_dicts(hdf5_inputs_path, split):
    hdf5_split_group = h5py.File(hdf5_inputs_path, "r", libver="latest")[split]
    np_order = np.array([item.decode("utf-8") for item in hdf5_split_group["order"][:]])
    np_family = np.array([item.decode("utf-8") for item in hdf5_split_group["family"][:]])
    np_genus = np.array([item.decode("utf-8") for item in hdf5_split_group["genus"][:]])
    np_species = np.array([item.decode("utf-8") for item in hdf5_split_group["species"][:]])
    array_of_dicts = np.array(
        [
            {"order": o, "family": f, "genus": g, "species": s}
            for o, f, g, s in zip(np_order, np_family, np_genus, np_species)
        ],
        dtype=object,
    )
    return array_of_dicts


def load_split_df_and_merge_with_df(args, df):
    split_df = pd.read_csv(args.bioscan_data.path_to_the_split, sep="\t")
    split_df = split_df[["sampleid", "split"]]
    df = pd.merge(df, split_df, on="sampleid", how="inner")

    return df


def get_bin_from_tsv(split, hef5_path, tsv_path):
    with h5py.File(hef5_path, "r") as h5file:
        sample_id_list = [item.decode('utf-8') for item in h5file[split]['sampleid']]
    df = pd.read_csv(tsv_path, sep='\t')
    filtered_df = df[df['sampleid'].isin(sample_id_list)]
    uri_list = filtered_df['uri'].tolist()
    return uri_list


def convert_uri_to_index_list(uri_list):
    string_to_int = {}
    next_int = 0
    integers = []
    for s in uri_list:
        if s not in string_to_int:
            string_to_int[s] = next_int
            next_int += 1
        integers.append(string_to_int[s])

    return integers


class Dataset_for_CL(Dataset):
    def __init__(
            self,
            args,
            split,
            length=None,
            image_type="image",
            dna_type="sequence",
            dna_tokens=None,
            return_language=False,
            labels=None,
            for_training=False,
            for_open_clip=False,
    ):
        if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
            if hasattr(args.model_config, "train_with_small_subset") and args.model_config.train_with_small_subset:
                self.hdf5_inputs_path = args.bioscan_5m_data.path_to_smaller_hdf5_data
            else:
                self.hdf5_inputs_path = args.bioscan_5m_data.path_to_hdf5_data
            self.dataset = "bioscan_5m"
        else:
            self.hdf5_inputs_path = args.bioscan_data.path_to_hdf5_data
            self.dataset = "bioscan_1m"
        self.split = split
        self.image_input_type = image_type
        self.dna_inout_type = dna_type
        if dna_tokens is not None:
            self.dna_tokens = torch.tensor(dna_tokens)
        else:
            self.dna_tokens = None
        self.length = length
        self.return_language = return_language
        self.for_training = for_training
        self.for_open_clip = for_open_clip
        self.pre_train_with_small_set = False
        if hasattr(args.model_config, "train_with_small_subset"):
            self.pre_train_with_small_set = args.model_config.train_with_small_subset
        language_model_name = "prajjwal1/bert-small"
        self.tokenizer, _ = load_pre_trained_bert(language_model_name)

        if self.for_open_clip:
            # self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.tokenizer = None
        else:
            if hasattr(args.model_config, "language"):
                language_model_name = "prajjwal1/bert-small"
                if hasattr(args.model_config.language, "pre_train_model"):
                    language_model_name = args.model_config.language.pre_train_model
                self.tokenizer, _ = load_pre_trained_bert(language_model_name)

        list_of_label_dict = get_array_of_label_dicts(self.hdf5_inputs_path, split)
        self.list_of_label_string = []
        for i in list_of_label_dict:
            self.list_of_label_string.append(i['order'] + ' ' + i['family'] + ' ' + i['genus'] + ' ' + i['species'])

        if self.for_training:
            if hasattr(args.model_config,
                       "bin_for_positive_and_negative_pairs") and args.model_config.bin_for_positive_and_negative_pairs:
                self.labels = list(get_bin_from_tsv(split, self.hdf5_inputs_path, args.bioscan_data.path_to_tsv_data))
                self.labels = np.array(convert_uri_to_index_list(self.labels))
            elif labels is None:
                self.labels = np.array(list(range(length)))
            else:
                self.labels = labels
        else:
            self.labels = get_array_of_label_dicts(self.hdf5_inputs_path, split)

        if self.image_input_type == "image":
            if self.for_training:
                if self.for_open_clip:
                    self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),

                            transforms.Resize(size=256, antialias=True),
                            transforms.RandomResizedCrop(224, antialias=True),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomRotation(degrees=(-45, 45)),
                            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        ]
                    )
                else:
                    self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize(size=256, antialias=True),
                            transforms.RandomResizedCrop(224, antialias=True),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomRotation(degrees=(-45, 45)),
                            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        ]
                    )
            else:
                if self.for_open_clip:
                    self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize(size=256, antialias=True),
                            transforms.CenterCrop(224),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711)),
                        ]
                    )
                else:
                    self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize(size=256, antialias=True),
                            transforms.CenterCrop(224),
                        ]
                    )
        elif self.image_input_type == "feature":
            self.transform = None
        else:
            raise TypeError(
                f"Image input type can not be {self.image_input_type}, it must be either 'image' or 'feature'. Please check the config file."
            )

        self.dne_tokenizer = None
        if self.dna_inout_type == "sequence":
            self.dne_tokenizer = load_kmer_tokenizer(args, k=4)

        if self.dna_inout_type not in ["sequence", "feature"]:
            raise TypeError(
                f"DNA input type can not be {self.dna_inout_type}, it must be either 'sequence' or 'feature'. Please check the config file."
            )

    def __len__(self):
        return self.length

    def _open_hdf5(self):
        self.hdf5_split_group = h5py.File(self.hdf5_inputs_path, "r", libver="latest")[self.split]

    def load_image(self, idx):
        image_enc_padded = self.hdf5_split_group["image"][idx].astype(np.uint8)
        enc_length = self.hdf5_split_group["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc))
        if self.transform is not None:
            curr_image = self.transform(curr_image)
        return curr_image

    def __getitem__(self, idx):
        if not hasattr(self, "hdf5_split_group"):
            self._open_hdf5()
        if self.image_input_type == "image":
            curr_image_input = self.load_image(idx)
        else:
            # Check if image feature are loaded correctly
            curr_image_input = self.hdf5_split_group["image_features"][idx].astype(np.float32)

        if self.dna_inout_type == "sequence":
            if self.dna_tokens is None:
                curr_dna_input = self.hdf5_split_group["barcode"][idx].decode("utf-8")
            else:
                dna_sequences = self.hdf5_split_group["barcode"][idx].decode("utf-8")
                processed_barcode, att_mask = self.tokenizer(dna_sequences, offset=0)
                curr_dna_input = [processed_barcode, att_mask]
                import pdb; pdb.set_trace()
        else:
            curr_dna_input = self.hdf5_split_group["dna_features"][idx].astype(np.float32)

        if self.dataset == "bioscan_5m":
            curr_processid = self.hdf5_split_group["processid"][idx].decode("utf-8")
        else:
            curr_processid = self.hdf5_split_group["image_file"][idx].decode("utf-8")

        if self.for_open_clip:
            language_input = self.list_of_label_string[idx]
            language_input_ids = language_input
            language_token_type_ids = torch.zeros(1, )
            language_attention_mask = torch.zeros(1, )
        else:

            language_tokens = self.tokenizer([self.list_of_label_string[idx]], padding="max_length", max_length=20,
                                             truncation=True)
            language_input_ids = language_tokens['input_ids']
            language_token_type_ids = language_tokens['token_type_ids']
            language_attention_mask = language_tokens['attention_mask']

            language_input_ids = torch.tensor(language_input_ids[0])
            language_token_type_ids = torch.tensor(language_token_type_ids[0])
            language_attention_mask = torch.tensor(language_attention_mask[0])

            # language_input_ids = self.hdf5_split_group["language_tokens_input_ids"][idx]
            # language_token_type_ids = self.hdf5_split_group["language_tokens_token_type_ids"][idx]
            # language_attention_mask = self.hdf5_split_group["language_tokens_attention_mask"][idx]

        return (
            curr_processid,
            curr_image_input,
            curr_dna_input,
            language_input_ids,
            language_token_type_ids,
            language_attention_mask,
            self.labels[idx],
        )


class GaussianBlur(object):
    def __init__(self, kernel_size):
        radius = kernel_size // 2
        self.blur = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    def __call__(self, img):
        return self.blur(img)


def get_simclr_pipeline_transform(size, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    kernel_size = int(0.1 * size)
    if kernel_size % 2 == 0:  # Check if the number is even
        kernel_size -= 1  # Subtract 1 if it's even
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=kernel_size),
                                          transforms.ToTensor()])
    return data_transforms


class DatasetForSimCLRStyleTraining(Dataset):
    def __init__(self, args, split, length, transform=None):
        # Initialize paths and dataset type based on configuration
        if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
            self.hdf5_inputs_path = (args.bioscan_5m_data.path_to_smaller_hdf5_data
                                     if hasattr(args.model_config,
                                                "train_with_small_subset") and args.model_config.train_with_small_subset
                                     else args.bioscan_5m_data.path_to_hdf5_data)
            self.dataset = "bioscan_5m"
        else:
            self.hdf5_inputs_path = args.bioscan_data.path_to_hdf5_data
            self.dataset = "bioscan_1m"

        self.split = split
        self.transform = get_simclr_pipeline_transform(size=224)
        self.length = length

    def __len__(self):
        return self.length

    def _open_hdf5(self):
        # Open the HDF5 file and access the split group
        self.hdf5_split_group = h5py.File(self.hdf5_inputs_path, "r", libver="latest")[self.split]

    def load_image(self, idx):
        # Load an image from HDF5 file
        image_enc_padded = self.hdf5_split_group["image"][idx].astype(np.uint8)
        enc_length = self.hdf5_split_group["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc))
        return curr_image

    def __getitem__(self, idx):
        if not hasattr(self, 'hdf5_split_group'):
            self._open_hdf5()
        # Fetch the image and apply transformations to create a positive pair
        image = self.load_image(idx)
        transformed_image = self.transform(image)
        transformed_image2 = self.transform(image)
        return transformed_image, transformed_image2


def get_len_dict(args):
    length_dict = {}
    if hasattr(args.model_config, 'dataset') and args.model_config.dataset == "bioscan_5m":
        if hasattr(args.model_config, "train_with_small_subset") and args.model_config.train_with_small_subset:
            with h5py.File(args.bioscan_5m_data.path_to_smaller_hdf5_data, "r") as h5file:
                for split in list(h5file.keys()):
                    length_dict[split] = len(h5file[split]["image"])
        else:
            with h5py.File(args.bioscan_5m_data.path_to_hdf5_data, "r") as h5file:
                for split in list(h5file.keys()):
                    length_dict[split] = len(h5file[split]["image"])


    else:
        with h5py.File(args.bioscan_data.path_to_hdf5_data, "r") as h5file:
            for split in list(h5file.keys()):
                length_dict[split] = len(h5file[split]["image"])
    return length_dict


def construct_dataloader(
        args,
        split,
        length,
        return_language=False,
        labels=None,
        for_pre_train=False,
        world_size=None,
        rank=None,
        shuffle=False,
):
    for_open_clip = False
    if hasattr(args.model_config, "for_open_clip"):
        for_open_clip = args.model_config.for_open_clip

    barcode_bert_dna_tokens = None
    # For now, just use sequence, but not feature.
    image_type = "image"
    if hasattr(args.model_config, "image") and hasattr(args.model_config.image, "input_type"):
        image_type = args.model_config.image.input_type

    dna_type = "sequence"
    if hasattr(args.model_config, "dna") and hasattr(args.model_config.dna, "input_type"):
        dna_type = args.model_config.dna.input_type

    if dna_type == "sequence":
        if args.model_config.dataset == "bioscan_5m":
            if hasattr(args.model_config, "train_with_small_subset") and args.model_config.train_with_small_subset:
                hdf5_file = h5py.File(args.bioscan_5m_data.path_to_smaller_hdf5_data, "r", libver="latest")
            else:
                hdf5_file = h5py.File(args.bioscan_5m_data.path_to_hdf5_data, "r", libver="latest")
        else:
            hdf5_file = h5py.File(args.bioscan_data.path_to_hdf5_data, "r", libver="latest")

    dataset = Dataset_for_CL(
        args,
        split,
        length,
        image_type=image_type,
        dna_type=dna_type,
        dna_tokens=barcode_bert_dna_tokens,
        return_language=return_language,
        labels=labels,
        for_training=for_pre_train,
        for_open_clip=for_open_clip,
    )

    num_workers = 8

    if hasattr(args.model_config, "num_workers"):
        num_workers = args.model_config.num_workers

    if for_pre_train:
        if world_size is not None and rank is not None:
            dataloader = prepare(
                dataset,
                rank,
                batch_size=args.model_config.batch_size,
                world_size=world_size,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=args.model_config.batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=num_workers,
            )
    else:
        dataloader = DataLoader(
            dataset, batch_size=args.model_config.batch_size, num_workers=num_workers, shuffle=shuffle
        )
    return dataloader


def load_bioscan_dataloader_with_train_seen_and_separate_keys(args, world_size=None, rank=None, for_pretrain=True):
    length_dict = get_len_dict(args)

    return_language = True

    train_seen_dataloader = construct_dataloader(
        args,
        "train_seen",
        length_dict["train_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
        shuffle=True,
    )

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_keys_dataloader = construct_dataloader(
        args,
        "seen_keys",
        length_dict["seen_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    val_unseen_keys_dataloader = construct_dataloader(
        args,
        "val_unseen_keys",
        length_dict["val_unseen_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    test_unseen_keys_dataloader = construct_dataloader(
        args,
        "test_unseen_keys",
        length_dict["test_unseen_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    return (
        train_seen_dataloader,
        seen_val_dataloader,
        unseen_val_dataloader,
        seen_keys_dataloader,
        val_unseen_keys_dataloader,
        test_unseen_keys_dataloader,
    )

def load_dataloader_for_everything_in_5m(args, world_size=None, rank=None):
    length_dict = get_len_dict(args)

    return_language = True

    

    pre_train_dataloader = construct_dataloader(
        args,
        "no_split_and_seen_train",
        length_dict["no_split_and_seen_train"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
        shuffle=False,
    )

    all_keys_dataloader = construct_dataloader(
        args,
        "all_keys",
        length_dict["all_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_test_dataloader = construct_dataloader(
        args,
        "test_seen",
        length_dict["test_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_test_dataloader = construct_dataloader(
        args,
        "test_unseen",
        length_dict["test_unseen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    other_heldout_dataloader = construct_dataloader(
        args,
        "other_heldout",
        length_dict["other_heldout"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    return pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader, other_heldout_dataloader


def load_dataloader(args, world_size=None, rank=None, for_pretrain=True):
    length_dict = get_len_dict(args)

    return_language = True

    

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    all_keys_dataloader = construct_dataloader(
        args,
        "all_keys",
        length_dict["all_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    if for_pretrain:
        if (
                hasattr(args.model_config, "using_train_seen_for_pre_train")
                and args.model_config.using_train_seen_for_pre_train
        ):
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split_and_seen_train",
                length_dict["no_split_and_seen_train"],
        
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split",
                length_dict["no_split"],
        
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        return pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader
    else:
        train_seen_dataloader = construct_dataloader(
            args,
            "train_seen",
            length_dict["train_seen"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )
        return train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader


def load_bioscan_dataloader_all_small_splits(args, world_size=None, rank=None):
    length_dict = get_len_dict(args)

    return_language = True

    

    if hasattr(args.model_config, 'dataset') and args.model_config.dataset == "bioscan_5m":
        train_seen_dataloader = construct_dataloader(
            args,
            "seen_keys",
            length_dict["seen_keys"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
        )
    else:
        train_seen_dataloader = construct_dataloader(
            args,
            "train_seen",
            length_dict["train_seen"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
        )

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_test_dataloader = construct_dataloader(
        args,
        "test_seen",
        length_dict["test_seen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_test_dataloader = construct_dataloader(
        args,
        "test_unseen",
        length_dict["test_unseen"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_keys_dataloader = construct_dataloader(
        args,
        "seen_keys",
        length_dict["seen_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    if hasattr(args.model_config, 'dataset') and args.model_config.dataset == "bioscan_5m":

        val_unseen_keys_dataloader = construct_dataloader(
            args,
            "unseen_keys",
            length_dict["unseen_keys"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
        )
        test_unseen_keys_dataloader = construct_dataloader(
            args,
            "unseen_keys",
            length_dict["unseen_keys"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
        )
    else:

        val_unseen_keys_dataloader = construct_dataloader(
            args,
            "val_unseen_keys",
            length_dict["val_unseen_keys"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
        )
        test_unseen_keys_dataloader = construct_dataloader(
            args,
            "test_unseen_keys",
            length_dict["test_unseen_keys"],
    
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
        )

    all_keys_dataloader = construct_dataloader(
        args,
        "all_keys",
        length_dict["all_keys"],

        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    return (
        train_seen_dataloader,
        seen_val_dataloader,
        unseen_val_dataloader,
        seen_test_dataloader,
        unseen_test_dataloader,
        seen_keys_dataloader,
        val_unseen_keys_dataloader,
        test_unseen_keys_dataloader,
        all_keys_dataloader
    )


# Following functions are used for bzsl experiments

def species_list_to_input_string_list(species_list, species_to_others):
    levels = ['order', 'family', 'genus']
    four_labels_lits = []

    for species in species_list:
        curr_str = ""
        for level in levels:
            if level not in species_to_others[species]:
                species_to_others[species][level] = 'not_classified'

            curr_str = curr_str + species_to_others[species][level] + " "
        curr_str = curr_str + species
        four_labels_lits.append(curr_str)

    return four_labels_lits


def species_list_to_labels(species_list, species_to_others):
    levels = ['order', 'family', 'genus']
    four_labels_lits = []
    for key in species_to_others.keys():
        for level in levels:
            if level not in species_to_others[key].keys():
                species_to_others[key][level] = 'not_classified'

    np_order = np.array([species_to_others[species]['order'] for species in species_list])
    np_family = np.array([species_to_others[species]['family'] for species in species_list])
    np_genus = np.array([species_to_others[species]['genus'] for species in species_list])
    np_species = np.array(species_list)

    array_of_dicts = np.array([
        {'order': o, 'family': f, 'genus': g, 'species': s}
        for o, f, g, s in zip(np_order, np_family, np_genus, np_species)
    ], dtype=object)

    return array_of_dicts


class INSECTDataset(Dataset):
    def __init__(self, path_to_att_splits_mat, path_to_res_101_mat, image_hdf5_path, species_to_others,
                 split, for_training=False, cl_label=False, for_open_clip=False, **kwargs) -> None:
        super().__init__()
        # self.metadata = pd.read_csv(metadata, sep="\t")

        att_splits_mat = sio.loadmat(path_to_att_splits_mat)
        res_101_mat = sio.loadmat(path_to_res_101_mat)
        # split_loc = att_splits_mat[split][0]

        image_ids = [id_.item() for id_ in res_101_mat["ids"].flatten()]
        barcodes = [bc.item() for bc in res_101_mat["nucleotides"].flatten()]
        species = [sp.item() for sp in res_101_mat["species"].flatten()]

        if split != "all":
            split_loc = att_splits_mat[split][0]
            image_ids = [image_ids[i - 1] for i in split_loc]
            barcodes = [barcodes[i - 1] for i in split_loc]
            species = [species[i - 1] for i in split_loc]

        four_label_str = species_list_to_input_string_list(species, species_to_others)

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        batch_encoded = tokenizer(four_label_str, return_tensors='pt', padding=True)

        self.for_training = for_training

        if self.for_training and cl_label:
            self.labels = np.array(list(range(len(image_ids))))
        else:
            self.labels = species_list_to_labels(species, species_to_others)

        self.input_ids = batch_encoded['input_ids']
        self.attention_mask = batch_encoded['attention_mask']
        self.token_type_ids = batch_encoded.get('token_type_ids')
        self.metadata = pd.DataFrame.from_dict({"image_id": image_ids, "barcode": barcodes, "species": species})

        # self.images = INSECTDataset.compile_images(image_folder)
        self.images = image_hdf5_path
        self.for_open_clip = for_open_clip

        if self.for_training:
            if self.for_open_clip:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.RandomResizedCrop(224, antialias=True),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                             (0.26862954, 0.26130258, 0.27577711)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=(-45, 45)),
                        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.RandomResizedCrop(224, antialias=True),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=(-45, 45)),
                        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    ]
                )
        else:
            if self.for_open_clip:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.CenterCrop(224),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                             (0.26862954, 0.26130258, 0.27577711)),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.CenterCrop(224),
                    ]
                )

    @property
    def image_ids(self):
        return self.metadata["image_id"]

    def _open_hdf5(self):
        self.hdf5_images = h5py.File(self.images, "r")

    def load_image(self, id):
        selected_dataset = self.hdf5_images['images'][id]
        curr_image = Image.open(io.BytesIO(selected_dataset[:]))
        if self.transform is not None:
            curr_image = self.transform(curr_image)
        return curr_image

    def __len__(self):
        return len(self.metadata["image_id"])

    def __getitem__(self, index) -> Any:
        if not hasattr(self, 'hdf5_images'):
            self._open_hdf5()
        row = self.metadata.iloc[index]
        image_id = row["image_id"]

        curr_image = self.load_image(image_id)

        # prepare dna input
        dna_barcode = row["barcode"]
        dna_barcode = torch.tensor(dna_barcode, dtype=torch.int64)

        label = self.labels[index]

        return image_id, curr_image, dna_barcode, self.input_ids[index], self.attention_mask[index], \
            self.token_type_ids[index], label


def load_insect_dataloader_trainval(args, num_workers=8, shuffle_for_train_seen_key=False):
    filename = args.insect_data.species_to_other
    with open(filename, 'r') as file:
        specie_to_other_labels = json.load(file)

    

    trainval_dataset = INSECTDataset(
        args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
        species_to_others=specie_to_other_labels, split="trainval_loc",
        image_hdf5_path=args.insect_data.path_to_image_hdf5,
         for_training=True, cl_label=False,
        for_open_clip=args.model_config.for_open_clip
    )

    insect_trainval_dataloader = DataLoader(trainval_dataset, batch_size=args.model_config.batch_size,
                                            num_workers=num_workers, shuffle=True)

    return insect_trainval_dataloader


def load_insect_dataloader(args, world_size=None, rank=None, num_workers=8, load_all_in_one=False,
                           shuffle_for_train_seen_key=False):
    filename = args.insect_data.species_to_other
    with open(filename, 'r') as file:
        specie_to_other_labels = json.load(file)

    

    if load_all_in_one:
        all_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="all",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            for_training=False, for_open_clip=args.model_config.for_open_clip
        )

        all_dataloader = DataLoader(all_dataset, batch_size=args.model_config.batch_size,
                                    num_workers=num_workers, shuffle=False)
        return all_dataloader

    else:
        train_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="train_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            for_training=True, cl_label=True,
            for_open_clip=args.model_config.for_open_clip
        )

        train_dataset_for_key = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="train_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            for_training=False, for_open_clip=args.model_config.for_open_clip
        )

        val_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="val_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            for_training=False, for_open_clip=args.model_config.for_open_clip
        )

        test_seen_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="test_seen_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            for_training=False, for_open_clip=args.model_config.for_open_clip
        )

        test_unseen_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="test_unseen_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            for_training=False, for_open_clip=args.model_config.for_open_clip
        )
        if rank is None:
            print(rank)
            insect_train_dataloader = DataLoader(train_dataset, batch_size=args.model_config.batch_size,
                                                 num_workers=num_workers, shuffle=True)
        else:
            insect_train_dataloader = prepare(train_dataset, rank, batch_size=args.model_config.batch_size,
                                              world_size=world_size,
                                              num_workers=num_workers, shuffle=True)

        insect_train_dataloader_for_key = DataLoader(train_dataset_for_key, batch_size=args.model_config.batch_size,
                                                     num_workers=num_workers, shuffle=shuffle_for_train_seen_key)
        insect_val_dataloader = DataLoader(val_dataset, batch_size=args.model_config.batch_size,
                                           num_workers=num_workers, shuffle=False)

        insect_test_seen_dataloader = DataLoader(test_seen_dataset, batch_size=args.model_config.batch_size,
                                                 num_workers=num_workers, shuffle=False)

        insect_test_unseen_dataloader = DataLoader(test_unseen_dataset, batch_size=args.model_config.batch_size,
                                                   num_workers=num_workers, shuffle=False)

        return insect_train_dataloader, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader, insect_test_unseen_dataloader
