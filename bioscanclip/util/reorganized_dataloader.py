import io
from itertools import product

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from bioscanclip.model.dna_encoder import get_sequence_pipeline


class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))


class KmerTokenizer(object):
    def __init__(self, k, stride=1):
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i: i + self.k]
            tokens.append(k_mer)
        return tokens


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


class Dataset_for_Image_and_DNA(Dataset):
    def __init__(
            self,
            dataset,
            hdf5_path,
            split,
            dna_tokens=None,
            for_training=False,
    ):
        # Path to the hdf5 file and indicate the dataset is 1M or 5M
        self.hdf5_inputs_path = hdf5_path
        self.dataset = dataset

        # Specify the split
        self.split = split

        if dna_tokens is not None:
            self.dna_tokens = torch.tensor(dna_tokens)
        self.for_training = for_training
        self.pre_train_with_small_set = False

        # language_model_name = "prajjwal1/bert-small"
        # self.tokenizer, _ = load_pre_trained_bert(language_model_name)

        self.labels = get_array_of_label_dicts(self.hdf5_inputs_path, split)

        if self.for_training:
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
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=256, antialias=True),
                    transforms.CenterCrop(224),
                ]
            )

    def __len__(self):
        if not hasattr(self, "length"):
            self._open_hdf5()
        return self.length

    def _open_hdf5(self):
        self.hdf5_split_group = h5py.File(self.hdf5_inputs_path, "r", libver="latest")[self.split]
        self.length = len(self.hdf5_split_group["image"])

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
        curr_image_input = self.load_image(idx)


        if self.dna_tokens is None:
            curr_dna_input = self.hdf5_split_group["barcode"][idx].decode("utf-8")
        else:
            curr_dna_input = self.dna_tokens[idx]

        if self.dataset == "bioscan_5m":
            curr_processid = self.hdf5_split_group["processid"][idx].decode("utf-8")
        else:
            curr_processid = self.hdf5_split_group["image_file"][idx].decode("utf-8")

        return (
            curr_processid,
            curr_image_input,
            curr_dna_input,
            self.labels[idx],
        )



def tokenize_dna_sequence(pipeline, dna_input):
    list_of_output = []
    for i in dna_input:
        list_of_output.append(pipeline(i))
    return list_of_output


if __name__ == '__main__':
    # tokenize DNA barcode

    dataset = "bioscan_5m"  # or "bioscan_1m"
    path_to_hdf5_data = "/localhome/zmgong/second_ssd/projects/bioscan-clip/data/BIOSCAN_5M/BIOSCAN_5M.hdf5"
    # split = "no_split_and_seen_train"  # data we use to train
    split = "val_seen"  # data we use to validate
    hdf5_file = h5py.File(path_to_hdf5_data, "r", libver="latest")

    sequence_pipeline = get_sequence_pipeline()

    unprocessed_dna_barcode = np.array([item.decode("utf-8") for item in hdf5_file[split]["barcode"][:]])
    barcode_bert_dna_tokens = tokenize_dna_sequence(sequence_pipeline, unprocessed_dna_barcode)

    dataset = Dataset_for_Image_and_DNA(
        dataset=dataset,
        hdf5_path=path_to_hdf5_data,
        split=split,
        dna_tokens=barcode_bert_dna_tokens,
        for_training=False,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for processid, image, dna, label in dataloader:
        print(processid, image, dna, label)
        break
