import io
import json
import os
import random

import h5py
import numpy as np
from PIL import Image


def load_image_from_h5(data, idx):
    """Load image file from HDF file"""
    enc_length = data["image_mask"][idx]
    image_enc_padded = data["image"][idx].astype(np.uint8)
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image.resize((256, 256))


def main():
    # load path to bioscan-1m hdf5
    path_to_5m = "Path to the hdf5 file"
    # You can rename the path to the hdf5 file here, so you can remove the parts about hydra

    # create a dictionary. Key: sampleid, value: split, index
    sample_id_to_dna_bin = {}
    if os.path.exists("sample_id_to_dna_bin.json"):
        with open("sample_id_to_dna_bin.json", "r") as f:
            sample_id_to_dna_bin = json.load(f)
    else:
        with h5py.File(path_to_5m, "r") as f:
            for split in f.keys():
                for index, sample_id in enumerate(f[split]['sampleid'][:]):
                    sample_id_to_dna_bin[sample_id.decode("utf-8")] = {"split": split, "index": index}
            # Save the dictionary to a json file
            with open("sample_id_to_dna_bin.json", "w") as f:
                json.dump(sample_id_to_dna_bin, f)

    # To load single image from the hdf5 file with the sample_id as the key
    random_sample_id = random.choice(list(sample_id_to_dna_bin.keys()))

    with h5py.File(path_to_5m, "r") as f:
        split = sample_id_to_dna_bin[random_sample_id]["split"]
        index = sample_id_to_dna_bin[random_sample_id]["index"]
        data = f[split]
        image = load_image_from_h5(data, index)
        # plt.imshow(image)
        save_image_path = f"{random_sample_id}.png"
        image.save(save_image_path)


if __name__ == "__main__":
    main()
