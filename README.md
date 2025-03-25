# CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale
This is the official implementation for "CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale".
Links: [website](https://bioscan-ml.github.io/clibd/) | [paper](https://arxiv.org/abs/2405.17537)

# Overview
![Teaser](./docs/static/images/method.svg)
Taxonomically classifying organisms at scale is crucial for monitoring biodiversity, understanding ecosystems, and preserving sustainability.  It is possible to taxonomically classify organisms based on their image or their [DNA barcode](https://en.wikipedia.org/wiki/DNA_barcoding).  While DNA barcodes are precise at species identification, they are less readily available than images.  Thus, we investigate whether we can use DNA barcodes to improve taxonomic classification using image.  

We introduce CLIBD, a model uses contrastive learning to map biological images, DNA barcodes, and textual taxonomic labels to the same latent space.  The model is initialized using pretrained encoders for images ([vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)), DNA barcodes ([BarcodeBERT](https://github.com/Kari-Genomics-Lab/BarcodeBERT)), and textual taxonomic labels ([BERT-small](https://huggingface.co/prajjwal1/bert-small)), and the weights of the encoders are fine-tuned using LoRA.
The aligned image-DNA embedding space improves taxonomic classification using images and allows us to do cross-modal retrieval from image to DNA. We train CLIBD on the [BIOSCAN-1M](https://biodiversitygenomics.net/projects/1m-insects/) and [BIOSCAN-5M](https://biodiversitygenomics.net/projects/5m-insects/) insect datasets.  These datasets provides paired images of insects and their DNA barcodes, along with their taxonomic labels.  

# Setup environment
CLIBD was developed using Python 3.10 and PyTorch 2.0.1.  We recommend the use of GPU and CUDA for efficient training and inference.  Our models were developed with CUDA 11.7 and 12.4.  
We also recommend the use of [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) for managing your environments. 

To setup the environment width necessary dependencies, type the following commands:
```shell
conda create -n CLIBD python=3.10 -y
conda activate CLIBD
conda install pytorch=2.0.1 torchvision=0.15.2 torchtext=0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/Baijiong-Lin/LoRA-Torch
```



Depending on your GPU version, you may have to modify the torch version and other package versions in [requirements.txt](./requirements.txt).

# Pretrained embeddings and models
We provide pretrained embeddings and model weights.  We evaluate our models by encoding the image or DNA barcode, and using the taxonomic labels from the closest matching embedding (either using image or DNA barcode).  See [Download dataset](#download-dataset) and [Running Experiments](#running-experiments) for how to get the data, and to train and evaluate the models.

**NOTE: Currently the checkpoints and config files pointed by these links are expired, we are working to update them as soon as possible. We deeply apologize for any inconvenience this may have caused.**

| Training data |  Aligned modalities |  Embeddings |  Model  | Config |
|---------------|---------------------|-------------|---------|--------|
| BIOSCAN-1M    |  None               |  [Embedding](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/no_alignment/extracted_feature_from_test_split.hdf5) |  N/A   |  [Link](https://github.com/bioscan-ml/clibd/blob/main/bioscanclip/config/model_config/for_bioscan_1m/final_experiments/image_dna_text_no_loading.yaml)  |
| BIOSCAN-1M    |  **I**mage + **D**NA        |  [Embedding](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/image_dna/extracted_feature_from_test_split.hdf5)|  [Link](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/image_dna/best.pth)   |  [Link](https://github.com/bioscan-ml/clibd/blob/main/bioscanclip/config/model_config/for_bioscan_1m/final_experiments/image_dna_seed_42.yaml)  |
| BIOSCAN-1M    |  **I**mage + **D**NA + **T**ax  |  [Embedding](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/image_dna_text/extracted_feature_from_test_split.hdf5)       |  [Link](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/image_dna_text/best.pth) |  [Link](https://github.com/bioscan-ml/clibd/blob/main/bioscanclip/config/model_config/for_bioscan_1m/final_experiments/image_dna_text_seed_42.yaml) |
| BIOSCAN-5M    |  None               |  [Embedding](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/bioscan_5m/no_alignment/extracted_feature_from_test_split.hdf5)       |  N/A   |  [Link](https://github.com/bioscan-ml/clibd/blob/main/bioscanclip/config/model_config/for_bioscan_5m/no_alignment_baseline/no_align.yaml)  |
| BIOSCAN-5M    |  **I**mage + **D**NA        |  [Embedding](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/bioscan_5m/image_dna/extracted_feature_from_val_split.hdf5)      |  [Link](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/bioscan_5m/image_dna/best.pth)    |  [Link](https://github.com/bioscan-ml/clibd/blob/main/bioscanclip/config/model_config/for_bioscan_5m/final_experiments/image_dna_seed_42.yaml)  |
| BIOSCAN-5M    |  **I**mage + **D**NA + **T**ax  |  [Embedding](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/bioscan_5m/image_dna_text/extracted_feature_from_val_split.hdf5)      |  [Link](https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/bioscan_5m/image_dna_text/best.pth)|   [Link](https://github.com/bioscan-ml/clibd/blob/main/bioscanclip/config/model_config/for_bioscan_5m/final_experiments/image_dna_text_seed_42.yaml)  |

We also provide checkpoints trained with LoRA layers. You can download them from this [Link](https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/ckpt/bioscan_clip/lora_version_ckpt.zip)

## Quick start
Instead of conducting a full training, you can choose to download pre-trained models or pre-extracted embeddings for evaluation from the table. You may need to posistion the downloaded checkpoints and extracted features in to the proper position based on the config file.

# Download dataset
![Data Partioning Visual](./docs/static/images/partition.svg) <br>
For BIOSCAN 1M, we partition the dataset for our CLIBD experiments into a training set for contrastive learning, and validation and test partitions. The training set has records without any species labels as well as a set of seen species. The validation and test sets include seen and unseen species. These images are further split into subpartitions of queries and keys for evaluation.

For BIOSCAN 5M, we use the dataset partitioning established in the BIOSCAN-5M paper.

For training and reproducing our experiments, we provide HDF5 files with BIOSCAN-1M and BIOSCAN-5M images.  See [DATA.md](DATA.md) for format details. We also provide scripts for generating the HDF5 files directly from the BIOSCAN-1M and BIOSCAN-5M data.

### Download BIOSCAN-1M data (79.7 GB)
```shell
# From project folder
mkdir -p data/BIOSCAN_1M/split_data
cd data/BIOSCAN_1M/split_data
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/data/version_0.2.1/BioScan_data_in_splits.hdf5
```

### Download BIOSCAN-5M data (190.4 GB)
```shell
# From project folder
mkdir -p data/BIOSCAN_5M
cd data/BIOSCAN_5M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_CLIP_for_downloading/BIOSCAN_5M.hdf5
```
For more information about the hdf5 files, please check [DATA.md](DATA.md).

You can also download the processed data by checking our [huggimgface repo](https://huggingface.co/datasets/bioscan-ml/bioscan-clibd)

### Download data for generating hdf5 files

You can check [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M) and [BIOSCAN-5M](https://github.com/zahrag/BIOSCAN-5M) to download tsv files. But they are actually not necessary.

# Running experiments
We recommend the use of [weights and biases](https://wandb.ai/site) to track and log experiments

## Activate Wandb
#### Register/Login for a [free wandb account](https://wandb.ai/site)
```shell
wandb login
# Paste your wandb's API key
```
Note: To enable wandb, you also need to modify /bioscanclip/config/global_config and set:
```yaml
debug_flag: false
```


## Checkpoints

Download checkpoint for BarcodeBERT and bioscan_clip and place them under `ckpt`.  
```shell
# From project folder
mkdir -p ckpt/BarcodeBERT/5_mer
cd ckpt/BarcodeBERT/5_mer
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/ckpt/BarcodeBERT/model_41.pth
cd ../../..
mkdir -p ckpt/bioscan_clip/final_experiments/image_dna_text_4gpu_50epoch
cd ckpt/bioscan_clip/final_experiments/image_dna_text_4gpu_50epoch
wget https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/image_dna_text/best.pth
cd ../../../..
mkdir -p ckpt/bioscan_clip/new_5M_training/trained_with_5M_image_dna_text
cd ckpt/bioscan_clip/new_5M_training/trained_with_5M_image_dna_text
wget https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/bioscan_5m/image_dna_text/best.pth
```
For downloading all CLIBD pre-trained models: [Link](https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_CLIP_for_downloading/ckpt.zip)

## Train

Use [train_cl.py](./scripts/train_cl.py) with the appropriate `model_config` to train CLIBD.
```shell
# From project folder
python scripts/train_cl.py 'model_config={config_name}'
```

To train the full model (I+D+T) using BIOSCAN-1M:
```shell
# From project folder
python scripts/train_cl.py 'model_config=for_bioscan_1m/final_experiments/image_dna_text_seed_42.yaml'
```
For multi-GPU training, you may need to specify the transport communication between the GPU using NCCL_P2P_LEVEL:
```shell
NCCL_P2P_LEVEL=NVL python scripts/train_cl.py 'model_config=for_bioscan_1m/final_experiments/image_dna_text_seed_42.yaml'
```

For example, using the following command, you can load the pre-trained ViT-B, BarcodeBERT, and BERT-small and fine-tune them through contrastive learning. Note that this training will only update their LoRA layers, not all the parameters.
```shell
python scripts/train_cl.py 'model_config=for_bioscan_5m/lora_vit_lora_barcode_bert_lora_bert_5m_no_loading.yaml'
```

## Evaluation

During evaluation, we using the trained encoders to obtain embeddings for input image or DNA, and the find the closest matching image or DNA and use the corresponding taxonomical labels as the predicted labels.  We report both the micro and class averaged accuracy for seen and unseen species. 

To run evaluation for BIOSCAN-1M:
```shell
# From project folder
python scripts/inference_and_eval.py 'model_config=for_bioscan_1m/final_experiments/image_dna_text_seed_42.yaml'
```

To run evaluation for BIOSCAN-5M:
```shell
python scripts/inference_and_eval.py 'model_config=for_bioscan_5m/final_experiments/image_dna_text_seed_42.yaml'
```

## For BZSL experiment with the INSECT dataset.

To download unprocessed INSECT dataset, you can reference [BZSL](https://github.com/sbadirli/Fine-Grained-ZSL-with-DNA): 

```shell
mkdir -p data/INSECT
cd data/INSECT
# Download the images and metadata here.


# Note that we need to get the other three labels because the INSECT dataset only has the species label.
# For that, please edit get_all_species_taxo_labels_dict_and_save_to_json.py, change Entrez.email = None to your email 
pip install biopython
python get_all_species_taxo_labels_dict_and_save_to_json.py

# Then, generate CSV and hdf5 file for the dataset.
python process_insect_dataset.py
```
The downloaded data should be organized in this way:
```shell
data
├── INSECT
│   ├── att_splits.mat
│   ├── res101.mat
│   ├── images
│   │   │   ├── Abax parallelepipedus
│   │   │   │   ├── BC_ZSM_COL_02878+1311934584.jpg
│   │   │   │   ├── BC_ZSM_COL_05487+1338577126.JPG
│   │   │   │   ├── ...
│   │   │   ├── Abax parallelus
│   │   │   ├── Acordulecera dorsalis
│   │   │   ├── ...
```

You can also download the processed file with:
```shell
wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_CLIP_for_downloading/INSECT_data/processed_data.zip
unzip processed_data.zip
```

### Train CLIBD with INSECT dataset
```shell
python scripts/train_cl.py 'model_config=for_bioscan_1m/lora_vit_lora_barcode_bert_lora_bert_ssl_on_insect.yaml'
```

###  Extract image and DNA features of INSECT dataset.

To perform contrastive learning for fine-tuning on the INSECT dataset.

```shell
python scripts/train_cl.py 'model_config=for_bioscan_1m/fine_tune_on_INSECT_dataset/image_dna_text_seed_42_on_INSECT_dataset.yaml'
```

To perform supervise fine-tune image encoder with INSECT dataset.
```shell
python scripts/BZSL/fine_tune_bioscan_clip_image_on_insect.py 'model_config=for_bioscan_1m/final_experiments/image_dna_text_seed_42.yaml'
```

For feature extracting

```shell
python scripts/extract_feature_for_insect_dataset.py 'model_config=for_bioscan_1m/fine_tune_on_INSECT_dataset/image_dna_text_seed_42_on_INSECT_dataset.yaml'
```
Then, you may move the extracted features to the BZSL folder or download the pre-extracted feature.

```shell
mkdir -p Fine-Grained-ZSL-with-DNA/data/INSECT/embeddings_from_bioscan_clip
cp extracted_embedding/INSECT/dna_embedding_from_bioscan_clip.csv Fine-Grained-ZSL-with-DNA/data/INSECT/embeddings_from_bioscan_clip/dna_embedding_from_bioscan_clip.csv
cp extracted_embedding/INSECT/image_embedding_from_bioscan_clip.csvFine-Grained-ZSL-with-DNA/data/INSECT/embeddings_from_bioscan_clip/image_embedding_from_bioscan_clip.csv
```

###  Run BZSL for evaluation.
```shell
cd Fine-Grained-ZSL-with-DNA/BZSL-Python
python Demo.py --using_bioscan_clip_image_feature --datapath ../data --side_info dna_bioscan_clip --alignment --tuning
```

###  Flatten the `results.csv`.
```shell
python scripts/flattenCsv.pya -i PATH_TO_RESULTS_CSV -o PATH_TO_FLATTEN_CSV
```

# Citing CLIBD
If you use CLIBD in your research, please cite:
```bibtex
@article{gong2024clibd,
  title={{CLIBD}: Bridging Vision and Genomics for Biodiversity Monitoring at Scale},
  author={Gong, ZeMing and Wang, Austin T. and Huo, Xiaoliang and Haurum, Joakim Bruslund and Lowe, Scott C. and Taylor, Graham W. and Chang, Angel X.},
  journal={arXiv preprint arXiv:2405.17537},
  year={2024},
  eprint={2405.17537},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arxiv.2405.17537},
}
```

# Acknowledgements
We would like to express our gratitude for the use of the INSECT dataset, which played a pivotal role in the completion of our experiments. Additionally, we acknowledge the use and modification of code from the [Fine-Grained-ZSL-with-DNA](https://github.com/sbadirli/Fine-Grained-ZSL-with-DNA) repository, which facilitated part of our experimental work. The contributions of these resources have been invaluable to our project, and we appreciate the efforts of all developers and researchers involved.

This reseach was supported by the Government of Canada’s New Frontiers in Research Fund (NFRF) [NFRFT-2020-00073], 
Canada CIFAR AI Chair grants, and the Pioneer Centre for AI (DNRF grant number P1).
This research was also enabled in part by support provided by the Digital Research Alliance of Canada (alliancecan.ca).
