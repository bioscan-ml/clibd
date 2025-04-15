Configuration Documentation

This document explains the purpose and usage of each configuration item in the global configuration file.

## Table of Contents

- [Global Configuration Details](#global-configuration-details)
  - [General Settings](#general-settings)
  - [Data Paths and Directories](#data-paths-and-directories)
  - [Dataset Configurations](#dataset-configurations)
    - [BIOSCAN 1M Data](#bioscan-1m-data)
    - [BIOSCAN 5M Data](#bioscan-5m-data)
    - [INSECT Data](#insect-data)
  - [Checkpoints and Output](#checkpoints-and-output)
  - [Inference and Evaluation Settings](#inference-and-evaluation-settings)
  - [Visualization Settings](#visualization-settings)
  - [Fine-tuning Settings](#fine-tuning-settings)
  - [Random Seed](#random-seed)
- [Model Configuration](#model-configuration)
- [Usage Instructions](#usage-instructions)
- [Notes](#notes)
- [Future Updates](#future-updates)

## Global Configuration Details

### General Settings

- **debug_flag: false**
  A flag that used to debug. When set to true, during training and inference, wandb will be disabled, and neither the inference results nor the model's checkpoint will be saved.

- **project_root_path: ${hydra:runtime.cwd}**
  The root directory of the project, set to the current working directory at runtime.

- **data_dir: ${project_root_path}/data**
  The directory where data is stored. All data files are located within this directory or its subdirectories.

- **activate_wandb: true**
  Whether to enable Weights & Biases (wandb) for experiment tracking and logging.

- **save_inference: true**
  Whether to save inference results. If set to true, the inference embeddings and evaluation results will be saved.

- **load_inference: false**
  Whether to load previously saved inference embeddings for evaluation.

- **enable_early_stopping: false**
  Whether to enable early stopping during model training.

### Data Paths and Directories

All data-related paths are defined based on `project_root_path` and `data_dir` to ensure consistency in data storage and management.

### Dataset Configurations

#### BIOSCAN 1M Data

- **bioscan_data.dir: ${data_dir}/BIOSCAN_1M**
  The directory where the BIOSCAN 1M dataset is stored.

- **bioscan_data.path_to_hdf5_data: ${bioscan_data.dir}/split_data/BioScan_data_in_splits.hdf5**
  The HDF5 file path containing the BIOSCAN 1M dataset splits.

- **bioscan_data.path_to_tsv_data: ${bioscan_data.dir}/BioScan_Insect_Dataset_metadata_2.tsv**
  The TSV metadata file for the BIOSCAN 1M dataset.

- **bioscan_data.path_to_id_to_position_mapping: ${bioscan_data.dir}/id_to_position_mapping.json**
  The JSON file mapping IDs to their corresponding positions.

#### BIOSCAN 5M Data

- **bioscan_5m_data.dir: ${data_dir}/BIOSCAN_5M**
  The directory where the BIOSCAN 5M dataset is stored.

- **bioscan_5m_data.image_dir: ${bioscan_5m_data.dir}/images**
  The directory where BIOSCAN 5M dataset images are stored.

- **bioscan_5m_data.path_to_hdf5_data: ${bioscan_5m_data.dir}/BIOSCAN_5M.hdf5**
  The HDF5 file path containing the BIOSCAN 5M dataset.

- **bioscan_5m_data.path_to_smaller_hdf5_data: ${bioscan_5m_data.dir}/BIOSCAN_5M_with_small_pre_train.hdf5**
  The HDF5 file path that includes a subset for pre-training.

- **bioscan_5m_data.path_to_tsv_data: ${bioscan_5m_data.dir}/BIOSCAN-5M_Dataset_v3.4.csv**
  The CSV metadata file for the BIOSCAN 5M dataset.

- **bioscan_5m_data.path_to_small_species_list_json: ${bioscan_5m_data.dir}/**
  The directory for the JSON file containing the small species list (please add the specific file name as needed).

- **bioscan_5m_data.path_to_id_to_position_mapping: ${bioscan_5m_data.dir}/id_to_position_mapping.json**
  The JSON file mapping IDs to positions for the BIOSCAN 5M dataset.

#### INSECT Data

- **insect_data.dir: ${data_dir}/INSECT**
  The directory where the INSECT dataset is stored.

- **insect_data.path_to_att_splits_mat: ${insect_data.dir}/att_splits.mat**
  The MATLAB file containing attribute split information.

- **insect_data.path_to_res_101_mat: ${insect_data.dir}/res101.mat**
  The MATLAB file containing ResNet-101 features.

- **insect_data.path_to_feature_hdf5: ${insect_data.dir}/extracted_features/clip_barcodebert_extracted_features.hdf5**
  The HDF5 file containing extracted feature data.

- **insect_data.path_to_image_hdf5: ${insect_data.dir}/INSECT_images.hdf5**
  The HDF5 file for INSECT dataset images.

- **insect_data.path_to_meta_csv: ${insect_data.dir}/INSECT_metadata.csv**
  The CSV metadata file for the INSECT dataset.

- **insect_data.image_dir: ${insect_data.dir}/images**
  The directory where INSECT dataset images are stored.

- **insect_data.species_to_other: ${insect_data.dir}/specie_to_other_labels.json**
  The JSON file mapping species to other labels.

### Checkpoints and Output

- **save_ckpt: true**
  Whether to save checkpoints generated during training.

- **bioscan_bert_checkpoint: ${project_root_path}/ckpt/BarcodeBERT/5_mer/model_41.pth**
  The checkpoint file path for the BarcodeBERT pre-trained model.

- **model_output_dir: ${project_root_path}/ckpt/bioscan_clip**
  The directory for storing model outputs and checkpoints.

### Inference and Evaluation Settings

- **inference_and_eval_setting.plot_embeddings: true**
  Whether to plot or save feature embeddings after inference.

- **inference_and_eval_setting.retrieve_images: false**
  Whether to retrieve images based on queries after inference.

- **inference_and_eval_setting.k_list: [1, 3, 5]**
  The list of top-k values used for evaluation.

- **inference_and_eval_setting.levels: ['order', 'family', 'genus', 'species']**
  The classification levels for evaluation, such as order, family, genus, and species.

- **inference_and_eval_setting.embeddings_filters**
  Filters for embeddings:
  - `order`: e.g., "Diptera"
  - `family`: e.g., "Sciaridae"
  - `genus`: e.g., "Corynoptera"
  For example, during plot embeddings for genus, the filter is "Diptera". Only the genus in the "Diptera" order will be plotted.

- **inference_and_eval_setting.retrieve_settings**
  Retrieval settings include:
  - `num_queries`: Number of queries (e.g., 5)
  - `max_k`: Maximum retrieval count (e.g., 3)
  - `seed`: Random seed (e.g., 413)
  - `independent`: Whether retrieval is performed independently (false indicates not independent)
  - `load_cached_results`: Whether to load cached results (false indicates not to load)

- **inference_and_eval_setting.eval_on: test**
  Specifies which data partition is used for evaluation (val or test).


### Visualization Settings
- **visualization.output_dir: ${project_root_path}/visualizations**
  The directory for storing visualization outputs.

### Random Seed

- **default_seed: 42**
  The default random seed to ensure the reproducibility of experimental results.

