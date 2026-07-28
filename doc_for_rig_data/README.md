# CBG RIG Image Processing and CLIBD Retrieval Evaluation

Last verified: 2026-07-28

## Project goal

CBG provided a new collection of RIG images, DNA records, and taxonomic
metadata. Unlike most BIOSCAN images, which show insects photographed in
dishes, the RIG images show insects mounted on metal pins.

The goal is to evaluate whether the existing CLIBD model trained on
BIOSCAN-5M transfers to this new image domain. We first use BIOSCAN-5M as the
retrieval reference, then compare it with a larger BIOSCAN-5M reference and an
in-domain RIG reference. These experiments will help determine whether CLIBD
can be used directly or should be fine-tuned or retrained with the RIG data.

## Current status

### Completed

1. Converted the CBG metadata to a CLIBD/BIOSCAN-compatible CSV.
2. Selected the `COI-5P` records.
3. Resized all selected images to a 256-pixel short edge while preserving
   aspect ratio and saved them as lossless RGB PNG files.
4. Generated and validated a CLIBD-compatible HDF5 file.
5. Computed CBG taxonomy statistics and exact label overlap with BIOSCAN-5M.
6. Completed Phase 1 retrieval using the BIOSCAN-5M `all_keys` reference.

### Not completed

1. Phase 2 retrieval against the deduplicated full BIOSCAN-5M reference.
2. Phase 3 retrieval using disjoint RIG query and key sets.
3. Any CLIBD fine-tuning or retraining with RIG data.

## Important release status

The public repository is:

<https://github.com/bioscan-ml/clibd>

All RIG processing, statistics, retrieval, configuration, and documentation
files have now been organized under the local `clibd/` repository. They are
**not yet available from a clean public clone** because the changes still need
to be reviewed, committed, and pushed. The workflow does not depend on the
separate `clibd_hyperbolic` repository.

Before handing the project to a collaborator, commit and push the files listed
in [Code map](#code-map), then run the smoke test from a clean clone.

## Data summary

| Item | Count |
|---|---:|
| CBG image files | 464,963 |
| Metadata rows | 468,007 |
| Unique `sampleid` values | 464,893 |
| Samples with more than one `marker_code` | 3,096 |
| `COI-5P` records | 464,541 |
| Final HDF5 records | 464,530 |
| Species-labelled Phase 1 queries | 116,161 |

The final HDF5 excludes 11 `COI-5P` records with an empty `dna_barcode`.
It does not filter on taxonomy: missing labels are retained as
`not_classified`.

Final HDF5:

- filename: `BIOSCAN_CBG_COI5P.hdf5`;
- group: `no_split`;
- size: 43,561,852,642 bytes (about 40.6 GiB);
- records: 464,530;
- unique `sampleid`: 464,530;
- unique `processid`: 464,530;
- image format: lossless RGB PNG bytes;
- image short edge: 256 pixels;
- stored fields: image, DNA barcode, DNA BIN, order, family, genus, species,
  `sampleid`, `processid`, `record_id`, `specimenid`, `image_file`,
  `source_image_file`, and `marker_code`.

The full validation compared all 464,530 HDF5 rows with the filtered CSV and
also decoded and inspected a fixed sample of 100 records.

## Data distribution

Proposed public staging directory:

```text
/project/3dlg-hcvc/bioscan/www/processed_rig_image_data/
```

Proposed HTTPS base:

```text
https://aspis.cmpt.sfu.ca/projects/bioscan/processed_rig_image_data/
```

The directory currently exists but is empty. The base URL currently returns
HTTP 403 because directory listing is disabled. This does not prevent direct
file downloads after files are published, but the URL for the directory itself
is not a downloadable dataset.

Publish explicit filenames and a checksum manifest:

```text
processed_rig_image_data/
├── BIOSCAN_CBG_COI5P.hdf5
├── BIOSCAN_CBG_COI5P_generation_report.json
├── BIOSCAN_CBG_COI5P_validation_report.json
├── phase1_all_keys_results.tar.gz              # optional
├── MANIFEST.tsv
└── SHA256SUMS
```

Minimum files for evaluation:

1. `BIOSCAN_CBG_COI5P.hdf5`
2. `BIOSCAN_CBG_COI5P_generation_report.json`
3. `BIOSCAN_CBG_COI5P_validation_report.json`

Optional Phase 1 artifacts:

- query and `all_keys` reference embeddings;
- query/reference Parquet manifests;
- Top-10 prediction Parquet files;
- `retrieval_metrics.json`;
- resolved `config.yaml` and run log.

The resized-image directory is about 40 GiB, but it is not required for
evaluation because the final HDF5 already contains the resized image bytes.
The FAISS indices can also be rebuilt from cached embeddings and do not need to
be in the minimum download.

After publication, download files by their complete URLs:

```bash
BASE=https://aspis.cmpt.sfu.ca/projects/bioscan/processed_rig_image_data

wget -c "$BASE/BIOSCAN_CBG_COI5P.hdf5"
wget -c "$BASE/BIOSCAN_CBG_COI5P_generation_report.json"
wget -c "$BASE/BIOSCAN_CBG_COI5P_validation_report.json"
wget -c "$BASE/SHA256SUMS"
sha256sum -c SHA256SUMS
```

Do **not** use only:

```bash
wget https://aspis.cmpt.sfu.ca/projects/bioscan/processed_rig_image_data
```

That command requests a directory page rather than the dataset files.

### Data-release check

The `/project/.../www` tree is publicly accessible. Confirm with CBG that the
RIG images, DNA sequences, and specimen identifiers may be redistributed
before publishing them. Do not publish the raw metadata or the complete
converted CSV without review: they contain additional collection information,
including geographic coordinates. Publish only the minimum approved fields and
include the agreed license/citation.

## Code map

The following paths are the intended final paths in the public `clibd`
repository:

| Purpose | Path |
|---|---|
| Inspect the source ZIP and build safe batches | `scripts/data_processing/inspect_rig_zip_and_build_batches.py` |
| Copy, extract, transfer, and verify ZIP batches | `scripts/data_processing/process_rig_zip_batches.py` |
| Convert RIG metadata to BIOSCAN CSV | `scripts/data_processing/convert_rig_metadata_to_bioscan_csv.py` |
| Resize `COI-5P` images | `scripts/data_processing/resize_cbg_coi5p_images.py` |
| Generate HDF5 | `scripts/data_processing/generate_cbg_coi5p_hdf5.py` |
| Validate HDF5 | `scripts/data_processing/validate_cbg_coi5p_hdf5.py` |
| Generate taxonomy statistics | `scripts/stat/generate_coi5p_taxonomy_statistics.py` |
| RIG × BIOSCAN-5M retrieval runner | `scripts/retrieval/run_cbg_bioscan5m_retrieval.py` |
| Retrieval Hydra config | `bioscanclip/config/cbg_retrieval.yaml` |
| Streaming embedding helper | `bioscanclip/epoch/inference_epoch.py` |
| Skip unused text encoding | `bioscanclip/model/simple_clip.py` |
| Detailed retrieval design | `doc_for_rig_data/cbg_rig_data_bioscan5m_retrieval_plan.md` |

The retrieval pipeline is stage-driven:

```text
prepare -> embed -> index -> search -> evaluate
```

Each stage saves reusable artifacts and can be resumed independently.

The metadata converter validates against its built-in BIOSCAN-compatible
schema. Passing `--reference-csv` is optional and provides an additional
header check against an existing BIOSCAN-5M metadata CSV.

## Environment setup

The work was run in the `CLIBD-hyperbolic` conda environment with Python 3.10,
PyTorch 2.0.1, CUDA 11.7, FAISS 1.7.2, Hydra 1.3.2, HDF5, Pillow, pandas, and
PyArrow.

For a new machine, start with the public CLIBD setup:

```bash
git clone https://github.com/bioscan-ml/clibd.git
cd clibd

conda create -n CLIBD python=3.10 -y
conda activate CLIBD
conda install pytorch=2.0.1 torchvision=0.15.2 \
  torchtext=0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/Baijiong-Lin/LoRA-Torch
```

`pyarrow` is required by the RIG retrieval runner and is included in
`requirements.txt`.

## Download the CLIBD checkpoint and BIOSCAN-5M reference

The Phase 1 model is the CLIBD model trained with BIOSCAN-5M image, DNA, and
text modalities:

- model config:
  `bioscanclip/config/model_config/for_bioscan_5m/final_experiments/image_dna_text_seed_42.yaml`;
- `model_output_name`: `image_dna_text_4gpu`;
- embedding dimension: 768;
- local checkpoint SHA256 recorded during the experiment:
  `098fb1ac663450f3d6350c18d9d26a5fb177ecf7411ccc7420833fba66a555bd`.

Download the public checkpoint:

```bash
mkdir -p ckpt/bioscan_clip/ver_1_0/bioscan_5m/image_dna_text_4gpu
wget -c \
  -O ckpt/bioscan_clip/ver_1_0/bioscan_5m/image_dna_text_4gpu/best.pth \
  https://aspis.cmpt.sfu.ca/projects/bioscan/checkpoint/for_readme/ver_1_0/bioscan_5m/image_dna_text/best.pth
```

Model construction also reads the pretrained BarcodeBERT checkpoint configured
in `bioscanclip/config/global_config.yaml`. Download it from the public CLIBD
Hugging Face repository:

```bash
huggingface-cli download bioscan-ml/clibd \
  --include \
  "ckpt/BarcodeBERT/old_checkpoints/trained_with_canada_1_5M/model_41.pth" \
  --local-dir .
```

Download BIOSCAN-5M if it is not already available on shared storage:

```bash
mkdir -p data/BIOSCAN_5M
wget -c \
  -O data/BIOSCAN_5M/BIOSCAN_5M.hdf5 \
  https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_CLIP_for_downloading/BIOSCAN_5M.hdf5
```

The BIOSCAN-5M HDF5 is approximately 190 GB.

## Reproduce the data processing

This section requires authorized access to the original CBG metadata and image
files. It can be skipped when using the released
`BIOSCAN_CBG_COI5P.hdf5`.

Define local paths:

```bash
export REPO=$PWD
export RAW=/path/to/cbg_raw_data
export WORK=/path/to/cbg_processed_data
# Required only for the optional taxonomy-overlap report in Step 2.
export BIOSCAN5M_METADATA_CSV=/path/to/BIOSCAN-5M_Dataset_v3.4.csv
mkdir -p "$WORK"
```

### 0. Optional: safely inspect and extract the source ZIP

Skip this step if the image directory is already available. The inspection
script rejects unsafe member paths and symlinks before writing batch
manifests. The processing script copies the ZIP to scratch, verifies it,
extracts and transfers one batch at a time, and verifies each transfer before
deleting that local extracted batch. It never deletes the source ZIP.

```bash
export SOURCE_ZIP=/path/to/CBG_RIG_images.zip
export SCRATCH=/path/to/local_scratch/cbg_rig
export EXTRACTED=/path/to/shared_storage/cbg_rig_images
export COMMON_ROOT=top_level_directory_reported_by_inspection

python scripts/data_processing/inspect_rig_zip_and_build_batches.py \
  --zip "$SOURCE_ZIP" \
  --state-dir "$SCRATCH/state" \
  --target-gib 150

ZIP_ARGS=(
  --source-zip "$SOURCE_ZIP"
  --local-root "$SCRATCH"
  --shared-output-root "$EXTRACTED"
  --common-root "$COMMON_ROOT"
  --margin-gib 100
)

python scripts/data_processing/process_rig_zip_batches.py "${ZIP_ARGS[@]}" copy-zip
python scripts/data_processing/process_rig_zip_batches.py "${ZIP_ARGS[@]}" \
  verify-zip --method sha256
python scripts/data_processing/process_rig_zip_batches.py "${ZIP_ARGS[@]}" run
python scripts/data_processing/process_rig_zip_batches.py "${ZIP_ARGS[@]}" final-verify
```

Use the `common_root` value in
`$SCRATCH/state/inspection_report.json`. The two scripts require Linux,
`rsync`, and sufficient local scratch space.

### 1. Convert the metadata

```bash
python scripts/data_processing/convert_rig_metadata_to_bioscan_csv.py \
  --metadata "$RAW/rig_images_metadata.txt" \
  --image-dir "$RAW/extracted_images" \
  --output "$WORK/BIOSCAN_Large_Arthropods.csv"
```

Expected output:

- 468,007 rows;
- 464,893 unique `sampleid` values;
- 468,007 unique (`sampleid`, `marker_code`) combinations;
- every metadata `sampleid` has an exact image-stem match;
- 70 additional image files have no exact `sampleid` match.

### 2. Generate optional taxonomy statistics

```bash
python scripts/stat/generate_coi5p_taxonomy_statistics.py \
  --input "$WORK/BIOSCAN_Large_Arthropods.csv" \
  --reference-csv "$BIOSCAN5M_METADATA_CSV" \
  --output-dir "$WORK/coi5p_taxonomy_statistics"
```

### 3. Resize the images

```bash
python scripts/data_processing/resize_cbg_coi5p_images.py \
  --metadata "$WORK/BIOSCAN_Large_Arthropods.csv" \
  --source-dir "$RAW/extracted_images" \
  --output-dir "$WORK/resized_images_short_edge_256" \
  --short-edge 256 \
  --workers 16
```

Expected output:

- 464,541 `COI-5P` PNG images;
- RGB, lossless PNG;
- 256-pixel short edge with proportional long-edge resize;
- no crop;
- `resize_manifest.csv` and `resize_report.json`.

### 4. Generate the HDF5

```bash
python scripts/data_processing/generate_cbg_coi5p_hdf5.py \
  --metadata "$WORK/BIOSCAN_Large_Arthropods.csv" \
  --resize-manifest "$WORK/resized_images_short_edge_256/resize_manifest.csv" \
  --resized-dir "$WORK/resized_images_short_edge_256" \
  --output "$WORK/BIOSCAN_CBG_COI5P.hdf5" \
  --batch-size 128 \
  --read-workers 8
```

Expected output: 464,530 records. Eleven `COI-5P` rows with an empty
`dna_barcode` are recorded in the excluded-record CSV.

### 5. Validate the HDF5

```bash
python scripts/data_processing/validate_cbg_coi5p_hdf5.py \
  --hdf5 "$WORK/BIOSCAN_CBG_COI5P.hdf5" \
  --group no_split \
  --metadata "$WORK/BIOSCAN_Large_Arthropods.csv" \
  --resized-dir "$WORK/resized_images_short_edge_256" \
  --sample-count 100 \
  --extract-dir "$WORK/hdf5_validation_sample_100" \
  --report "$WORK/BIOSCAN_CBG_COI5P_validation_report.json"
```

The validation should report `status: COMPLETE`.

## Reproduce Phase 1 retrieval

### Evaluation definition

- Model: CLIBD trained on BIOSCAN-5M image, DNA, and text.
- Queries: 116,161 RIG records with an available species label.
- Reference: 325,668 BIOSCAN-5M `all_keys` records.
- Retrieval modes:
  - image to image;
  - DNA to DNA;
  - image to DNA.
- Metrics: Micro and Macro Top-1, Top-5, and Top-10 accuracy at order, family,
  genus, and species.
- Similarity: inner product between L2-normalized embeddings, equivalent to
  cosine similarity.

### Smoke test

```bash
export RIG_HDF5=/path/to/BIOSCAN_CBG_COI5P.hdf5
export BIOSCAN5M_HDF5=/path/to/BIOSCAN_5M.hdf5
export CLIBD_CKPT="$PWD/ckpt/bioscan_clip/ver_1_0/bioscan_5m/image_dna_text_4gpu/best.pth"

python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
  project_root_path="$PWD" \
  bioscan_5m_data.path_to_hdf5_data="$BIOSCAN5M_HDF5" \
  cbg_retrieval.query_hdf5="$RIG_HDF5" \
  cbg_retrieval.reference_hdf5="$BIOSCAN5M_HDF5" \
  cbg_retrieval.checkpoint_path="$CLIBD_CKPT" \
  cbg_retrieval.output_dir="$PWD/retrieval_results/cbg_rig_data/image_dna_text_4gpu" \
  cbg_retrieval.reference_set=all_keys \
  cbg_retrieval.stage=all \
  cbg_retrieval.limits.query=64 \
  cbg_retrieval.limits.reference=64
```

### Complete Phase 1 run

```bash
python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
  project_root_path="$PWD" \
  bioscan_5m_data.path_to_hdf5_data="$BIOSCAN5M_HDF5" \
  cbg_retrieval.query_hdf5="$RIG_HDF5" \
  cbg_retrieval.reference_hdf5="$BIOSCAN5M_HDF5" \
  cbg_retrieval.checkpoint_path="$CLIBD_CKPT" \
  cbg_retrieval.output_dir="$PWD/retrieval_results/cbg_rig_data/image_dna_text_4gpu" \
  cbg_retrieval.reference_set=all_keys \
  cbg_retrieval.stage=all
```

The stages may also be run separately:

```bash
for STAGE in prepare embed index search evaluate; do
  python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
    project_root_path="$PWD" \
    bioscan_5m_data.path_to_hdf5_data="$BIOSCAN5M_HDF5" \
    cbg_retrieval.query_hdf5="$RIG_HDF5" \
    cbg_retrieval.reference_hdf5="$BIOSCAN5M_HDF5" \
    cbg_retrieval.checkpoint_path="$CLIBD_CKPT" \
    cbg_retrieval.output_dir="$PWD/retrieval_results/cbg_rig_data/image_dna_text_4gpu" \
    cbg_retrieval.reference_set=all_keys \
    cbg_retrieval.stage="$STAGE"
done
```

The pipeline resumes complete caches by default. Do not enable an overwrite
option unless the corresponding artifact should be regenerated.

## Phase 1 results

Top-1 retrieval accuracy:

| Retrieval | Micro Order | Micro Family | Micro Genus | Micro Species | Macro Order | Macro Family | Macro Genus | Macro Species |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Image to image | 82.24 | 32.03 | 5.97 | 2.92 | 14.60 | 11.53 | 3.02 | 1.23 |
| DNA to DNA | 52.00 | 26.12 | 19.48 | 18.25 | 21.86 | 17.26 | 9.23 | 6.54 |
| Image to DNA | 25.60 | 1.69 | 0.16 | 0.10 | 3.71 | 0.23 | 0.02 | <0.01 |

Main observations:

- image-to-image retrieval is strong at order but weak at genus and species;
- DNA-to-DNA gives the best fine-grained retrieval;
- image-to-DNA is weak below order;
- Macro accuracy is consistently below Micro accuracy, indicating weaker
  performance on less common labels;
- only 45.99% of the species-labelled queries have a species label present in
  BIOSCAN-5M `all_keys`, so the reference-set coverage limits exact species
  retrieval.

Raw metrics:

```text
retrieval_results/cbg_rig_data/image_dna_text_4gpu/
└── full/all_keys/metrics/retrieval_metrics.json
```

## Next steps

### Phase 2: full BIOSCAN-5M reference

Use the configured `full_unique` reference, which is a deduplicated union of
the relevant BIOSCAN-5M groups:

```bash
python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
  project_root_path="$PWD" \
  bioscan_5m_data.path_to_hdf5_data="$BIOSCAN5M_HDF5" \
  cbg_retrieval.query_hdf5="$RIG_HDF5" \
  cbg_retrieval.reference_hdf5="$BIOSCAN5M_HDF5" \
  cbg_retrieval.checkpoint_path="$CLIBD_CKPT" \
  cbg_retrieval.output_dir="$PWD/retrieval_results/cbg_rig_data/image_dna_text_4gpu" \
  cbg_retrieval.reference_set=full_unique \
  cbg_retrieval.stage=prepare
```

The expected manifest contains 5,150,850 unique `processid` values. Inspect
that manifest before beginning the expensive embedding stage.

Two `float32` reference modalities at this size require approximately 31.6 GB
for embeddings alone; the indices require substantial additional storage.
The current `full_unique` configuration uses a CPU IVF-Flat index.

Important implementation note: runs with `limits.query` or
`limits.reference` are written under a smoke-test namespace. Their full
reference embedding/index artifacts are not automatically reused by a later
unlimited run. Before an expensive Phase 2 pilot, either fix the cache
namespace so that a full reference cache is shared across query limits or run
the full namespace directly.

### Phase 3: RIG reference

Create disjoint RIG query and key sets and repeat the same three retrieval
modes. The split should:

- contain no shared `sampleid` or image between query and key;
- define how classes with one or very few samples are handled;
- preserve enough labelled queries for Micro and Macro evaluation;
- record whether identical DNA barcodes or DNA BINs are allowed across the
  query/key boundary;
- save the split manifest and random seed.

This phase separates image-domain transfer from BIOSCAN-5M reference coverage:
if retrieval improves strongly with RIG keys, the primary limitation is likely
the reference/domain mismatch rather than a complete failure of the learned
representation.

### Model decision

Fine-tune or retrain CLIBD with RIG data only after Phases 2 and 3 if:

- overlapping BIOSCAN-5M labels still retrieve poorly;
- adding full BIOSCAN-5M instances does not improve retrieval;
- an in-domain RIG key set is also insufficient;
- image-to-DNA alignment remains weak after reference coverage is addressed.

## Verification performed

The reorganized code was tested on 2026-07-28 in the
`CLIBD-hyperbolic` environment:

- all new and modified Python files passed `py_compile`;
- every new command-line script passed `--help`;
- the complete 468,007-row raw metadata passed converter dry-run checks;
- a two-batch ZIP fixture passed inspection, SHA256 verification, extraction,
  rsync transfer, cleanup, and final file/byte verification;
- eight real RIG images passed resize and HDF5-generation smoke tests;
- the final 464,530-record HDF5 passed full metadata comparison and sampled
  PNG, source-byte, CLIBD image-transform, and DNA-tokenizer checks;
- taxonomy statistics and both PNG/PDF histograms were generated from a
  2,000-row fixture;
- a fresh 8-query/64-reference run completed all retrieval stages:
  `prepare`, `embed`, `index`, `search`, and `evaluate`;
- `git diff --check` reported no whitespace errors.

These tests used the current working tree. Repeat the documented retrieval
smoke test from a clean clone after the changes are committed and pushed.

## Handoff checklist

Before sending this README to a collaborator:

- [ ] Confirm CBG redistribution permission and license.
- [ ] Commit and push the retrieval runner, config, CLIBD helper changes, and
      this documentation to the public repository.
- [ ] Publish the approved HDF5 and reports under the HTTPS base.
- [ ] Generate `SHA256SUMS` and `MANIFEST.tsv`.
- [ ] Verify each direct HTTPS file URL returns HTTP 200.
- [ ] Reproduce the 64-query/64-reference smoke test from a clean clone.
- [ ] Record the public CLIBD commit SHA used for all future results.
