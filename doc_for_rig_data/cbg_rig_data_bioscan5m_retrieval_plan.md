# CBG RIG Data × BIOSCAN-5M CLIBD Retrieval Plan

Last updated: 2026-07-18

## 1. Goal

Use a CLIBD model trained on BIOSCAN-5M, without initially retraining or
fine-tuning it, to evaluate how well the new CBG RIG dataset can retrieve
biologically relevant references from BIOSCAN-5M.

We will compare two main reference strategies:

1. **BIOSCAN-5M `all_keys` as the reference**
   - Standard key-set retrieval.
   - Smaller, faster, and easier to compare with existing CLIBD evaluation.

2. **A large or full BIOSCAN-5M instance collection as the reference**
   - Instance-level retrieval against millions of historical specimens.
   - More likely to contain a visually or genetically close specimen.
   - More computationally expensive and more sensitive to class-frequency
     imbalance.

The results will help decide whether we should:

- continue using the existing BIOSCAN-5M key set;
- build a new key set that includes representative CBG samples;
- fine-tune or retrain CLIBD using the CBG data.

## 2. Current data and artifacts

### CBG query dataset

- HDF5:
  `/local-scratch/research/new_data_from_cbg/BIOSCAN_CBG_COI5P.hdf5`
- Group: `no_split`
- COI-5P records in the HDF5: **464,530**
- First-round query records with a non-empty species label: **116,161**
- Of these, **116,158** have all four taxonomy ranks populated; the remaining
  three are retained and excluded only from the missing rank's denominator.
- Marker: `COI-5P`
- Every included row has:
  - a non-empty DNA barcode;
  - a valid resized image;
  - unique `sampleid`;
  - unique `processid`.

The CBG HDF5 contains image bytes, DNA barcode, DNA BIN, and the four taxonomy
ranks `order`, `family`, `genus`, and `species`.

### BIOSCAN-5M reference dataset

Current BIOSCAN-5M HDF5 used by the implementation:

`/local-scratch/research/projects/data_from_14/BIOSCAN_5M.hdf5`

Important group sizes observed on 2026-07-18:

| Group | Rows |
|---|---:|
| `all_keys` | 325,668 |
| `no_split_and_seen_train` | 4,966,959 |
| `other_heldout` | 76,590 |
| `seen_keys` | 289,203 |
| `unseen_keys` | 36,465 |
| `val_seen` | 14,757 |
| `val_unseen` | 8,819 |
| `test_seen` | 39,373 |
| `test_unseen` | 7,887 |

These groups are **not disjoint**. For example, `seen_keys`, `train_keys`, and
part of `all_keys` describe overlapping records. Therefore, a “full
BIOSCAN-5M reference” must not be constructed by blindly concatenating every
HDF5 group. It must be deduplicated using `processid` (and checked with
`sampleid` as a secondary identifier).

## 3. Model selection

The selected baseline is the best original CLIBD BIOSCAN-5M model:

- config:
  `bioscanclip/config/model_config/for_bioscan_5m/final_experiments/image_dna_text_seed_42.yaml`;
- `model_output_name`: `image_dna_text_4gpu`;
- checkpoint:
  `ckpt/bioscan_clip/ver_1_0/bioscan_5m/image_dna_text_4gpu/best.pth`;
- checkpoint SHA256:
  `098fb1ac663450f3d6350c18d9d26a5fb177ecf7411ccc7420833fba66a555bd`;
- embedding dimension: **768**;
- similarity: inner product on L2-normalized embeddings, equivalent to cosine
  similarity.

The same checkpoint and preprocessing must be used for both CBG queries and
BIOSCAN-5M references. Existing reference embeddings may only be reused when
they were produced by exactly the same checkpoint and preprocessing.

## 4. Reference variants

### R0: BIOSCAN-5M `all_keys`

- Reference size: 325,668 rows.
- Purpose: standard key-set baseline.
- Advantages:
  - fastest reference to encode and search;
  - aligns most closely with the existing CLIBD retrieval setup;
  - less dominated by the number of specimens per class.
- Limitation:
  - a class may have very few representatives;
  - the nearest real specimen may not be included in the key set.

This should be the first complete experiment.

### R1: Deduplicated full BIOSCAN-5M union

The second reference is the union of the relevant BIOSCAN-5M groups,
deduplicated by `processid`. Its confirmed size is **5,150,850** unique
records. The canonical group priority is:

1. `no_split_and_seen_train`;
2. `unseen_keys`;
3. `other_heldout`;
4. `test_seen`;
5. `test_unseen`;
6. `val_seen`;
7. `val_unseen`.

This consolidates the earlier R1/R2 ideas: `no_split_and_seen_train` is not a
separate main experiment because the deduplicated union adds only 183,891
records and provides the more complete definition.

## 5. Retrieval pairs

For both R0 and R1, run the following confirmed retrieval directions:

| Query modality | Reference modality | Question |
|---|---|---|
| CBG image | BIOSCAN image | Can a new RIG image find a similar BIOSCAN specimen? |
| CBG image | BIOSCAN DNA | Does the shared CLIBD space align an image with a genetically relevant specimen? |
| CBG DNA | BIOSCAN DNA | Does barcode embedding retrieve genetically/taxonomically related records? |

Optional key-set-only extensions:

- CBG image → BIOSCAN taxonomy text;
- CBG DNA → BIOSCAN taxonomy text.

For the full instance reference, image and DNA reference embeddings are the
main priority. Repeating identical taxonomy text for millions of specimens is
not necessary for the first full-reference experiment.

## 6. Embedding extraction

### CBG queries

Extract and cache:

- `encoded_image_feature`;
- `encoded_dna_feature`;
- `sampleid`, `processid`, DNA BIN, and taxonomy labels.

Process the 116,161 species-labeled rows in batches and write embeddings incrementally. The
extraction must be resumable and must never hold the entire image dataset in
memory.

### BIOSCAN-5M references

For each reference variant, either:

- reuse checkpoint-matched cached embeddings; or
- extract reference embeddings once and cache them.

Reference artifacts must include a manifest that maps every embedding row back
to:

- HDF5 group;
- HDF5 row index;
- `processid`;
- `sampleid`;
- `image_file`;
- DNA BIN;
- order/family/genus/species.

## 7. Runtime and memory considerations

For 768-dimensional `float32` embeddings:

| Data | Approximate embedding memory per modality |
|---|---:|
| 325,668 `all_keys` | 1.00 GB |
| 4,966,959 large-reference rows | 15.26 GB |
| 116,161 species-labeled CBG queries | 0.36 GB |

Using both image and DNA embeddings doubles the reference storage. Additional
memory is needed for FAISS index structures, query batches, IDs, and labels.

A naive matrix comparison between all 116,161 first-round CBG queries and
approximately 5 million references would involve about 580 billion query-reference
pairs. It should **not** be implemented as one dense similarity matrix.

Required controls:

- use FAISS;
- search CBG queries in batches;
- save only Top-K indices and scores;
- avoid writing a dense query-by-reference matrix;
- cache reference indices for reuse;
- use GPU indices only when GPU memory is sufficient;
- otherwise use a CPU index or sharded GPU search.

### Measured batch sizes on the RTX 2080 Ti

Measured on 2026-07-18 with the selected CLIBD checkpoint and real CBG rows:

- Embedding batch 680 completed once, while 688 produced CUDA OOM.
- In a multi-batch stability test, 512 completed and 640 produced CUDA OOM.
- For the same 2,048 samples, batch sizes 64/128/256/512 took
  26.24/26.51/27.23/28.41 seconds respectively, including the same model-load
  overhead. Larger embedding batches did not improve throughput; keep
  `embedding.batch_size=64`.
- A GPU `IndexFlatIP` with the R0 shape (325,668 × 768) sustained about 20,000
  queries/s for query batches from 1,024 through all 116,161 queries.
  Increasing the FAISS batch did not improve throughput and would enlarge the
  following Arrow/Parquet metadata batch; keep
  `retrieval.query_batch_size=1024`.

### Numerical reproducibility of embedding extraction

The inference path uses `model.eval()`, deterministic evaluation transforms,
`shuffle=False`, and a fixed manifest order. Repeated extraction is therefore
semantically reproducible, but the current code does not promise bitwise
identity across different GPU batch shapes. Empirical comparisons of identical
samples extracted in independent processes and different batch sizes found:

- image maximum absolute difference: approximately 1.7e-7 to 2.7e-7;
- image minimum cosine similarity: 0.999999881;
- DNA was bitwise identical for tested batch sizes of 64 and above, while
  batch 4 versus batch 64 differed by at most approximately 8.6e-7.

Cache and reuse the produced embeddings for all reported retrieval comparisons.
If bitwise-identical re-extraction becomes a formal requirement, fix the batch
size and enable PyTorch/CUDA deterministic algorithms before generating a new
complete cache; changing batch shape may still change floating-point reduction
order.

### Index strategy

For R0:

- start with exact `IndexFlatIP` after applying the model's required
  normalization;
- this provides a clean and reproducible baseline.

For R1:

1. test exact search on a small query subset;
2. if full exact search is too slow, use a FAISS approximate index such as
   IVF-Flat or IVF-PQ;
3. measure approximate-index recall against exact search on a fixed validation
   subset;
4. record the index parameters in every result artifact.

The full-reference experiment may take substantially longer to encode and
search, but the expensive reference embedding and index construction are
one-time costs. Once cached, multiple CBG evaluations can reuse them.

## 8. Staged execution plan

### Phase A: Pipeline smoke test — completed

- The final checkpoint/config is fixed above.
- An initial 8-query × 64-reference end-to-end test has completed.
- All three modality pairs completed.
- Verify:
  - embedding shapes and finite values;
  - normalization/similarity behavior;
  - Top-K IDs map to correct metadata;
  - output can be resumed;
  - qualitative retrieval images can be generated.

### Phase B: Complete `all_keys` baseline

- Encode all 116,161 species-labeled CBG queries.
- Build or load the 325,668-row `all_keys` index.
- Run Top-1, Top-5, Top-10 retrieval.
- Compute all metrics and qualitative examples.

This is the first main result.

### Phase C: Large-reference pilot

- Keep the complete R1 reference.
- Search only 1,000, then 10,000 CBG queries.
- Measure:
  - reference index build time;
  - query throughput;
  - CPU/GPU memory;
  - disk use;
  - exact versus approximate agreement.

Use this pilot to choose the full-run index parameters.

### Phase D: Complete large-reference retrieval

- Search all 116,161 species-labeled CBG queries against R1.
- Store Top-K IDs and similarities only.
- Run the same metrics as R0.
- Compare accuracy, confidence, runtime, and class-frequency effects.

## 9. Evaluation

### Retrieval outputs

For each query and retrieval pair, save at least:

- query `sampleid` and `processid`;
- query labels;
- Top-1/5/10 reference IDs;
- similarity scores;
- reference labels;
- reference HDF5 group and row index.

Prefer chunked Parquet or HDF5 outputs rather than one very large JSON file.

### Taxonomy metrics

Compute at:

- order;
- family;
- genus;
- species;
- DNA BIN, where available.

Metrics:

- Micro Top-1/Top-5/Top-10;
- Macro Top-1/Top-5/Top-10;
- per-class accuracy;
- similarity of the first correct result;
- Top-1 versus Top-2 similarity margin.

Rows with `not_classified` at a rank are excluded from that rank's accuracy
denominator but remain available for ranks where labels are present.

### Known versus novel subsets

Report separately:

- CBG classes present in BIOSCAN-5M;
- CBG classes absent from BIOSCAN-5M;
- CBG DNA BINs present in BIOSCAN-5M;
- CBG DNA BINs absent from BIOSCAN-5M.

For novel classes, ordinary exact-label accuracy is not sufficient. Analyze:

- nearest-neighbor taxonomy;
- similarity-score distribution;
- distance/margin relative to known classes;
- whether retrieval stays within the correct higher rank;
- potential out-of-reference or novelty thresholds.

## 10. Important full-reference bias analysis

The full BIOSCAN-5M reference has a strongly skewed number of specimens per
class. Frequent classes receive more chances to produce a very close
nearest-neighbor. Therefore, R0 and R1 are not directly comparable without
additional analysis.

For R1, report at least:

1. **Raw instance nearest neighbor**
   - Standard nearest specimen.

2. **Top-K class voting**
   - Aggregate the labels of the nearest instances.

3. **Class-balanced score**
   - For example, maximum similarity per class, or a controlled number of
     reference specimens per class.

4. **Accuracy versus reference class size**
   - Determine whether performance gains are mostly driven by classes with
     many reference specimens.

Also consider a size-controlled reference, such as sampling at most N
specimens per species or DNA BIN. This separates the benefit of broader
coverage from the benefit of simply having more examples for common classes.

## 10.1 Expected image-domain gap

Most BIOSCAN-5M images were captured in a petri-dish-style setup, whereas the
new RIG images use a visibly different and generally cleaner acquisition
setup. This domain shift is expected to affect Image → Image and Image → DNA
more strongly. DNA → DNA is therefore the important domain-invariant control:
if DNA retrieval is strong while image-based retrieval is weak, the image
domain gap is the leading explanation rather than reference taxonomy coverage.

## 11. Qualitative analysis

Generate retrieval panels containing:

- CBG query image;
- Top-1 through Top-5 BIOSCAN images;
- query and reference taxonomy;
- DNA BIN;
- similarity score;
- whether each taxonomy rank matches.

Include examples from:

- correct species retrieval;
- correct genus but wrong species;
- correct family but wrong genus;
- complete mismatch;
- novel species or DNA BIN;
- high-confidence failure;
- low-confidence query;
- cases where R1 succeeds but R0 fails;
- cases where R0 succeeds but R1 fails.

## 12. Comparison table

The final report should contain a table similar to:

| Reference | Query → Key | Rank | Micro Top-1 | Macro Top-1 | Top-5 | Runtime | Index size |
|---|---|---|---:|---:|---:|---:|---:|
| `all_keys` | Image → Image | species | TBD | TBD | TBD | TBD | TBD |
| `all_keys` | DNA → DNA | species | TBD | TBD | TBD | TBD | TBD |
| Full/R1 | Image → Image | species | TBD | TBD | TBD | TBD | TBD |
| Full/R1 | DNA → DNA | species | TBD | TBD | TBD | TBD | TBD |

The report should also compare known/novel subsets and performance as a
function of reference class size.

## 13. Decision criteria

### Continue using BIOSCAN-5M `all_keys` if:

- accuracy is already strong;
- similarity margins are reliable;
- full-reference retrieval adds little benefit;
- runtime and storage simplicity are important.

### Create or extend a CBG key set if:

- performance is strong for overlapping classes but poor for CBG-only classes;
- CBG contains substantial new genus/species/DNA BIN diversity;
- a small number of representative CBG samples closes most retrieval gaps.

### Fine-tune or retrain CLIBD if:

- overlapping classes also retrieve poorly;
- cross-modal alignment is weak;
- RIG images exhibit a meaningful domain shift;
- adding more reference instances does not solve the failures.

## 14. Implemented code and artifacts

The implementation deliberately uses one stage-driven entrypoint rather than
several scripts:

- runner:
  `scripts/retrieval/run_cbg_bioscan5m_retrieval.py`;
- config:
  `bioscanclip/config/cbg_retrieval.yaml`;
- reusable streaming inference:
  `bioscanclip/epoch/inference_epoch.py::iter_feature_and_label_batches`.

Artifact root:

`retrieval_results/cbg_rig_data/<model_output_name>/`

Suggested structure:

```text
retrieval_results/cbg_rig_data/<model_output_name>/
├── manifests/
├── full/
│   ├── shared/embeddings/query.hdf5
│   ├── all_keys/
│   │   ├── embeddings/
│   │   ├── indices/
│   │   ├── predictions/
│   │   └── metrics/
│   └── full_unique/
└── smoke_q<N>_r<N>/
```

Each run should save a machine-readable config recording:

- checkpoint path and checksum;
- model config and `model_output_name`;
- query/reference definitions;
- modalities;
- embedding dimension and dtype;
- normalization and similarity rule;
- FAISS index type and parameters;
- Top-K;
- random seed;
- runtime and hardware information.

## 15. Execution commands and immediate next steps

Run from the original `clibd` repository with
`conda activate CLIBD-hyperbolic`.

Small end-to-end test:

```bash
python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
  cbg_retrieval.stage=all \
  cbg_retrieval.limits.query=1000 \
  cbg_retrieval.limits.reference=10000
```

Complete `all_keys` baseline:

```bash
python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
  cbg_retrieval.stage=all \
  cbg_retrieval.reference_set=all_keys
```

Prepare the full unique reference manifest without starting embedding:

```bash
python -m scripts.retrieval.run_cbg_bioscan5m_retrieval \
  cbg_retrieval.stage=prepare \
  cbg_retrieval.reference_set=full_unique
```

Then run a 1,000/10,000-query pilot against `full_unique`, measure throughput,
and tune `ivf_nlist`/`ivf_nprobe` before the complete run. All stages are
independently resumable. Every embedding directory and final result directory
also contains the complete resolved Hydra `config.yaml`, recording the model,
checkpoint path, data selection, reference definition, batch size, index
parameters, retrieval pairs, and Top-K settings used to produce the artifact.

This sequence gives an early, interpretable result while keeping the
full-reference experiment feasible and reproducible.
