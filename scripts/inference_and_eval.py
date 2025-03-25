import io
import json
import os
import random
from collections import Counter, defaultdict

import h5py
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import torch
from PIL import Image
from omegaconf import DictConfig
from sklearn.metrics import silhouette_samples
from umap import UMAP

from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits
from bioscanclip.util.util import (
    categorical_cmap,
    inference_and_print_result,
    get_features_and_label,
    make_prediction,
    All_TYPE_OF_FEATURES_OF_KEY,
)

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"


def get_all_unique_species_from_dataloader(dataloader):
    all_species = []

    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch["species"]
    all_species = list(set(all_species))
    return all_species


def save_prediction(pred_dict, gt_dict, json_path):
    data = {"gt_labels": gt_dict, "pred_labels": pred_dict}

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)


def load_from_json(path):
    with open(path, "r") as file:
        data = json.load(file)

    pred_list = data["pred_labels"]
    gt_list = data["gt_labels"]
    correct_predictions = sum(1 for true, predicted in zip(gt_list, pred_list) if true == predicted)
    total_samples = len(gt_list)
    eval_bioscan_1m_acc = correct_predictions / total_samples
    return pred_list, gt_list, eval_bioscan_1m_acc


def show_distribution(list):
    counts = Counter(list)

    # Get values and corresponding counts, sorted by count in descending order
    sorted_values, sorted_occurrences = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # Create bar plot with log-scaled y-axis and raw counts
    plt.bar(sorted_values, sorted_occurrences)
    plt.yscale("log")  # Set y-axis to a logarithmic scale

    # Add labels and title
    plt.title("Distribution of BioScan-1M validation data")

    plt.xticks(rotation=30)
    # Display the raw count on top of each bar
    for value, occurrence in zip(sorted_values, sorted_occurrences):
        plt.text(value, occurrence, f"{occurrence}", ha="center", va="bottom")

    # Show the plot
    plt.show()


def get_labels(my_list):
    counts = Counter(my_list)

    # Sort values by count in descending order
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    list_of_labels = []
    for i in sorted_counts:
        list_of_labels.append(i[0])

    return list_of_labels


def generate_embedding_plot(args, image_features, dna_features, language_features, gt_labels, num_classifications=10):
    def get_language_feature_mapping(language_features):
        if language_features is None:
            return None, None, None
        return np.unique(language_features, axis=0, return_index=True, return_inverse=True)

    levels = ["order", "family", "genus", "species"]

    unique_lang_features, lang_indices, inv_indices = get_language_feature_mapping(language_features)
    # compute 2D embeddings
    umap_2d = UMAP(n_components=2, init="random", random_state=0, min_dist=0.5, metric="cosine")
    features = []
    feature_names = []
    num_samples = image_features.shape[0] if image_features is not None else dna_features.shape[0]
    for name, feature in [("image", image_features), ("dna", dna_features), ("text", unique_lang_features)]:
        if feature is not None:
            features.append(feature)
            feature_names.append(name)
    if not features:
        raise ValueError("No image, DNA, or language features provided.")
    proj_2d = umap_2d.fit_transform(np.concatenate(features, axis=0))

    all_indices = []
    for level_idx, level in enumerate(levels):
        # apply filter to points
        if level_idx > 0 and levels[level_idx - 1] in args.inference_and_eval_setting.embeddings_filters:
            prev_level = levels[level_idx - 1]
            indices = [
                i
                for i in range(len(gt_labels))
                if gt_labels[i][prev_level] == args.inference_and_eval_setting.embeddings_filters[prev_level]
            ]
            # print(np.unique([gt_labels[i][prev_level] for i in indices]))
        else:
            indices = [i for i in range(num_samples)]

        all_indices.append(indices)
        # filter small classes
        taxon_counts = dict(zip(*np.unique([gt_labels[i][level] for i in indices], return_counts=True)))
        taxons = sorted(taxon_counts.keys(), key=lambda x: taxon_counts[x], reverse=True)[:num_classifications]
        indices = [i for i in indices if gt_labels[i][level] in taxons]
        random.shuffle(indices)

        number_of_sample = len(indices)

        gt_list = [gt_labels[i][level] for i in indices]
        unique_values, unique_counts = np.unique(gt_list, return_counts=True)
        idx_sorted = np.argsort(unique_counts)[::-1]
        level_order = unique_values[idx_sorted]
        level_order = [f"{level_name}-{type_}" for level_name in level_order for type_ in feature_names]
        count_unique = len(unique_values)

        gt_list_full = []
        full_indices = []
        if image_features is not None:
            gt_list_full.extend([f"{gt}-image" for gt in gt_list])
            full_indices.extend(indices)
        if dna_features is not None:
            gt_list_full.extend([f"{gt}-dna" for gt in gt_list])
            full_indices.extend([i + number_of_sample for i in indices])
        if language_features is not None:
            gt_list_full.extend([f"{gt_labels[i][level]}-text" for i in lang_indices[np.unique(inv_indices[indices])]])
            full_indices.extend((np.unique(inv_indices[indices]) + len(full_indices)).tolist())
        proj_2d_selected = proj_2d[full_indices]

        # colors = [matplotlib.colors.to_rgb(col) for col in px.colors.qualitative.Dark24][:count_unique]
        colors = [matplotlib.colors.to_hex(col) for col in categorical_cmap(count_unique, len(feature_names)).colors]

        fig_2d = px.scatter(
            proj_2d_selected,
            x=0,
            y=1,
            color=gt_list_full,
            opacity=1.0,
            labels={"color": level},
            color_discrete_sequence=colors,
            size_max=1,
            # title=f"Embedding plot for image and DNA features with {level} labels",
            category_orders={"color": level_order},
        )

        # fix legend to not render symbol
        region_lst = set()
        for trace in fig_2d["data"]:
            trace["name"] = trace["name"].split(",")[0]

            if trace["name"] not in region_lst:
                trace["showlegend"] = True
                region_lst.add(trace["name"])
            else:
                trace["showlegend"] = False

        fig_2d.update_layout(
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                title="",
                itemsizing="constant",
                traceorder="normal",
                itemwidth=30,
                tracegroupgap=5,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        folder_path = os.path.join(args.project_root_path, f"{PLOT_FOLDER}/{args.model_config.model_output_name}")
        os.makedirs(folder_path, exist_ok=True)
        # fig_3d.update_traces(marker_size=5)
        fig_2d.write_html(os.path.join(folder_path, f"{level}_2d.html"))
        plotly.io.write_image(
            fig_2d, os.path.join(folder_path, f"{level}_2d.pdf"), format="pdf", height=3200, width=3200
        )
        # plotly.io.write_image(fig_2d, os.path.join(folder_path, f"{level}_2d.pdf"), format="pdf", height=1200, width=1600)
        print(f"Saved {level} plot in {os.path.join(folder_path, f'{level}_2d.html')}")
        # fig_3d.write_html(os.path.join(folder_path, f'{level}_3d.html'))
        # fig_2d.show()
        plotly.io.write_image(
            fig_2d, os.path.join(folder_path, f"{level}_2d.pdf"), format="pdf", height=3200, width=3200
        )
        # fig_3d.show()


def retrieve_images(
    args,
    name,
    query_dict,
    keys_dict,
    query_keys,
    query_data,
    key_data,
    num_queries=5,
    max_k=5,
    taxon="order",
    independent=True,
    seed=None,
    load_cached_results=False,
    is_bioscan5m: bool = False,
):
    """
    for X in {image, DNA}:
        for _ in range(num_queries):
            1 random input image as the query (per taxon)
            {num_retrieved} retrieved images as the key using the closest X embedding
    """

    def load_image_from_h5(data, idx):
        """Load image file from HDF file"""
        enc_length = data["image_mask"][idx]
        image_enc_padded = data["image"][idx].astype(np.uint8)
        image_enc = image_enc_padded[:enc_length]
        image = Image.open(io.BytesIO(image_enc))
        return image.resize((256, 256))

    # drawing parameters
    # colors = [matplotlib.colors.to_hex(col) for col in categorical_cmap(4, 1).colors]
    colors = ["green", "greenyellow", "gold", "orange"]
    error_color = "tab:red"
    correct_cls_colors = {taxon: col for taxon, col in zip(["species", "genus", "family", "order"], colors)}
    display_map = {"encoded_dna_feature": "DNA", "encoded_image_feature": "Image", "encoded_language_feature": "Text"}
    linewidth = 6

    # setup directory
    folder_path = os.path.join(
        args.project_root_path,
        RETRIEVAL_FOLDER,
        args.model_config.model_output_name,
        name,
    )
    os.makedirs(folder_path, exist_ok=True)

    # select queries
    rng = np.random.default_rng(seed)
    query_indices_by_taxon = defaultdict(list)
    for i, label in enumerate(query_dict["label_list"]):
        query_indices_by_taxon[label[taxon]].append(i)
    taxon_to_sample = rng.choice(
        list(query_indices_by_taxon.keys()), size=num_queries, replace=len(query_indices_by_taxon) < num_queries
    )
    query_indices = [rng.choice(query_indices_by_taxon[taxon], size=1, replace=False)[0] for taxon in taxon_to_sample]

    # retrieve with image keys
    keys_label = keys_dict["label_list"]

    # initialize retrieval results
    retrieved_images_json_path = os.path.join(folder_path, "retrieved_images.json")
    if load_cached_results and os.path.exists(retrieved_images_json_path):
        with open(retrieved_images_json_path, "r") as json_file:
            retrieval_results = json.load(json_file)
        loaded_cached_results = True
        print(f"Loaded cached retrieval results from {retrieved_images_json_path}")
    else:
        retrieval_results = []
        for query_index in query_indices:
            retrieval_results.append(
                {
                    "query": {
                        "file_name": query_dict["processed_id_list"][query_index],
                        "taxonomy": query_dict["label_list"][query_index],
                    },
                    "results": [],
                }
            )
        loaded_cached_results = False

    # use these to reverse lookup the indices for each image file later
    if is_bioscan5m:
        query_image_file_map = {
            filename.decode("utf-8").partition(".")[0]: j for j, filename in enumerate(query_data["image_file"])
        }
        key_image_file_map = {
            filename.decode("utf-8").partition(".")[0]: j for j, filename in enumerate(key_data["image_file"])
        }
    else:
        query_image_file_map = {filename.decode("utf-8"): j for j, filename in enumerate(query_data["image_file"])}
        key_image_file_map = {filename.decode("utf-8"): j for j, filename in enumerate(key_data["image_file"])}

    # making one large file: create figure upfront and add the query images to start in the first column
    if not independent:
        width_ratios = [1]
        for i in range(len(query_keys)):
            width_ratios.append(0.1)
            width_ratios.extend([1 for _ in range(max_k)])
        fig, axes = plt.subplots(
            nrows=num_queries,
            ncols=(max_k + 1) * len(query_keys) + 1,
            figsize=(22, 13),
            gridspec_kw={"width_ratios": width_ratios, "hspace": 0.05, "wspace": 0.05},
        )
        for i, pred_dict in enumerate(retrieval_results):
            # save query
            query_file_name = pred_dict["query"]["file_name"]
            image_idx = query_image_file_map[query_file_name]
            image = load_image_from_h5(query_data, image_idx)
            axes[i, 0].imshow(image)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 0].set_ylabel(
                "\n".join(pred_dict["query"]["taxonomy"]["species"].split()),
                rotation="horizontal",
                ha="right",
                fontsize=20,
            )
            plt.setp(axes[i, 0].spines.values(), color=None)

            axes[0, 0].set_xlabel("Original", loc="left", fontsize=24, labelpad=10)
            axes[0, 0].xaxis.set_label_position("top")
        last_col_idx = 2

    for query_key_idx, (query_feature_type, key_feature_type) in enumerate(query_keys):
        queries_feature = query_dict[query_feature_type]
        keys_feature = keys_dict[key_feature_type]

        # select random queries
        queries_feature = queries_feature[query_indices, :]

        if keys_feature is None or queries_feature is None or keys_feature.shape[-1] != queries_feature.shape[-1]:
            continue

        # retrieve keys for each query
        _, indices_per_query = make_prediction(
            queries_feature, keys_feature, keys_label, with_indices=True, max_k=max_k
        )

        if not loaded_cached_results:
            for idx, indices_per_query in enumerate(indices_per_query):
                retrieval_results[idx]["results"].append(
                    {"query_type": query_feature_type, "key_type": key_feature_type, "predictions": []}
                )
                for retrieved_index in indices_per_query:
                    retrieval_results[idx]["results"][query_key_idx]["predictions"].append(
                        {
                            "file_name": keys_dict["processed_id_list"][retrieved_index],
                            "taxonomy": keys_dict["label_list"][retrieved_index],
                        }
                    )

        # save out images
        if independent:
            width_ratios = [1, 0.1, *[1 for _ in range(max_k)]]
            fig, axes = plt.subplots(
                nrows=num_queries,
                ncols=max_k + 2,
                figsize=(22, 14.5),
                gridspec_kw={"width_ratios": width_ratios, "hspace": 0.05, "wspace": 0.05},
            )
            axes[0, 0].set_xlabel("Original", loc="left", fontsize=24, labelpad=10)
            axes[0, 0].xaxis.set_label_position("top")

        # iterate through queries
        for i, pred_dict in enumerate(retrieval_results):
            # add queries to each image
            if independent:
                # setup figure if we are making them independently
                query_file_name = pred_dict["query"]["file_name"]
                image_idx = query_image_file_map[query_file_name]
                image = load_image_from_h5(query_data, image_idx)
                axes[i, 0].imshow(image)
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
                axes[i, 0].set_ylabel(
                    "\n".join(pred_dict["query"]["taxonomy"]["species"].split()),
                    rotation="horizontal",
                    ha="right",
                    fontsize=20,
                )
                plt.setp(axes[i, 0].spines.values(), color=None)

                axes[i, 1].axis("off")

                last_col_idx = 2

            # iterate through retrieved results
            for j, pred in enumerate(pred_dict["results"][query_key_idx]["predictions"]):
                key_file_name = pred["file_name"]
                image_idx = key_image_file_map[key_file_name]
                image = load_image_from_h5(key_data, image_idx)
                axes[i, last_col_idx + j].imshow(image)  # subplot in col 1 is invisible
                if i != 0 or j != 0:
                    axes[i, last_col_idx + j].axis("off")
                else:
                    axes[i, last_col_idx].set_xticks([])
                    axes[i, last_col_idx].set_yticks([])
                    plt.setp(axes[i, last_col_idx].spines.values(), color=None)

                # add box around images which were correct predictions
                for taxon in ["species", "genus", "family", "order"]:
                    if pred_dict["query"]["taxonomy"][taxon] == pred["taxonomy"][taxon]:
                        bbox = axes[i, last_col_idx + j].get_tightbbox(fig.canvas.get_renderer())
                        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
                        fig.add_artist(
                            plt.Rectangle(
                                (x0, y0),
                                width,
                                height,
                                edgecolor=correct_cls_colors[taxon],
                                linewidth=linewidth,
                                fill=False,
                            )
                        )
                        break
                else:
                    bbox = axes[i, last_col_idx + j].get_tightbbox(fig.canvas.get_renderer())
                    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
                    fig.add_artist(
                        plt.Rectangle(
                            (x0, y0),
                            width,
                            height,
                            edgecolor=error_color,
                            linewidth=linewidth,
                            fill=False,
                        )
                    )

            axes[i, last_col_idx - 1].axis("off")

        axes[0, last_col_idx].set_xlabel(
            f"{display_map[query_feature_type]} to {display_map[key_feature_type]}",
            loc="left",
            fontsize=24,
            labelpad=10,
        )
        axes[0, last_col_idx].xaxis.set_label_position("top")

        # draw line in between queries and keys
        x0, _, width, _ = (
            axes[0, last_col_idx - 1]
            .get_tightbbox(fig.canvas.get_renderer())
            .transformed(fig.transFigure.inverted())
            .bounds
        )
        x1, _, width, _ = (
            axes[0, last_col_idx]
            .get_tightbbox(fig.canvas.get_renderer())
            .transformed(fig.transFigure.inverted())
            .bounds
        )
        line_x = (x0 + x1) / 2
        line = plt.Line2D((line_x, line_x), (0.1, 0.9), color="k", linewidth=1.5)
        fig.add_artist(line)

        last_col_idx += len(pred_dict["results"][-1]["predictions"]) + 1

        if independent:
            filename = f"retrieval-images-{name}-query-{query_feature_type}-key-{key_feature_type}.pdf"
            fig.tight_layout()
            fig.savefig(
                os.path.join(folder_path, filename),
                transparent=True,
                bbox_inches="tight",
            )
            print(f"Saved retrieved images to {os.path.join(folder_path, filename)}")

    # save final image if we put it all together in one
    if not independent:
        fig.tight_layout()
        fig.savefig(os.path.join(folder_path, "retrieved_images.pdf"), transparent=True, bbox_inches="tight")
        print(f"Saved retrieved images to {os.path.join(folder_path, 'retrieved_images.pdf')}")

    # save retrieval results JSON
    with open(retrieved_images_json_path, "w") as json_file:
        json.dump(retrieval_results, json_file, indent=4)
    print(f"Saved retrieval results to {os.path.join(folder_path, 'retrieved_images.json')}")

    return retrieval_results


def avg_list(l):
    return sum(l) * 1.0 / len(l)


def calculate_silhouette_score(args, image_features, labels):
    for level in ["order", "family", "genus", "species"]:
        gt_list = [labels[1][i][level] for i in range(len(labels[1]))]
        silhouette_score = silhouette_samples(image_features, gt_list)
        print(f"The silhouette score for {level} level is : {avg_list(silhouette_score)}")


def check_for_acc_about_correct_predict_seen_or_unseen(final_pred_list, species_list):
    for k in [1, 3, 5]:
        correct = 0
        total = 0
        for record in final_pred_list:
            top_k_species = record["species"]
            curr_top_k_pred = top_k_species[:k]
            for single_pred in curr_top_k_pred:
                if single_pred in species_list:
                    correct = correct + 1
                    break
            total = total + 1

        print(f"for k = {k}: {correct * 1.0 / total}")


@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join(
        args.project_root_path, "extracted_embedding", args.model_config.dataset, args.model_config.model_output_name
    )
    os.makedirs(folder_for_saving, exist_ok=True)
    labels_path = os.path.join(folder_for_saving, f"labels_{args.inference_and_eval_setting.eval_on}.json")
    processed_id_path = os.path.join(folder_for_saving, f"processed_id_{args.inference_and_eval_setting.eval_on}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted_features_path = os.path.join(
        folder_for_saving, f"extracted_feature_from_{args.inference_and_eval_setting.eval_on}_split.hdf5"
    )

    if os.path.exists(extracted_features_path) and os.path.exists(labels_path) and os.path.exists(processed_id_path) and args.load_inference:
        print("Loading embeddings from file...")

        with h5py.File(extracted_features_path, "r") as hdf5_file:
            seen_dict = {}
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["seen"].keys():
                    seen_dict[type_of_feature] = hdf5_file["seen"][type_of_feature][:]

            unseen_dict = {}
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["unseen"].keys():
                    unseen_dict[type_of_feature] = hdf5_file["unseen"][type_of_feature][:]
            keys_dict = {}
            for type_of_feature in All_TYPE_OF_FEATURES_OF_KEY:
                if type_of_feature in hdf5_file["key"].keys():
                    keys_dict[type_of_feature] = hdf5_file["key"][type_of_feature][:]

        with open(labels_path, "r") as json_file:
            total_dict = json.load(json_file)
        seen_dict["label_list"] = total_dict["seen_gt_dict"]
        unseen_dict["label_list"] = total_dict["unseen_gt_dict"]
        keys_dict["label_list"] = total_dict["key_gt_dict"]
        keys_dict["all_key_features_label"] = (
            total_dict["key_gt_dict"] + total_dict["key_gt_dict"] + total_dict["key_gt_dict"]
        )

        with open(processed_id_path, "r") as json_file:
            id_dict = json.load(json_file)
        seen_dict["processed_id_list"] = id_dict["seen_id_list"]
        unseen_dict["processed_id_list"] = id_dict["unseen_id_list"]
        keys_dict["processed_id_list"] = id_dict["key_id_list"]
        keys_dict["all_processed_id_list"] = id_dict["key_id_list"] + id_dict["key_id_list"] + id_dict["key_id_list"]
        print(f"Done loading embeddings from ‘{extracted_features_path}’")

    else:
        # initialize model
        print("Initialize model...")

        model = load_clip_model(args, device)

        if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
            pass
        else:
            checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
            print(f"Loading model from {args.model_config.ckpt_path}")
            print()

            model.load_state_dict(checkpoint)

        # Load data
        # args.model_config.batch_size = 24

        if args.inference_and_eval_setting.eval_on == "val":
            (
                _,
                seen_dataloader,
                unseen_dataloader,
                _,
                _,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        elif args.inference_and_eval_setting.eval_on == "test":
            (
                _,
                _,
                _,
                seen_dataloader,
                unseen_dataloader,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        else:
            raise ValueError(
                "Invalid value for eval_on, specify by 'python inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl_ver_0_1_2.yaml' inference_and_eval_setting.eval_on=test/val'"
            )
        for_open_clip = False

        if hasattr(args.model_config, "for_open_clip"):
            for_open_clip = args.model_config.for_open_clip

        keys_dict = get_features_and_label(
            all_keys_dataloader, model, device, for_key_set=True, for_open_clip=for_open_clip
        )

        seen_dict = get_features_and_label(seen_dataloader, model, device, for_open_clip=for_open_clip)

        unseen_dict = get_features_and_label(unseen_dataloader, model, device, for_open_clip=for_open_clip)

        if args.save_inference and not (os.path.exists(extracted_features_path) and os.path.exists(labels_path)):
            new_file = h5py.File(extracted_features_path, "w")
            name_of_splits = ["seen", "unseen", "key"]
            split_dicts = [seen_dict, unseen_dict, keys_dict]
            for split_name, split in zip(name_of_splits, split_dicts):
                group = new_file.create_group(split_name)
                for embedding_type in All_TYPE_OF_FEATURES_OF_KEY:
                    if embedding_type in split.keys():
                        try:
                            group.create_dataset(embedding_type, data=split[embedding_type])
                            print(f"Created dataset for {embedding_type}")
                        except:
                            print(f"Error in creating dataset for {embedding_type}")
                        # group.create_dataset(embedding_type, data=split[embedding_type])
            new_file.close()
            total_dict = {
                "seen_gt_dict": seen_dict["label_list"],
                "unseen_gt_dict": unseen_dict["label_list"],
                "key_gt_dict": keys_dict["label_list"],
            }
            with open(labels_path, "w") as json_file:
                json.dump(total_dict, json_file, indent=4)

            id_dict = {
                "seen_id_list": seen_dict["file_name_list"],
                "unseen_id_list": unseen_dict["file_name_list"],
                "key_id_list": keys_dict["file_name_list"],
            }
            with open(processed_id_path, "w") as json_file:
                json.dump(id_dict, json_file, indent=4)

    acc_dict, per_class_acc, pred_dict = inference_and_print_result(
        keys_dict,
        seen_dict,
        unseen_dict,
        args,
        small_species_list=None,
        k_list=args.inference_and_eval_setting.k_list,
    )

    per_claSS_acc_path = os.path.join(
        folder_for_saving, f"per_class_acc_{args.inference_and_eval_setting.eval_on}.json"
    )
    with open(per_claSS_acc_path, "w") as json_file:
        json.dump(per_class_acc, json_file, indent=4)
    print(f"Per class accuracy is saved in {per_claSS_acc_path}")

    acc_dict_path = os.path.join(folder_for_saving, f"acc_dict_{args.inference_and_eval_setting.eval_on}.json")
    with open(acc_dict_path, "w") as json_file:
        json.dump(acc_dict, json_file, indent=4)
    print(f"Accuracy is saved in {acc_dict_path}")

    pred_dict_path = os.path.join(folder_for_saving, f"pred_dict_{args.inference_and_eval_setting.eval_on}.json")
    with open(pred_dict_path, "w") as json_file:
        json.dump(pred_dict, json_file, indent=4)
    print(f"Prediction is saved in {pred_dict_path}")

    try:
        seen_keys_dataloader
        val_unseen_keys_dataloader
        test_unseen_keys_dataloader
    except:
        if args.inference_and_eval_setting.eval_on == "val":
            (
                _,
                seen_dataloader,
                unseen_dataloader,
                _,
                _,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        elif args.inference_and_eval_setting.eval_on == "test":
            (
                _,
                _,
                _,
                seen_dataloader,
                unseen_dataloader,
                seen_keys_dataloader,
                val_unseen_keys_dataloader,
                test_unseen_keys_dataloader,
                all_keys_dataloader,
            ) = load_bioscan_dataloader_all_small_splits(args)
        else:
            raise ValueError(
                "Invalid value for eval_on, specify by 'python inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl_ver_0_1_2.yaml' inference_and_eval_setting.eval_on=test/val'"
            )



    # seen_final_pred = pred_dict["encoded_image_feature"]["encoded_dna_feature"]["curr_seen_pred_list"]
    # unseen_final_pred = pred_dict["encoded_image_feature"]["encoded_dna_feature"]["curr_unseen_pred_list"]
    # all_unique_seen_species = get_all_unique_species_from_dataloader(seen_keys_dataloader)
    # all_unique_val_unseen_species = get_all_unique_species_from_dataloader(val_unseen_keys_dataloader)
    # all_unique_test_unseen_species = get_all_unique_species_from_dataloader(test_unseen_keys_dataloader)
    # print("For seen")
    # check_for_acc_about_correct_predict_seen_or_unseen(seen_final_pred, all_unique_seen_species)
    # print("For unseen")
    # check_for_acc_about_correct_predict_seen_or_unseen(
    #     unseen_final_pred, all_unique_val_unseen_species + all_unique_test_unseen_species
    # )

    if args.inference_and_eval_setting.plot_embeddings:
        generate_embedding_plot(
            args,
            seen_dict.get("encoded_image_feature"),
            seen_dict.get("encoded_dna_feature"),
            seen_dict.get("encoded_language_feature"),
            seen_dict["label_list"],
        )

    if args.inference_and_eval_setting.retrieve_images:
        if args.model_config.dataset == "bioscan_5m":
            if hasattr(args.model_config, "train_with_small_subset") and args.model_config.train_with_small_subset:
                image_data = h5py.File(args.bioscan_5m_data.path_to_smaller_hdf5_data, "r", libver="latest")
            else:
                image_data = h5py.File(args.bioscan_5m_data.path_to_hdf5_data, "r", libver="latest")
        else:
            image_data = h5py.File(args.bioscan_data.path_to_hdf5_data, "r")
        retrieve_images(
            args,
            f"{args.inference_and_eval_setting.eval_on}_seen",
            seen_dict,
            keys_dict,
            query_keys=[
                ("encoded_dna_feature", "encoded_dna_feature"),
                ("encoded_image_feature", "encoded_image_feature"),
                ("encoded_image_feature", "encoded_dna_feature"),
            ],
            query_data=image_data[f"{args.inference_and_eval_setting.eval_on}_seen"],
            key_data=image_data["all_keys"],
            is_bioscan5m=(args.model_config.dataset == "bioscan_5m"),
            **args.inference_and_eval_setting.retrieve_settings,
        )
        retrieve_images(
            args,
            f"{args.inference_and_eval_setting.eval_on}_unseen",
            unseen_dict,
            keys_dict,
            query_keys=[
                ("encoded_dna_feature", "encoded_dna_feature"),
                ("encoded_image_feature", "encoded_image_feature"),
                ("encoded_image_feature", "encoded_dna_feature"),
            ],
            query_data=image_data[f"{args.inference_and_eval_setting.eval_on}_unseen"],
            key_data=image_data["all_keys"],
            is_bioscan5m=(args.model_config.dataset == "bioscan_5m"),
            **args.inference_and_eval_setting.retrieve_settings,
        )


if __name__ == "__main__":
    main()
