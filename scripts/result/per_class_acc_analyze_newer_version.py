import h5py
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import math
import pandas as pd
from brokenaxes import brokenaxes
import json
import copy
from scipy.interpolate import make_interp_spline


def load_hdf5_data(data_hdf5_path):
    h5file = h5py.File(data_hdf5_path, "r")
    return h5file


def avg_list(l):

    return sum(l) * 1.0 / len(l)


def get_avg_acc_by_key_count(count_list, acc_list, species_names):
    record_num_to_acc = {}
    record_num_to_species = {}
    for record_number_of_species_in_key_set, acc, species_name in zip(count_list, acc_list, species_names):
        if record_number_of_species_in_key_set not in record_num_to_acc.keys():
            record_num_to_acc[record_number_of_species_in_key_set] = []
            record_num_to_species[record_number_of_species_in_key_set] = []
        record_num_to_acc[record_number_of_species_in_key_set].append(acc)
        record_num_to_species[record_number_of_species_in_key_set].append(species_name)

    number_of_record_list = []
    averaged_acc = []
    num_species = []
    species = []

    for record_number_of_species_in_key_set in record_num_to_acc.keys():
        number_of_record_list.append(record_number_of_species_in_key_set)
        averaged_acc.append(avg_list(record_num_to_acc[record_number_of_species_in_key_set]))
        num_species.append(len(record_num_to_acc[record_number_of_species_in_key_set]))
        species.append(record_num_to_species[record_number_of_species_in_key_set])
    return number_of_record_list, averaged_acc, num_species, species


def format_title(query_type, key_type):
    # Mapping the feature types to readable names
    feature_name_map = {"encoded_image_feature": "Image", "encoded_dna_feature": "DNA"}
    return f"{feature_name_map[query_type]} to {feature_name_map[key_type]}"


def rand_jitter(arr, factor=0.01):
    """From https://stackoverflow.com/questions/8671808/avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot"""
    stdev = factor * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def get_modified_cmap(name, cmap, start=0.0, end=1.0, num_colors=256):
    colors = cmap(np.linspace(start, end, num_colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, colors)
    return cmap


def get_scaled_size(values: np.ndarray, min_size, max_size):
    """Scale the values to be between min_size and max_size"""
    min_val = np.min(values)
    max_val = np.max(values)
    return min_size + (max_size - min_size) * ((values - min_val) / (max_val - min_val))


def plot_scatterplot(
    species_2_query_count_and_acc, title, ax, show_y_label=True, min_point_size=100, max_point_size=400
):
    seen_species_count_list = []
    seen_species_acc_list = []
    unseen_species_count_list = []
    unseen_species_acc_list = []
    seen_species_names = []
    unseen_species_names = []

    for species in species_2_query_count_and_acc.keys():
        if "seen" in species_2_query_count_and_acc[species].keys():
            seen_species_names.append(species)
            seen_species_count_list.append(species_2_query_count_and_acc[species]["key_count"])
            seen_species_acc_list.append(species_2_query_count_and_acc[species]["seen"])
        if "unseen" in species_2_query_count_and_acc[species].keys():
            unseen_species_names.append(species)
            unseen_species_count_list.append(species_2_query_count_and_acc[species]["key_count"])
            unseen_species_acc_list.append(species_2_query_count_and_acc[species]["unseen"])

    seen_species_count_list, seen_species_acc_list, seen_num_species_list, seen_species = get_avg_acc_by_key_count(
        seen_species_count_list, seen_species_acc_list, seen_species_names
    )
    unseen_species_count_list, unseen_species_acc_list, unseen_num_species_list, unseen_species = (
        get_avg_acc_by_key_count(unseen_species_count_list, unseen_species_acc_list, unseen_species_names)
    )

    # print("SEEN:")
    # seen_high_performing = np.where(np.array(seen_species_acc_list) > 0.5)[0]
    # for idx in seen_high_performing:
    #     print(" ", seen_species[idx], seen_species_acc_list[idx], seen_species_count_list[idx])

    # print("UNSEEN:")
    # unseen_high_performing = np.where(np.array(unseen_species_acc_list) > 0.4)[0]
    # for idx in unseen_high_performing:
    #     print(" ", unseen_species[idx], unseen_species_acc_list[idx], unseen_species_count_list[idx])

    # Plotting both seen and unseen species data
    dot_size = 25
    fonr_size = 18
    # colors = sns.color_palette("pastel", n_colors=2)
    colors = [
        get_modified_cmap(
            "modBlues",
            plt.cm.Blues,
            start=0.3,
            end=0.9,
            num_colors=max(seen_num_species_list) - min(seen_num_species_list) + 1,
        ),
        get_modified_cmap(
            "modOranges",
            plt.cm.Oranges,
            start=0.3,
            end=0.8,
            num_colors=max(unseen_num_species_list) - min(unseen_num_species_list) + 1,
        ),
    ]
    scatter_seen = ax.scatter(
        seen_species_count_list + np.random.rand(len(seen_species_count_list)) * 0.1,
        seen_species_acc_list + np.random.rand(len(seen_species_acc_list)) * 0.02,
        c=seen_num_species_list,
        cmap=colors[0],
        label="Seen Species",
        s=get_scaled_size(seen_num_species_list, min_point_size, max_point_size),
        alpha=1.0,
    )
    scatter_unseen = ax.scatter(
        unseen_species_count_list + np.random.rand(len(unseen_species_count_list)) * 0.1,
        unseen_species_acc_list + np.random.rand(len(unseen_species_acc_list)) * 0.02,
        c=unseen_num_species_list,
        cmap=colors[1],
        label="Unseen Species",
        s=get_scaled_size(unseen_num_species_list, min_point_size, max_point_size),
        alpha=1.0,
    )

    print("Seen species count:", min(seen_num_species_list), max(seen_num_species_list))
    print("Unseen species count:", min(unseen_num_species_list), max(unseen_num_species_list))

    # ax.set_title(title, fontsize=fonr_size, pad=30)

    if show_y_label:
        ax.tick_params(axis="y", labelsize=fonr_size - 2)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        legend = ax.legend(
            handles=[
                mlines.Line2D([], [], color=colors[0](0.7), marker="o", markersize=10, ls="", label="Seen Species"),
                mlines.Line2D([], [], color=colors[1](0.7), marker="o", markersize=10, ls="", label="Unseen Species"),
            ],
            fontsize=fonr_size,
            framealpha=1.0,
            loc="lower right",
        )
        # legend.legend_handles[0].set_facecolor(colors[0](1.0))
        # legend.legend_handles[0].set_edgecolor(colors[0](1.0))
        # legend.legend_handles[1].set_facecolor(colors[1](1.0))
        # legend.legend_handles[1].set_edgecolor(colors[1](1.0))

    else:
        ax.set_ylabel("")
        ax.set_yticks([])

        # ax.set_xlabel('Number of records of the species in the key set', fontsize=fonr_size)

    ax.tick_params(axis="x", labelsize=fonr_size - 2)

    ax.set_ylim(-0.02, 1.1)
    ax.set_xscale("log")
    return scatter_seen, scatter_unseen


def plot_multiple_scatterplot(
    per_class_acc_dict, all_keys_species, query_key_combinations, seen_and_unseen, k_list, levels
):
    # get dict for species to count
    species_2_key_record_count = {}
    for species in all_keys_species:
        if species not in species_2_key_record_count.keys():
            species_2_key_record_count[species] = {}
            species_2_key_record_count[species]["key_count"] = 0
        species_2_key_record_count[species]["key_count"] = species_2_key_record_count[species]["key_count"] + 1

    # For the combination of query and key
    fig, axs = plt.subplots(1, len(query_key_combinations), figsize=(26, 4))  # Adjust the size as needed
    fig.subplots_adjust(top=0.919, bottom=0.137, left=0.053, right=0.972)
    axs = axs.flatten()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03)
    ax_index = 0
    for query_type, key_type in query_key_combinations:
        species_2_query_count_and_acc = copy.deepcopy(species_2_key_record_count)
        for seen_or_unseen in seen_and_unseen:
            for k in k_list:
                for level in levels:
                    curr_acc_dict = per_class_acc_dict[query_type][key_type][seen_or_unseen][k][level]
                    for species in curr_acc_dict.keys():
                        species_2_query_count_and_acc[species][seen_or_unseen] = curr_acc_dict[species]
        if key_type == "encoded_image_feature":
            image_or_dna_as_key = "Image"
        else:
            image_or_dna_as_key = "DNA"
        show_y_label = False
        if ax_index == 0:
            show_y_label = True
        print(f"Plotting {query_type} to {key_type}")
        scatter_seen, scatter_unseen = plot_scatterplot(
            species_2_query_count_and_acc, None, axs[ax_index], show_y_label
        )
        ax_index += 1

    plt.colorbar(scatter_unseen, location="right", orientation="vertical", pad=-0.03)
    plt.colorbar(scatter_seen, location="right", orientation="vertical", pad=0.03)

    # set x-axis label for the whole figure
    # plt.xlabel('Number of records of the species in the key set', fontsize=18)
    plt.savefig(
        "per_class_acc_orig.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()


def load_per_class_acc(per_class_acc_path):
    with open(per_class_acc_path) as json_file:
        per_class_acc = json.load(json_file)
    return per_class_acc


if __name__ == "__main__":
    per_class_acc_path = "extracted_embedding/bioscan_1m/image_dna_text_4gpu/per_class_acc_test.json"
    per_class_acc_dict = load_per_class_acc(per_class_acc_path)

    query_feature_list = ["encoded_image_feature", "encoded_text_feature"]
    key_feature_list = ["encoded_image_feature", "encoded_dna_feature"]

    query_key_combinations = [
        ("encoded_image_feature", "encoded_image_feature"),
        ("encoded_image_feature", "encoded_dna_feature"),
        ("encoded_image_feature", "encoded_language_feature"),
        ("encoded_dna_feature", "encoded_dna_feature"),
    ]

    seen_and_unseen = ["seen", "unseen"]
    k_list = ["1"]
    levels = ["species"]

    data_hdf5_path = "data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5"
    # Get all species list in a dict
    data_h5file = load_hdf5_data(data_hdf5_path)

    all_keys_species = [item.decode("utf-8") for item in data_h5file["all_keys"]["species"]]

    plot_multiple_scatterplot(
        per_class_acc_dict, all_keys_species, query_key_combinations, seen_and_unseen, k_list, levels
    )