import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

MODALITY2COLORSTR = {

    'DNA to DNA': "tab:blue",
    'Image to Image': "tab:red",
    'Image to DNA': "tab:purple",
}
MODALITY2COLORSTRRGB = {k: matplotlib.colors.to_rgb(v) for k, v in MODALITY2COLORSTR.items()}

color_map = MODALITY2COLORSTRRGB
marker_map = {
    'seen': 'o',
    'unseen': 'X',
    'harmonic mean': 's'
}

query_key_combinations = [['encoded_dna_feature', 'encoded_dna_feature'],
                          ['encoded_image_feature', 'encoded_image_feature'],
                          ['encoded_image_feature', 'encoded_dna_feature']]


def add_harmonic_mean_acc_to_dict(acc_dict):
    for query in acc_dict.keys():
        for key in acc_dict[query].keys():

            if "seen" not in acc_dict[query][key] or "unseen" not in acc_dict[query][key]:
                continue
            acc_dict[query][key]['harmonic mean'] = {}
            for acc_type in ['micro_acc', 'macro_acc']:
                acc_dict[query][key]['harmonic mean'][acc_type] = {}
                for top_k in ['1', '3', '5']:
                    acc_dict[query][key]['harmonic mean'][acc_type][top_k] = {}
                    seen_acc = acc_dict[query][key]['seen'][acc_type][top_k]
                    unseen_acc = acc_dict[query][key]['unseen'][acc_type][top_k]

                    for level in ['order', 'family', 'genus', 'species']:
                        if seen_acc[level] == 0 or unseen_acc[level] == 0:
                            harmonic_mean = 0
                        else:
                            harmonic_mean = 2 / (1 / seen_acc[level] + 1 / unseen_acc[level])
                        acc_dict[query][key]['harmonic mean'][acc_type][top_k][level] = harmonic_mean
    return acc_dict


def plot_accuracy(acc_dict, output_folder, experiment_name, acc_type='macro_acc', ):
    plt.figure(figsize=(5, 3.5))
    taxonomy_levels = ['order', 'family', 'genus', 'species']
    color_lines = []

    for color_key, color_value in color_map.items():
        color_line, = plt.plot([], [], color=color_value, label=color_key)
        color_lines.append(color_line)

    marker_lines = []

    for (query, key), color_map_key in zip(query_key_combinations, color_map.keys()):
        for split in ['seen', 'unseen', 'harmonic mean']:
            try:
                return acc_dict[query][key][split][acc_type]['1']
            except (KeyError, TypeError) as e:
                continue

            curr_acc_list = [curr_acc_dict_with_four_level[level] for level in taxonomy_levels]
            # time 100 to get percentage
            curr_acc_list = [acc * 100 for acc in curr_acc_list]
            plt.plot(taxonomy_levels, curr_acc_list, color=color_map[color_map_key], marker=marker_map[split])

            marker_line, = plt.plot([], [], color='gray', marker=marker_map[split], label=split)
            marker_lines.append(marker_line)

    if experiment_name == "Image + DNA + Taxonomy":

        color_legend = plt.legend(handles=color_lines, loc='lower left', fontsize=12)
        plt.gca().add_artist(color_legend)

        marker_legend = plt.legend(handles=marker_lines[:3], loc='lower left', bbox_to_anchor=(0.285, 0), fontsize=12)

    plt.title(f"{experiment_name}", fontsize=14)
    plt.ylabel("Macro-accuracy (%)", fontsize=14)
    plt.grid(True, axis="y", which="major", linestyle="-", alpha=0.8)
    plt.grid(True, axis="y", which="minor", linestyle="-", alpha=0.2)
    plt.xticks(ticks=np.arange(len(taxonomy_levels)), labels=taxonomy_levels, rotation=45, fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    # save to output folder, in pdf
    # plt.savefig(f"{output_folder}/{experiment_name}.png")
    plt.savefig(f"{output_folder}/{experiment_name}.pdf")
    plt.close()

if __name__ == '__main__':
    json_root = 'extracted_embedding/bioscan_5m/acc_json'
    list_of_acc_val_json = [
        f"{json_root}/image_dna_text_4gpu/acc_dict_test.json",
        f"{json_root}/image_text_4gpu/acc_dict_test.json",
        f"{json_root}/image_dna_4gpu/acc_dict_test.json",
        f"{json_root}/no_alignment_baseline/acc_dict_test.json"
    ]

    experiment_names = ["Image + DNA + Taxonomy",
                        "Image + DNA",
                        "Image + Taxonomy",
                        "no align baseline",]

    line_descriptions = ['DNA to DNA',
                        'Image to Image',
                        'Image to DNA']
    output_folder = 'plots'

    acc_type = 'macro_acc'

    os.makedirs(output_folder, exist_ok=True)

    for path, experiment_name in zip(list_of_acc_val_json, experiment_names):
        with open(path, "r") as f:
            acc_dict = json.load(f)
            acc_dict = add_harmonic_mean_acc_to_dict(acc_dict)
            plot_accuracy(acc_dict, output_folder, experiment_name, acc_type)
