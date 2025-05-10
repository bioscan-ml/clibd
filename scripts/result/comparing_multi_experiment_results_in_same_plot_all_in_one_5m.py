import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages

from bioscanclip.util.util import get_acc_list_with_type_and_query_and_key

color_group = {
    'Image + DNA + Taxonomy seen': '#5778a4',
    'Image + DNA + Taxonomy unseen': '#5778a4',
    'Image + DNA + Taxonomy hmean': '#5778a4',
    'Image + Taxonomy seen': '#e49444',
    'Image + Taxonomy unseen': '#e49444',
    'Image + Taxonomy hmean': '#e49444',
    'Image + DNA seen': '#6a9f58',
    'Image + DNA unseen': '#6a9f58',
    'Image + DNA hmean': '#6a9f58',
    'no align baseline seen': '#b8b0ac',
    'no align baseline unseen': '#b8b0ac',
    'no align baseline hmean': '#b8b0ac',
    'BioCLIP seen': '#eedc82',
    'BioCLIP unseen': '#eedc82',
    'BioCLIP hmean': '#eedc82',
}

MODEL2COLORSTR = {
    "none": "black",
    "random_resnet50": "dimgrey",
    "random_vitb16": "dimgrey",
    "resnet50": "tab:red",
    "mocov3_resnet50": "tab:green",
    "dino_resnet50": "tab:purple",
    "vicreg_resnet50": "tab:orange",
    "clip_RN50": "tab:olive",
    "vitb16": "tab:red",
    "mocov3_vit_base": "tab:green",
    "dino_vitb16": "tab:purple",
    "timm_vit_base_patch16_224.mae": "tab:blue",
    "mae_pretrain_vit_base_global": "tab:brown",
    "clip_vitb16": "tab:olive",
    "mae_finetuned_vit_base_global": "tab:brown",
    "barcodebert": "tab:brown",
    "dnabert-2": "tab:orange",
    "dnabert-s": "tab:red",
    "hyenadna": "tab:cyan",
    "NucleotideTransformer": "tab:green",
}

COLOR_GROUP = {
    'Image + DNA + Taxonomy seen': 'tab:blue',
    'Image + DNA + Taxonomy unseen': 'tab:blue',
    'Image + DNA + Taxonomy hmean': 'tab:blue',
    'Image + Taxonomy seen': 'tab:orange',
    'Image + Taxonomy unseen': 'tab:orange',
    'Image + Taxonomy hmean': 'tab:orange',
    'Image + DNA seen': 'tab:green',
    'Image + DNA unseen': 'tab:green',
    'Image + DNA hmean': 'tab:green',
    'no align baseline seen': '#b8b0ac',
    'no align baseline unseen': '#b8b0ac',
    'no align baseline hmean': '#b8b0ac',
}
color_group = {k: matplotlib.colors.to_rgb(v) for k, v in COLOR_GROUP.items()}
# marker_group = {
#     'Image + DNA + Taxonomy seen': 'o',
#     'Image + DNA + Taxonomy unseen': 'X',
#     'Image + DNA + Taxonomy hmean': 's',
#     'Image + Taxonomy seen': 'o',
#     'Image + Taxonomy unseen': 'X',
#     'Image + Taxonomy hmean': 's',
#     'Image + DNA seen': 'o',
#     'Image + DNA unseen': 'X',
#     'Image + DNA hmean': 's',
#     'no align baseline seen': 'o',
#     'no align baseline unseen': 'X',
#     'no align baseline hmean': 's',
#     'BioCLIP seen': 'o',
#     'BioCLIP unseen': 'X',
#     'BioCLIP hmean': 's',
# }

# linestyle_group = {
#     'Image + DNA + Taxonomy seen': (),
#     'Image + DNA + Taxonomy unseen': (2,1),
#     'Image + DNA + Taxonomy hmean': (6,2),
#     'Image + Taxonomy seen': (),
#     'Image + Taxonomy unseen': (2,1),
#     'Image + Taxonomy hmean': (6,2),
#     'Image + DNA seen': (),
#     'Image + DNA unseen': (2,1),
#     'Image + DNA hmean': (6,2),
#     'no align baseline seen': (),
#     'no align baseline unseen': (2,1),
#     'no align baseline hmean': (6,2),
#     'BioCLIP seen': (),
#     'BioCLIP unseen': (2,1),
#     'BioCLIP hmean': (6,2),
# }

marker_group = {
    'Image + DNA + Taxonomy seen': 'o',
    'Image + DNA + Taxonomy unseen': 'o',
    'Image + DNA + Taxonomy hmean': 'o',
    'Image + Taxonomy seen': 'o',
    'Image + Taxonomy unseen': 'o',
    'Image + Taxonomy hmean': 'o',
    'Image + DNA seen': 'o',
    'Image + DNA unseen': 'o',
    'Image + DNA hmean': 'o',
    'no align baseline seen': 'o',
    'no align baseline unseen': 'o',
    'no align baseline hmean': 'o',
    'BioCLIP seen': 'o',
    'BioCLIP unseen': 'o',
    'BioCLIP hmean': 'o',
}

linestyle_group = {
    'Image + DNA + Taxonomy seen': (),
    'Image + DNA + Taxonomy unseen': (),
    'Image + DNA + Taxonomy hmean': (),
    'Image + Taxonomy seen': (),
    'Image + Taxonomy unseen': (),
    'Image + Taxonomy hmean': (),
    'Image + DNA seen': (),
    'Image + DNA unseen': (),
    'Image + DNA hmean': (),
    'no align baseline seen': (),
    'no align baseline unseen': (),
    'no align baseline hmean': (),
    'BioCLIP seen': (),
    'BioCLIP unseen': (),
    'BioCLIP hmean': (),
}

# query_key_combination_we_want

def add_harmonic_mean_acc_to_dict(acc_dict):
    for query in acc_dict.keys():
        for key in acc_dict[query].keys():
            acc_dict[query][key]['harmonic_mean'] = {}
            for acc_type in ['micro_acc', 'macro_acc']:
                for topk in ['1']:
                    if "seen" not in acc_dict[query][key] or "unseen" not in acc_dict[query][key]:
                        continue

                    seen_acc_for_four_level = acc_dict[query][key]['seen'][acc_type][topk]
                    unseen_acc_for_four_level = acc_dict[query][key]['unseen'][acc_type][topk]

                    acc_dict[query][key]['harmonic_mean'][acc_type] = {}
                    acc_dict[query][key]['harmonic_mean'][acc_type][topk] = {}
                    for level in ['order', 'family', 'genus', 'species']:
                        curr_level_seen_acc = seen_acc_for_four_level[level]
                        curr_level_unseen_acc = unseen_acc_for_four_level[level]
                        if curr_level_seen_acc == 0 or  curr_level_unseen_acc == 0:
                            harmonic_mean = 0
                        else:
                            harmonic_mean = 2 / (1 / curr_level_seen_acc + 1 / curr_level_unseen_acc)
                        acc_dict[query][key]['harmonic_mean'][acc_type][topk][level] = harmonic_mean
    return acc_dict


def generate_model_name_list(line_description, seen_or_unseen):
    model_name = []
    for model in line_description:
        for split in [seen_or_unseen]:
            model_name.append(model + " " + split)
    return model_name

def get_df(acc_dict, acc_type, seen_or_unseen, query, key, line_description):
    acc_list_for_micro = get_acc_list_with_type_and_query_and_key(acc_dict, acc_type, seen_or_unseen, query, key)
    model_name = generate_model_name_list(line_description, seen_or_unseen)
    print(np.tile(['order', 'family', 'genus', 'species'], len(model_name)).shape)
    print(len(acc_list_for_micro))
    print(np.repeat(model_name, 4).shape)
    data = {
        'Taxonomy Level': np.tile(['order', 'family', 'genus', 'species'], len(model_name)),
        'Accuracy': acc_list_for_micro,
        'Model': np.repeat(model_name, 4)
    }

    return pd.DataFrame(data)

def plot_acc(acc_dict, acc_type, seen_or_unseen, query, key, line_description):
    df = get_df(acc_dict, acc_type, seen_or_unseen, query, key, line_description)

    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=macro_df, x='Taxonomy Level', y='Accuracy', palette=color_group, dashes=linestyle_group, markers=marker_group, hue='Model', style='Model')
    sns.lineplot(data=df, x='Taxonomy Level', y='Accuracy', palette=color_group, dashes=False,
                 markers=False, hue='Model', style='Model')
    handles, labels = ax.get_legend_handles_labels()
    color_labels = line_description
    # color_handles = [handles[i] for i in range(0, len(handles), 3)]
    color_handles = handles

    # color_labels[1], color_labels[2] = color_labels[2], color_labels[1]
    # color_handles[1], color_handles[2] = color_handles[2], color_handles[1]

    color_legend = ax.legend(color_handles, color_labels, loc='lower left', bbox_to_anchor=(0, 0), fontsize=13)
    ax.add_artist(color_legend)
    ax.clear()

    sns.lineplot(data=df, x='Taxonomy Level', y='Accuracy', palette=color_group, dashes=linestyle_group,
                 markers=marker_group, hue='Model', style='Model', legend=False)

    if acc_type == "macro_acc":
        plt.ylabel("Macro Accuracy", fontsize=13)
    else:
        plt.ylabel("Micro Accuracy", fontsize=13)
    if seen_or_unseen == "seen":
        plt.title(f"Seen {acc_type} accuracy", fontsize=13)
    else:
        plt.title(f"Unseen {acc_type} accuracy", fontsize=13)
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.tick_params(axis='both', which='major', labelsize=13)

    ax.add_artist(color_legend)
    # ax.add_artist(marker_legend)
    plt.tight_layout()
    # Save the plot to pdf
    file_name = f"acc_plot_{acc_type}_{query}_{key}_{seen_or_unseen}.pdf"
    folder = "/local-scratch2/projects/bioscan-clip/plot"
    os.makedirs(folder, exist_ok=True)
    # plt.show()

    plt.savefig(os.path.join(folder, file_name))

def plot_all_in_one_v2(acc_dict, acc_types, seen_or_unseen_list, query, key, line_description, output_pdf_path):
    fig, axes = plt.subplots(1, 3, figsize=(7, 15))
    pdf_pages = PdfPages(output_pdf_path)  # 创建 PDF 文件

    plot_order = [(0, 0), (0, 1), (0, 2)]

    alignments = ['Image + Taxonomy', 'Image + DNA', 'Image + DNA + Taxonomy']
    for align_idx, alignment in enumerate(alignments):

        ax = axes[plot_order[align_idx][1]]

        for i, (acc_type, seen_or_unseen) in enumerate(zip(acc_types, seen_or_unseen_list)):
            df = get_df(acc_dict, acc_type, seen_or_unseen, query, key, line_description)

            print(f"alignment: {alignment}")
            print(acc_type, seen_or_unseen)
            print(f"df: {df}")


def plot_all_in_one(acc_dict, acc_types, seen_or_unseen_list, query, key, line_description, output_pdf_path):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pdf_pages = PdfPages(output_pdf_path)

    plot_order = [(0, 0), (0, 1)]

    for i, (acc_type, seen_or_unseen) in enumerate(zip(acc_types, seen_or_unseen_list)):
        ax = axes[plot_order[i][1]]
        df = get_df(acc_dict, acc_type, seen_or_unseen, query, key, line_description)

        if i==0:
            sns.lineplot(data=df, x='Taxonomy Level', y='Accuracy', palette=color_group, dashes=False,
                         markers=False, hue='Model', ax=ax, style='Model')
            handles, labels = ax.get_legend_handles_labels()
            print(f"labels: {labels}")
            print(f"handles: {handles}")
            color_labels = line_description
            color_handles = handles
            # color_labels[1], color_labels[2] = color_labels[2], color_labels[1]
            # color_handles[1], color_handles[2] = color_handles[2], color_handles[1]
            color_legend = ax.legend(color_handles, color_labels, loc='lower left', bbox_to_anchor=(0, 0), fontsize=13)

            ax.clear()

        sns.lineplot(data=df, x='Taxonomy Level', y='Accuracy', palette=color_group, dashes=linestyle_group,
                     markers=marker_group, hue='Model', style='Model', ax=ax, legend=False)
        if i==0:

            ax.add_artist(color_legend)

        ax.tick_params(axis='x', labelrotation=45)


        if acc_type == "macro_acc":
            ax.set_ylabel("Macro Accuracy (%)", fontsize=13)
        else:
            ax.set_ylabel("Micro Accuracy (%)", fontsize=13)

        if seen_or_unseen == "seen":
            ax.set_title(f"Seen Species", fontsize=13)
        else:
            ax.set_title(f"Unseen Species", fontsize=13)

        ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.8)
        ax.grid(True, axis="y", which="minor", linestyle="-", alpha=0.2)  # adj
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=13)




    plt.tight_layout()
    pdf_pages.savefig(fig)
    print(f"Save to {output_pdf_path}")
    pdf_pages.close()

if __name__ == '__main__':
    json_root = '/localhome/zmgong/second_ssd/projects/bioscan-clip/extracted_embedding/bioscan_5m/acc_json'
    no_align_base_line_val_json = f"{json_root}/no_alignment_baseline/acc_dict_test.json"
    path_to_I_D_T_acc_val_json = f"{json_root}/image_dna_text_4gpu/acc_dict_test.json"
    path_to_I_T_acc_val_json = f"{json_root}/image_text_4gpu/acc_dict_test.json"
    path_to_I_D_acc_val_json = f"{json_root}/image_dna_4gpu/acc_dict_test.json"
    list_of_acc_val_json = [path_to_I_D_T_acc_val_json, path_to_I_T_acc_val_json, path_to_I_D_acc_val_json, no_align_base_line_val_json]

    query = "encoded_image_feature"
    key = "encoded_image_feature"
    line_description = [ "Image + DNA + Taxonomy", "Image + Taxonomy", "Image + DNA","no align baseline",]
    acc_dict = {}

    for i, path in enumerate(list_of_acc_val_json):
        with open(path, "r") as f:
            acc_dict[line_description[i]] = json.load(f)
            acc_dict[line_description[i]] = add_harmonic_mean_acc_to_dict(acc_dict[line_description[i]])


    # acc_types = ["macro_acc", "macro_acc", "micro_acc", "micro_acc"]
    # seen_or_unseen_list = ["seen", "unseen", "seen", "unseen"]
    acc_types = ["macro_acc", "macro_acc"]
    seen_or_unseen_list = ["seen", "unseen"]







    output_pdf_path = "plot/all_plots.pdf"
    # save acc_dict_for_scott to json
    # with open("plot/acc_dict_for_scott.json", "w") as f:
    #     json.dump(acc_dict_for_scott, f)

    plot_all_in_one(acc_dict, acc_types, seen_or_unseen_list, query, key, line_description, output_pdf_path)
    # plot_all_in_one_v2(acc_dict, acc_types, seen_or_unseen_list, query, key, line_description, output_pdf_path)