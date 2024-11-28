import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

color_group = {
    'I+D+T seen': '#5778a4',
    'I+D+T unseen': '#5778a4',
    'I+D+T hmean': '#5778a4',
    'I+T seen': '#e49444',
    'I+T unseen': '#e49444',
    'I+T hmean': '#e49444',
    'I+D seen': '#6a9f58',
    'I+D unseen': '#6a9f58',
    'I+D hmean': '#6a9f58',
    'SimCLR seen': '#b8b0ac',
    'SimCLR unseen': '#b8b0ac',
    'SimCLR hmean': '#b8b0ac',
    'BioCLIP seen': '#eedc82',
    'BioCLIP unseen': '#eedc82',
    'BioCLIP hmean': '#eedc82',
}

# marker_group = {
#     'I+D+T seen': 'o',
#     'I+D+T unseen': 'X',
#     'I+D+T hmean': 's',
#     'I+T seen': 'o',
#     'I+T unseen': 'X',
#     'I+T hmean': 's',
#     'I+D seen': 'o',
#     'I+D unseen': 'X',
#     'I+D hmean': 's',
#     'SimCLR seen': 'o',
#     'SimCLR unseen': 'X',
#     'SimCLR hmean': 's',
#     'BioCLIP seen': 'o',
#     'BioCLIP unseen': 'X',
#     'BioCLIP hmean': 's',
# }

# linestyle_group = {
#     'I+D+T seen': (),
#     'I+D+T unseen': (2,1),
#     'I+D+T hmean': (6,2),
#     'I+T seen': (),
#     'I+T unseen': (2,1),
#     'I+T hmean': (6,2),
#     'I+D seen': (),
#     'I+D unseen': (2,1),
#     'I+D hmean': (6,2),
#     'SimCLR seen': (),
#     'SimCLR unseen': (2,1),
#     'SimCLR hmean': (6,2),
#     'BioCLIP seen': (),
#     'BioCLIP unseen': (2,1),
#     'BioCLIP hmean': (6,2),
# }

marker_group = {
    'I+D+T seen': 'o',
    'I+D+T unseen': 'o',
    'I+D+T hmean': 'o',
    'I+T seen': 'o',
    'I+T unseen': 'o',
    'I+T hmean': 'o',
    'I+D seen': 'o',
    'I+D unseen': 'o',
    'I+D hmean': 'o',
    'SimCLR seen': 'o',
    'SimCLR unseen': 'o',
    'SimCLR hmean': 'o',
    'BioCLIP seen': 'o',
    'BioCLIP unseen': 'o',
    'BioCLIP hmean': 'o',
}

linestyle_group = {
    'I+D+T seen': (),
    'I+D+T unseen': (),
    'I+D+T hmean': (),
    'I+T seen': (),
    'I+T unseen': (),
    'I+T hmean': (),
    'I+D seen': (),
    'I+D unseen': (),
    'I+D hmean': (),
    'SimCLR seen': (),
    'SimCLR unseen': (),
    'SimCLR hmean': (),
    'BioCLIP seen': (),
    'BioCLIP unseen': (),
    'BioCLIP hmean': (),
}

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
                        harmonic_mean = 2 / (1 / curr_level_seen_acc + 1 / curr_level_unseen_acc)
                        acc_dict[query][key]['harmonic_mean'][acc_type][topk][level] = harmonic_mean
    return acc_dict

def get_acc_list_with_type_and_query_and_key(acc_dict, acc_type, seen_or_unseen, query, key):
    acc_list = []
    for model_name in acc_dict.keys():
        for split in [seen_or_unseen]:
            for level in ['order', 'family', 'genus', 'species']:
                try:
                    acc_list.append(acc_dict[model_name][query][key][split][acc_type]['1'][level])
                except:
                    print(f"Error in getting acc for {model_name} {query} {key} {split} {acc_type} {level}")
                    import pdb; pdb.set_trace()


    return acc_list

def generate_model_name_list(line_description, seen_or_unseen):
    model_name = []
    for model in line_description:
        for split in [seen_or_unseen]:
            model_name.append(model + " " + split)
    return model_name

def get_df(acc_dict, acc_type, seen_or_unseen, query, key, line_description):
    acc_list_for_micro = get_acc_list_with_type_and_query_and_key(acc_dict, acc_type, seen_or_unseen, query, key)
    model_name = generate_model_name_list(line_description, seen_or_unseen)
    data = {
        'Taxonomy Level': np.tile(['Order', 'Family', 'Genus', 'Species'], len(model_name)),
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

    color_labels[1], color_labels[2] = color_labels[2], color_labels[1]
    color_handles[1], color_handles[2] = color_handles[2], color_handles[1]

    color_legend = ax.legend(color_handles, color_labels, loc='lower left', bbox_to_anchor=(0, 0), fontsize=13)
    ax.add_artist(color_legend)


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

if __name__ == '__main__':
    path_to_I_D_T_acc_val_json = "/local-scratch2/projects/bioscan-clip/extracted_embedding/bioscan_1m/image_dna_text_4gpu/acc_dict_val.json"
    path_to_I_T_acc_val_json = "/local-scratch2/projects/bioscan-clip/extracted_embedding/bioscan_1m/image_text_4gpu/acc_dict_val.json"
    path_to_I_D_acc_val_json = "/local-scratch2/projects/bioscan-clip/extracted_embedding/bioscan_1m/image_dna_4gpu/acc_dict_val.json"
    path_to_simclr_acc_val_json = "/local-scratch2/projects/bioscan-clip/extracted_embedding/bioscan_1m/image_simclr_style_bioscan_1m/acc_dict_val.json"
    path_to_bioclip_acc_val_json = "/local-scratch2/projects/bioscan-clip/extracted_embedding/bioscan_1m/pre_trained_bioclip/acc_dict_val.json"
    list_of_acc_val_json = [path_to_I_D_T_acc_val_json, path_to_I_T_acc_val_json, path_to_I_D_acc_val_json, path_to_simclr_acc_val_json, path_to_bioclip_acc_val_json]

    # Check if all the json files are present
    query = "encoded_image_feature"
    key = "encoded_image_feature"
    acc_type = "macro_acc"
    top_k = "1"

    line_description = ["I+D+T", "I+T", "I+D", "SimCLR", "BioCLIP"]
    acc_dict = {}

    # load dict from json
    for i, path in enumerate(list_of_acc_val_json):
        with open(path, "r") as f:
            acc_dict[line_description[i]] = json.load(f)
            acc_dict[line_description[i]] = add_harmonic_mean_acc_to_dict(acc_dict[line_description[i]])

    # plot for image to image, macro.
    plot_acc(acc_dict, acc_type, 'seen', query, key, line_description)
    plot_acc(acc_dict, 'micro_acc', 'seen', query, key, line_description)
    plot_acc(acc_dict, acc_type, 'unseen', query, key, line_description)
    plot_acc(acc_dict, 'micro_acc', 'unseen', query, key, line_description)






