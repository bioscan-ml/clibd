import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

yellow = "#FFBE7A"
red = "#FA7F6F"
blue = "#82B0D2"

x = np.arange(4)
labels = ['order', 'family', 'genus', 'species']

i_d_dna_to_dna_seen = [100, 99.1, 97.7, 94.4]
i_d_image_to_image_seen = [89.5, 89.1, 74.1, 59.2]
i_d_image_to_dna_seen = [99.7, 90.2, 73.4, 58.1]

i_d_dna_to_dna_unseen = [100.0, 97.6, 93.0, 86.9]
i_d_image_to_image_unseen = [97.6, 81.1, 59.7, 45.1]
i_d_image_to_dna_unseen = [71.8, 44.6, 18.7, 7.7]

baseline_dna_to_dna_seen = [78.8, 86.2, 82.1, 76.4]
baseline_image_to_image_seen = [54.9, 28.1, 14.2, 7.2]
baseline_image_to_dna_seen = [7.7, 0.5, 0.2, 0.1]

baseline_dna_to_dna_unseen = [91.8, 82.1, 69.4, 63.6]
baseline_image_to_image_unseen = [48.0, 21.7, 10.3, 5.0]
baseline_image_to_dna_unseen = [9.6, 0.8, 0, 0]

i_d_t_dna_to_dna_seen = [100, 100, 98.2, 95.6]
i_d_t_image_to_image_seen = [99.7, 90.9, 74.6, 59.3]
i_d_t_image_to_dna_seen = [99.4, 90.8, 70.6, 51.6]

i_d_t_dna_to_dna_unseen = [100, 98.3, 94.7, 90.4]
i_d_t_image_to_image_unseen = [94.4, 81.8, 60.4, 45.0]
i_d_t_image_to_dna_unseen = [88.5, 50.1, 20.8, 8.6]


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, baseline_image_to_image_seen, 'o-', color=red)
ax.plot(x, baseline_image_to_image_unseen, 'o--', color=red)
ax.plot(x, i_d_image_to_image_seen, 'o-', color=yellow)
ax.plot(x, i_d_image_to_image_unseen, 'o--', color=yellow)
ax.plot(x, i_d_t_image_to_image_seen, 'o-', color=blue)
ax.plot(x, i_d_t_image_to_image_unseen, 'o--', color=blue)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Taxonomic Rank', fontsize=16)
ax.set_ylabel('Macro-accuracy (%)', fontsize=16)
ax.set_title('Image to Image', fontsize=16)

for y in np.arange(0, 101, 5):
    if y % 10 == 0:
        ax.axhline(y=y, color='grey', linewidth=0.4, linestyle='-')
    else:
        ax.axhline(y=y, color='grey', linewidth=0.2, linestyle='-')

method_handles = [
    Line2D([0], [0], color=red, lw=2, label='No align'),
    Line2D([0], [0], color=yellow, lw=2, label='Image + DNA'),
    Line2D([0], [0], color=blue, lw=2, label='Image + DNA + Taxonomy')
]
style_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Seen'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Unseen')
]

legend1 = ax.legend(handles=method_handles, loc='lower left', bbox_to_anchor=(0, 0))
ax.add_artist(legend1)
legend2 = ax.legend(handles=style_handles, loc='lower left', bbox_to_anchor=(0.48, 0))
plt.tight_layout()
plt.show()