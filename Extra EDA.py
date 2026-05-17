"""
  Panel A: Three-class distribution, baseline vs enriched.
  Panel B: Composition of the recovered low-credibility class.

Output: eda_class_distribution.pdf (vector, for LaTeX) and .png (preview).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Numbers from the thesis. Adjust if your final counts differ.
# ---------------------------------------------------------------------------
CLASSES = ["High", "Medium", "Low"]
BASELINE_COUNTS  = [3232, 540,   7]
ENRICHED_COUNTS  = [3232, 540, 170]

# Low-credibility class composition (Panel B)
LOW_ORIGINAL   = 7     # already usable in the live-web MBFC dump
LOW_RECOVERED  = 163   # added via Wayback Machine enrichment

# ---------------------------------------------------------------------------
# Style: clean, academic, no chartjunk.
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]}
)

# ---------------------------------------------------------------------------
# Panel A: grouped bar chart, baseline vs enriched
# ---------------------------------------------------------------------------
x = np.arange(len(CLASSES))
width = 0.38
colour_baseline = "#9aa8b8"   # muted grey-blue
colour_enriched = "#2c5f8d"   # deeper academic blue

bars_b = ax_a.bar(x - width/2, BASELINE_COUNTS, width,
                  label="Baseline (n = 3,779)", color=colour_baseline,
                  edgecolor="black", linewidth=0.4)
bars_e = ax_a.bar(x + width/2, ENRICHED_COUNTS, width,
                  label="Enriched (n = 3,942)", color=colour_enriched,
                  edgecolor="black", linewidth=0.4)

# Annotate the low-credibility bars (the headline finding)
for bar, count in zip(bars_b, BASELINE_COUNTS):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
              f"{count:,}", ha="center", va="bottom", fontsize=9)
for bar, count in zip(bars_e, ENRICHED_COUNTS):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
              f"{count:,}", ha="center", va="bottom", fontsize=9)

ax_a.set_xticks(x)
ax_a.set_xticklabels(CLASSES)
ax_a.set_ylabel("Number of sources")
ax_a.set_title("(a) Class distribution before and after enrichment",
               loc="left", fontsize=11, pad=10)
ax_a.legend(frameon=False, loc="upper right")
ax_a.set_ylim(0, max(ENRICHED_COUNTS) * 1.12)

# ---------------------------------------------------------------------------
# Panel B: composition of the recovered low-credibility class
# ---------------------------------------------------------------------------
parts   = [LOW_ORIGINAL, LOW_RECOVERED]
labels  = [f"Originally usable\n(n = {LOW_ORIGINAL})",
           f"Recovered via\nWayback Machine\n(n = {LOW_RECOVERED})"]
colours = ["#c9ad7f", "#7a9c4e"]   # warm tan + muted olive — distinct from Panel A

ax_b.bar([0], [LOW_ORIGINAL],
         width=0.55, color=colours[0],
         edgecolor="black", linewidth=0.4, label=labels[0])
ax_b.bar([0], [LOW_RECOVERED], bottom=[LOW_ORIGINAL],
         width=0.55, color=colours[1],
         edgecolor="black", linewidth=0.4, label=labels[1])

# Inline annotation of the totals
ax_b.text(0, LOW_ORIGINAL / 2, f"{LOW_ORIGINAL}",
          ha="center", va="center", fontsize=9, color="black")
ax_b.text(0, LOW_ORIGINAL + LOW_RECOVERED / 2, f"{LOW_RECOVERED}",
          ha="center", va="center", fontsize=10, color="white", weight="bold")
ax_b.text(0, LOW_ORIGINAL + LOW_RECOVERED + 6,
          f"Total: {LOW_ORIGINAL + LOW_RECOVERED}",
          ha="center", va="bottom", fontsize=9, weight="bold")

ax_b.set_xticks([0])
ax_b.set_xticklabels(["Low-credibility class"])
ax_b.set_ylabel("Number of sources")
ax_b.set_ylim(0, (LOW_ORIGINAL + LOW_RECOVERED) * 1.25)
ax_b.set_title("(b) Recovery of the low-credibility class",
               loc="left", fontsize=11, pad=10)
ax_b.set_title("(b) Recovery of the low-credibility class",
               loc="left", fontsize=11, pad=10)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout()
plt.savefig("eda_class_distribution.pdf", bbox_inches="tight")
plt.savefig("eda_class_distribution.png", bbox_inches="tight", dpi=300)
print("Saved: eda_class_distribution.pdf and .png")
plt.show()