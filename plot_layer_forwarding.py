import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-colorblind")

# Safely extract prune and tp values
def safe_sort_key(prune_tp):
    match = re.match(r"prune(\d+)-tp(\d+)", prune_tp)
    if match:
        prune, tp = map(int, match.groups())
        return (tp, prune)
    return (float("inf"), float("inf"))

# Load and preprocess
data_updated = pd.read_csv("inference_log_stats.csv")
forwarding = data_updated[data_updated["type"] == "forwarding"].copy()
forwarding["prune_tp"] = forwarding["file"].str.extract(r'(prune\d+-tp\d+)')

# Extract TP level
forwarding["tp"] = forwarding["prune_tp"].str.extract(r'tp(\d+)').astype(int)

# Sorting
forwarding["sort_key"] = forwarding["prune_tp"].apply(safe_sort_key)
forwarding_sorted = forwarding.sort_values("sort_key")

# Assign colors by TP level
tp_color_map = {
    1: "#4C72B0",  # blue
    2: "#55A868",  # green
    4: "#FAA43A",  # orange
}
bar_colors = forwarding_sorted["tp"].map(tp_color_map)

# Plot
x = np.arange(len(forwarding_sorted))
bar_width = 0.6

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(x, forwarding_sorted["Mean (ms)"], width=bar_width, edgecolor='black', color=bar_colors, alpha=0.8, label="Mean")

# Percentile lines
markers = {"P70 (ms)": "v", "P80 (ms)": "s", "P90 (ms)": "D", "P99 (ms)": "o"}
for percentile, marker in markers.items():
    if percentile in forwarding_sorted.columns:
        ax.plot(x, forwarding_sorted[percentile], linestyle="--", linewidth=2, marker=marker, label=percentile)

# Legend for TP
from matplotlib.patches import Patch
legend_tp = [Patch(facecolor=color, edgecolor='black', label=f'TP={tp}') for tp, color in tp_color_map.items()]
ax.legend(handles=legend_tp + [plt.Line2D([0], [0], linestyle='--', marker=markers[p], label=p) for p in markers])

# Axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(forwarding_sorted["prune_tp"], rotation=45)
ax.set_title("Forwarding Latency by TP Level")
ax.set_xlabel("Pruning and TP setting")
ax.set_ylabel("Latency (ms)")
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("forwarding_latency_colored_by_tp.pdf")
plt.show()
