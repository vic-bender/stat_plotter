"""
Generates summary statistics, histogram, and boxplot for the given dataset.
Run with:  python3 stat_plotter.py (make sure dependencies are installed)
Outputs:   output_plots.png  (saved in the same directory)

Created by Victor A. Bender
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Data ─────────────────────────────────────────────────────────────────────
data = np.array([  # sample data, can be replaced with any 1D array of numeric values
    12, 2, 9, 2, 5, 5, 4, 7, 3,
     4, 6, 6, 4, 4, 3, 4, 7, 6,
     2, 3, 5, 6, 5, 8, 2, 8, 5,
     5, 3, 5, 2, 7, 5, 6, 1, 4
])

# ── Summary Statistics ────────────────────────────────────────────────────────
n       = len(data)
mean    = np.mean(data)
std     = np.std(data, ddof=1)
var     = np.var(data, ddof=1)
minimum = np.min(data)
q1      = np.percentile(data, 25)
median  = np.median(data)
q3      = np.percentile(data, 75)
maximum = np.max(data)
iqr     = q3 - q1

print("=" * 50)
print("  Summary Statistics")
print("=" * 50)
print(f"  n                : {n}")
print(f"  Mean (x̄)        : {mean:.4f}")
print(f"  Std Dev (s)      : {std:.4f}")
print(f"  Variance (s²)    : {var:.4f}")
print()
print("  Five-Number Summary:")
print(f"    Min            : {minimum}")
print(f"    Q1             : {q1}")
print(f"    Median (Q2)    : {median}")
print(f"    Q3             : {q3}")
print(f"    Max            : {maximum}")
print(f"    IQR (Q3-Q1)    : {iqr}")
print()
print(f"  Poisson λ̂ estimate (= x̄) : {mean:.4f}")
print("=" * 50)

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.suptitle("Output Plot Based on Array Data", fontsize=13, fontweight='bold')

# Three columns: histogram | boxplot | stats table
# Give the stats column a fixed narrow width via width_ratios
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4, width_ratios=[2.5, 2.5, 1])

# --- Histogram ---------------------------------------------------------------
ax1 = fig.add_subplot(gs[0])
bins = np.arange(minimum - 0.5, maximum + 1.5, 1)
ax1.hist(data, bins=bins, color='steelblue', edgecolor='white', linewidth=0.8)

x_vals = np.arange(minimum, maximum + 1)
poisson_probs = stats.poisson.pmf(x_vals, mu=mean)
ax1.plot(x_vals, poisson_probs * n, 'ro-', markersize=5, linewidth=1.5,
         label=f'Poisson(λ={mean:.2f}) × n')
ax1.axvline(mean, color='darkred', linestyle='--', linewidth=1.2,
            label=f'mean = {mean:.2f}')

ax1.set_xlabel("Value", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.set_title("Histogram", fontsize=12)
ax1.set_xticks(x_vals)
ax1.yaxis.get_major_locator().set_params(integer=True)
ax1.legend(fontsize=9)

# --- Boxplot -----------------------------------------------------------------
ax2 = fig.add_subplot(gs[1])
ax2.boxplot(data, vert=False, patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.6),
            medianprops=dict(color='darkred', linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', color='tomato', markersize=7, alpha=0.8))

ax2.set_xlabel("Value", fontsize=11)
ax2.set_title("Boxplot", fontsize=12)
ax2.set_yticks([])

# --- Stats Table (proper subplot, no clipping) -------------------------------
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')  # invisible axes — used purely as a text canvas

stats_lines = [
    ("n",        f"{n}"),
    ("x̄",        f"{mean:.4f}"),
    ("s",        f"{std:.4f}"),
    ("s²",       f"{var:.4f}"),
    ("Min",      f"{minimum}"),
    ("Q₁",       f"{q1}"),
    ("Median",   f"{median}"),
    ("Q₃",       f"{q3}"),
    ("Max",      f"{maximum}"),
    ("IQR",      f"{iqr}"),
    ("λ̂",        f"{mean:.4f}"),
]

# Draw a light box background
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.add_patch(plt.Rectangle((0, 0), 1, 1,
              facecolor='lightyellow', edgecolor='gray',
              linewidth=1, transform=ax3.transAxes, clip_on=False))

ax3.set_title("Summary Stats", fontsize=11, pad=8)

row_height = 1 / (len(stats_lines) + 1)
for i, (label, value) in enumerate(stats_lines):
    y = 1 - (i + 1) * row_height
    ax3.text(0.05, y, label, transform=ax3.transAxes,
             fontsize=10, verticalalignment='center', fontweight='bold')
    ax3.text(0.95, y, value, transform=ax3.transAxes,
             fontsize=10, verticalalignment='center', ha='right')
    # light divider line
    if i < len(stats_lines) - 1:
        ax3.axhline(y - row_height / 2, color='lightgray', linewidth=0.5)

plt.savefig("output_plots.png", dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to: output_plots.png")
plt.show()