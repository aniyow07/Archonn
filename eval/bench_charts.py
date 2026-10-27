"""Draw the benchmark charts in results/run2/bench from the scores in the README.

    python eval/bench_charts.py results/run2/bench
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]
BASE, FT = "#8a8f98", "#d9773d"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def grouped(fname, title, ylabel, cats, base, ft, fmt, ymax=None):
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    x = range(len(cats)); w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], base, w, label="Base Qwen3-8B", color=BASE)
    b2 = ax.bar([i + w / 2 for i in x], ft, w, label="Archon (run2)", color=FT)
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[fmt(v) for v in bars.datavalues], padding=2, fontsize=9)
    ax.set_xticks(list(x), cats); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold", pad=22)
    if ymax: ax.set_ylim(0, ymax)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=9); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{out}/{fname}", facecolor="white"); plt.close(fig)

grouped("accuracy.png", "Accuracy on full test sets", "Accuracy (%)",
        ["MMLU\n(14,042)", "ARC-Challenge\n(1,172)", "GSM8K\n(1,319)", "IFEval\n(541)"],
        [73.0, 56.7, 88.7, 85.2], [73.0, 56.9, 81.7, 76.9], lambda v: f"{v:.1f}", 105)
grouped("tokens.png", "Average tokens per answer, thinking included", "Tokens",
        ["GSM8K", "IFEval"], [896, 1204], [281, 481], lambda v: f"{v:,.0f}")
grouped("time.png", "Time per benchmark on one NVIDIA L4", "Minutes",
        ["MMLU + ARC", "GSM8K", "IFEval", "Total"], [42, 103, 31, 176], [38, 36, 18, 92], lambda v: f"{v:.0f}")
print("ok")
