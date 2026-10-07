"""Draw figures/ from results/*.json (no rerun).

    python -m benchmarks.figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {"L1": ("#a8a6a1", "L1 (basis pursuit)"),
         "ISD": ("#eb6834", "iterative support detection"),
         "Pandora": ("#2a78d6", "Pandora")}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10.5, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "legend.frameon": False, "legend.labelcolor": INK2,
    "figure.dpi": 140,
})


def phase():
    rows = json.loads((RES / "phase.json").read_text())
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(13, 3.7), sharey=True)
    for ax, n in zip(axes, ns):
        sel = sorted([r for r in rows if r["n"] == n], key=lambda r: r["k"])
        ratio = [r["k"] / n for r in sel]
        for m, (color, label) in STYLE.items():
            y = [100 * r[m] / r["trials"] for r in sel]
            ax.plot(ratio, y, "-o", color=color, lw=2.5 if m == "Pandora" else 2, ms=5,
                    label=label)
        ax.set_title(f"{n} measurements, 200 unknowns", loc="left")
        ax.set_xlabel("nonzeros / measurements")
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("exact recoveries (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Phase diagram: Pandora recovers as much or more at every size", x=0.01,
                 ha="left", color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "phase.png")
    plt.close(fig)


def ensembles():
    rows = json.loads((RES / "ensembles.json").read_text())
    kinds = list(dict.fromkeys(r["matrix"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, k in zip(axes, (11, 14)):
        sel = [next(r for r in rows if r["matrix"] == kd and r["k"] == k) for kd in kinds]
        x = np.arange(len(kinds))
        w = 0.27
        for i, (m, (color, label)) in enumerate(STYLE.items()):
            vals = [100 * r[m] / r["trials"] for r in sel]
            bars = ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x, kinds)
        ax.set_title(f"{k} nonzeros, 40 measurements, 200 unknowns", loc="left")
    axes[0].set_ylabel("exact recoveries (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Four kinds of measurement matrix", x=0.01, ha="left", color=INK,
                 weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "ensembles.png")
    plt.close(fig)


def measurements():
    rows = json.loads((RES / "measurements.json").read_text())
    fig, axes = plt.subplots(1, len(rows), figsize=(13, 3.8), sharey=True)
    for ax, r in zip(axes, rows):
        for m, (color, label) in STYLE.items():
            ax.plot(r["m"], [100 * v for v in r["curves"][m]], "-o", color=color,
                    lw=2.5 if m == "Pandora" else 2, ms=4, label=label)
            if r["m50"][m]:
                ax.axvline(r["m50"][m], color=color, lw=1, ls=":")
        ax.axhline(50, color=GRID, lw=1)
        ax.set_title(f"{r['k']} nonzeros: m50 is {r['m50']['L1']:.1f} for L1, "
                     f"{r['m50']['Pandora']:.1f} for Pandora", loc="left", fontsize=9.5)
        ax.set_xlabel("measurements")
        ax.set_xlim(min(r["m"]) - 1, min(max(r["m"]), 6 * r["k"]) + 1)
    axes[0].set_ylabel("exact recoveries (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Pandora needs fewer measurements for the same recovery", x=0.01, ha="left",
                 color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "measurements.png")
    plt.close(fig)


def baselines():
    rows = json.loads((RES / "validate_baselines.json").read_text())
    order = ["L1", "ISD", "OMP (given k)", "CoSaMP (given k)", "Subspace Pursuit (given k)",
             "IHT (given k)", "Pandora"]
    colors = [INK2, "#eb6834", "#e4e3df", "#d0cec9", "#bcbab5", "#a8a6a1", "#2a78d6"]
    sel = [r for r in rows if (r["n"], r["k"]) in ((40, 10), (40, 12), (40, 14), (100, 30))]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(sel))
    w = 0.12
    for i, (m, c) in enumerate(zip(order, colors)):
        vals = [r[m] for r in sel]
        lo = [r[m] - r["ci95"][m][0] for r in sel]
        hi = [r["ci95"][m][1] - r[m] for r in sel]
        ax.bar(x + (i - 3) * w, vals, width=w - 0.015, color=c, label=m.replace(" (given k)", " (told k)"),
               yerr=[lo, hi], error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x, [f"{r['k']} nonzeros\n{r['n']} x {r['d']}" for r in sel])
    ax.set_ylabel("exact recoveries of 100 (95% CI)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_title("Against seven methods; greedy ones are told the true sparsity", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "baselines.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    phase()
    ensembles()
    measurements()
    baselines()
