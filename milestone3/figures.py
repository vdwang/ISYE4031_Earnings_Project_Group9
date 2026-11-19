"""Report figures regenerated from the audit tables."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from milestone3.evaluation import EXPANDING, FAST, NEURAL, TUNED

INK = "#162843"
TEAL = "#087D78"
RUST = "#BB5336"
SLATE = "#717D8C"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)


def render(out, half_life):
    trace = pd.read_csv(out / "coefficient_trace.csv", parse_dates=["reaction_day"])
    fig, axes = plt.subplots(2, 1, figsize=(8.7, 3.7), sharex=True, layout="constrained")
    for ax, feature, title in zip(
        axes, ("reaction_day_return", "completed_event_return_lag1"),
        ("Initial reaction return", "Most recent completed event return"),
    ):
        for model, label, color in [(FAST, "30-day memory", RUST),
                                    (TUNED, f"{half_life:g}-day memory", TEAL),
                                    (EXPANDING, "All history", INK)]:
            series = trace.loc[trace.model.eq(model)]
            ax.step(series.reaction_day, series[feature], where="post", color=color, lw=1,
                    label=label)
        ax.axhline(0, color=SLATE, lw=.8, alpha=.7)
        ax.axvline(pd.Timestamp("2025-01-01"), color=SLATE, lw=.8, ls=":")
        ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=10)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="outside upper center", fontsize=8)
    fig.supylabel("Return pp per training SD", color=INK, fontsize=10)
    fig.savefig(out / "online_coefficients.png", dpi=180)
    plt.close(fig)

    contrasts = pd.read_csv(out / "issuer_offset_ablation.csv")
    paired = contrasts.loc[contrasts.reference.eq(NEURAL)].set_index("per_issuer_cap")
    paired = paired.reindex(["5", "10", "20", "full"])
    difference = paired.delta_rmse_pp
    interval = [difference - paired.lower_95_pp, paired.upper_95_pp - difference]
    fig, ax = plt.subplots(figsize=(6.8, 2.3), layout="constrained")
    ax.axhline(0, color=INK, lw=.8)
    ax.errorbar(range(len(paired)), difference, yerr=interval, fmt="o", capsize=4,
                markersize=5, color=TEAL, lw=1.3)
    ax.set(xticks=range(len(paired)), xticklabels=["5", "10", "20", "Full"],
           xlim=(-.45, 3.45), xlabel="Available training events per selected issuer",
           ylabel="Mixed minus pooled\nRMSE (pp)")
    style(ax)
    fig.savefig(out / "history_caps.png", dpi=180)
    plt.close(fig)
