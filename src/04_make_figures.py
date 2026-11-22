"""Step 4 - Figures of the paper (PDF for submission + PNG for preview).

At the end, the four figures of the paper produced by the code (figures 1, 3, 4 and 5)
are opened on screen. Use  --no-show  to only write the files (e.g. on a server).

Input : data/processed/features.csv, results/raw/*.csv, results/tables/*.csv
Output: results/figures/*.pdf and *.png (fig1, fig3, fig4, fig5 = paper figures; supp_* = supplementary)
Run   : python3 src/04_make_figures.py
"""
import os
import sys

import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wearlib as W  # noqa: E402

RAW = os.path.join(W.ROOT, "results", "raw")
TAB = os.path.join(W.ROOT, "results", "tables")
FIG = os.path.join(W.ROOT, "results", "figures")
VB_LIM = 0.5

# Categorical colours in fixed order (colour-blind validated); every method also has
# its own line style / marker so the figures survive greyscale printing.
MAIN = ["EMP", "EMP3", "RF", "MLP", "GRU", "PINN-soft", "HWR", "HWR3", "HWR-sel"]
# Paper names: HWR = prior selected in each fold (code name HWR-sel); HWR-P and HWR-T are the
# fixed-prior variants (code names HWR and HWR3). Variants of a model keep its colour and are
# told apart by hatching (bars), line style (curves) and their own label.
COL = {"EMP": "#2a78d6", "EMP3": "#2a78d6", "RF": "#eb6834", "MLP": "#1baf7a", "GRU": "#eda100",
       "PINN-soft": "#e87ba4", "HWR": "#008300", "HWR3": "#008300", "HWR-sel": "#008300"}
HATCH = {"EMP3": "////", "HWR": "////", "HWR3": "...."}
LS = {"EMP": (0, (4, 2)), "EMP3": (0, (1, 1.5)), "RF": ":", "MLP": (0, (1, 1)), "GRU": "-.",
      "PINN-soft": (0, (5, 1, 1, 1)), "HWR": (0, (6, 1.5)), "HWR3": (0, (2, 1)), "HWR-sel": "-"}
LABEL = {"EMP": "Power law", "EMP3": "Three-stage law", "RF": "Random forest", "MLP": "MLP",
         "GRU": "GRU", "PINN-soft": "Soft-constraint PINN", "HWR": "HWR-P (power-law prior)",
         "HWR3": "HWR-T (three-stage prior)", "HWR-sel": "HWR (prior selected)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PROT = {"LOCO": "Leave-one-case-out", "LOCD": "Leave-one-condition-out"}
W1, W2 = 3.35, 6.7       # single / double column width in inches (8.5 cm / 17 cm)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "sans-serif",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.2, "savefig.dpi": 600,
    "pdf.fonttype": 42, "figure.constrained_layout.use": True})


# Output names: figures of the paper keep their paper number, the others are supplementary.
PAPER_NAMES = {"Fig1_dataset": "fig1_dataset",
               "Fig5_wear_trajectories_LOCO": "fig3_trajectories",
               "Fig6_monotonicity_LOCO": "fig4_monotonicity",
               "Fig7_rate_multiplier_LOCO": "fig5_multiplier"}


def save(fig, name):
    name = PAPER_NAMES.get(name, "supp_" + name.split("_", 1)[1])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def fig_dataset(cases):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1), sharey=True)
    for ax, mat in zip(axes, (1, 2)):
        conds = sorted({c["cond"] for c in cases.values() if c["material"] == mat})
        seen = set()
        for c in cases.values():
            if c["material"] != mat:
                continue
            k = conds.index(c["cond"])
            first = c["cond"] not in seen
            seen.add(c["cond"])
            lab = ~np.isnan(c["vb"])
            ax.plot(c["t"][lab], c["vb"][lab], color=COND_COL[k], lw=1.0,
                    marker="o" if first else "s", ms=3.2,
                    mfc=COND_COL[k] if first else "white", mew=0.8,
                    label=f"$a_p$ = {c['cond'][1]} mm, $f$ = {c['cond'][2]} mm/rev" if first else None)
            ax.plot(c["t"][~lab], np.interp(c["t"][~lab], np.r_[0, c["t"][lab]], np.r_[0, c["vb"][lab]]),
                    ls="none", marker="x", ms=3.5, mew=0.8, color=INK, zorder=4)
            ax.annotate(str(c["case"]), (c["t"][lab][-1], c["vb"][lab][-1]), xytext=(2, 1),
                        textcoords="offset points", fontsize=6, color=MUTED)
        ax.axhline(VB_LIM, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(f"{W.MATERIAL_NAME[mat].capitalize()} "
                     f"({sum(c['material'] == mat for c in cases.values())} inserts)", loc="left")
        ax.set_xlabel("Cutting time (min)")
    axes[0].set_ylabel("Flank wear VB (mm)")
    h, l = axes[0].get_legend_handles_labels()
    order = np.argsort(l)
    h, l = [h[i] for i in order], [l[i] for i in order]
    h.append(plt.Line2D([], [], ls="none", marker="s", ms=3.2, mfc="white", mec=MUTED, mew=0.8))
    l.append("replicate insert")
    h.append(plt.Line2D([], [], ls="none", marker="x", ms=3.5, mew=0.8, color=INK))
    l.append("pass without VB measurement")
    fig.legend(h, l, loc="outside lower center", ncol=3, handlelength=1.6)
    save(fig, "Fig1_dataset")


def fig_overall(ov):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.3), sharey=True)
    for ax, prot in zip(axes, PROT):
        g = ov[ov.protocol == prot].set_index("method").loc[MAIN[::-1]]
        y = np.arange(len(g))
        bars = ax.barh(y, g.RMSE_seed_mean, xerr=g.RMSE_seed_std, height=0.62,
                       color=[COL[m] for m in g.index],
                       error_kw=dict(lw=0.7, ecolor=INK, capsize=1.5))
        for bar, m in zip(bars, g.index):
            bar.set(hatch=HATCH.get(m), edgecolor="white", linewidth=0)
        ax.plot(g.RMSE_ens, y, ls="none", marker="D", ms=3.5, mfc="white", mec=INK, mew=0.8,
                label="5-seed ensemble")
        for yi, (v, sd, e) in enumerate(zip(g.RMSE_seed_mean, g.RMSE_seed_std, g.RMSE_ens)):
            ax.text(max(v + sd, e) + 0.008, yi, f"{v:.3f}", va="center", fontsize=7)
        ax.set_yticks(y, [LABEL[m] for m in g.index])
        ax.set_xlim(0, ov.RMSE_seed_mean.max() * 1.22)
        ax.set_xlabel("RMSE (mm), mean ± s.d. over 5 seeds")
        ax.set_title(PROT[prot], loc="left")
        ax.grid(axis="y", visible=False)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center")
    save(fig, "Fig3_overall_rmse")


def fig_scatter(pred, cases, prot="LOCO"):
    g = pred[(pred.protocol == prot) & (pred.seed == "ens")].dropna(subset=["VB"])
    fig, axes = plt.subplots(3, 3, figsize=(W2, 6.6), sharex=True, sharey=True)
    lim = (0, 1.7)
    for ax, m in zip(axes.ravel(), MAIN):
        s = g[g.method == m]
        steel = s.case.map(lambda c: cases[c]["material"] == 2).to_numpy()
        ax.plot(lim, lim, color=MUTED, lw=0.6)
        ax.plot(s.VB[~steel], s.pred[~steel], ls="none", marker="o", ms=3, mfc=COL[m],
                mec="white", mew=0.3, label="cast iron")
        ax.plot(s.VB[steel], s.pred[steel], ls="none", marker="s", ms=3, mfc="white",
                mec=COL[m], mew=0.8, label="steel")
        r = W.metrics(s.VB.to_numpy(), s.pred.to_numpy())
        ax.set_title(LABEL[m], loc="left")
        ax.text(0.97, 0.04, f"RMSE {r['RMSE']:.3f}\n$R^2$ {r['R2']:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5)
        ax.set(xlim=lim, ylim=lim, aspect="equal")
    h = [plt.Line2D([], [], ls="none", marker="o", ms=3.5, mfc=MUTED, mec="white", mew=0.3),
         plt.Line2D([], [], ls="none", marker="s", ms=3.5, mfc="white", mec=MUTED, mew=0.8)]
    fig.legend(h, ["cast iron", "steel"], loc="outside lower center", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Measured VB (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted VB (mm)")
    save(fig, f"Fig4_predicted_vs_measured_{prot}")


def fig_trajectories(iv, cases, prot="LOCO"):
    g = iv[iv.protocol == prot]
    ids = sorted(g.case.unique())
    fig, axes = plt.subplots(3, 5, figsize=(W2, 4.9), sharex=False)
    for ax, c in zip(axes.ravel(), ids):
        for m in ("EMP", "PINN-soft", "HWR-sel"):
            s = g[(g.case == c) & (g.method == m)].sort_values("t")
            if m == "HWR-sel":
                ax.fill_between(s.t, s.lo, s.hi, color=COL[m], alpha=0.18, lw=0,
                                label="HWR 90 % interval")
            ax.plot(s.t, s.pred, color=COL[m], ls=LS[m], label=LABEL[m],
                    lw=1.4 if m == "HWR-sel" else 1.0)
        lab = s.dropna(subset=["VB"])
        ax.plot(lab.t, lab.VB, ls="none", marker="o", ms=2.6, mfc=INK, mec="white", mew=0.3,
                label="Measured", zorder=5)
        ax.axhline(VB_LIM, color=MUTED, lw=0.5, ls=(0, (2, 2)))
        k = cases[c]
        ax.set_title(f"{c} · {W.MATERIAL_NAME[k['material']]} · {k['cond'][1]}/{k['cond'][2]}",
                     loc="left", fontsize=6)
        ax.set_ylim(0, max(0.9, 1.1 * max(lab.VB.max(), s.hi.max())))
        ax.set_xlim(0, None)
    for ax in axes[-1]:
        ax.set_xlabel("Cutting time (min)", fontsize=7)
    fig.supylabel("Flank wear VB (mm)", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5)
    save(fig, f"Fig5_wear_trajectories_{prot}")


def fig_monotonicity(pred, ov, prot="LOCO"):
    ens = pred[(pred.protocol == prot) & (pred.seed == "ens")]
    # case with the largest total decrease of the soft-constraint PINN
    drop = {c: -np.minimum(np.diff(s.sort_values("t").pred), 0).sum()
            for c, s in ens[ens.method == "PINN-soft"].groupby("case")}
    c = max(drop, key=drop.get)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    for m in ("MLP", "GRU", "PINN-soft", "HWR-sel"):
        s = ens[(ens.case == c) & (ens.method == m)].sort_values("t")
        ax.plot(s.t, s.pred, color=COL[m], ls=LS[m], label=LABEL[m],
                lw=1.5 if m == "HWR-sel" else 1.0)
    lab = s.dropna(subset=["VB"])
    ax.plot(lab.t, lab.VB, ls="none", marker="o", ms=3, mfc=INK, mec="white", mew=0.3,
            label="Measured", zorder=5)
    ax.set(xlabel="Cutting time (min)", ylabel="VB (mm)")
    ax.set_title(f"(a) Held-out case {c}", loc="left")
    ax.legend(loc="upper left")
    ax = axes[1]
    g = ov[ov.protocol == prot].set_index("method").loc[MAIN[::-1]]
    y = np.arange(len(g))
    bars = ax.barh(y, g.decreasing_pairs_pct, height=0.62, color=[COL[m] for m in g.index])
    for bar, m in zip(bars, g.index):
        bar.set(hatch=HATCH.get(m), edgecolor="white", linewidth=0)
    for yi, v in enumerate(g.decreasing_pairs_pct):
        ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(y, [LABEL[m] for m in g.index])
    ax.set_xlim(0, max(5, g.decreasing_pairs_pct.max() * 1.25))
    ax.set_xlabel("Consecutive passes with decreasing\npredicted wear (%)")
    ax.set_title("(b) All held-out cases", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, f"Fig6_monotonicity_{prot}")


def fig_multiplier(pred, cases, prot="LOCO"):
    """Rate multiplier learned by the network: HWR increment / increment of its prior law."""
    ens = pred[(pred.protocol == prot) & (pred.seed == "ens")]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.4), sharey="row")
    for row, (model, law) in zip(axes, (("HWR", "EMP"), ("HWR3", "EMP3"))):
        for ax, mat in zip(row, (1, 2)):
            for c in sorted(ens.case.unique()):
                if cases[c]["material"] != mat:
                    continue
                h = ens[(ens.case == c) & (ens.method == model)].sort_values("t")
                e = ens[(ens.case == c) & (ens.method == law)].sort_values("t")
                dh, de = np.diff(np.r_[0, h.pred]), np.diff(np.r_[0, e.pred])
                ok = de > 1e-9
                ax.plot(h.t.to_numpy()[ok], dh[ok] / de[ok], color=COL[model], lw=0.8, alpha=0.75,
                        marker="o", ms=2)
                ax.annotate(str(c), (h.t.to_numpy()[ok][-1], (dh[ok] / de[ok])[-1]),
                            xytext=(2, 0), textcoords="offset points", fontsize=6, color=MUTED)
            ax.axhline(1, color=INK, lw=0.6)
            name = LABEL[model].split(" (")[0]
            ax.set_title(f"{name}: {LABEL[law].lower()} prior, {W.MATERIAL_NAME[mat]}", loc="left")
        row[0].set_ylabel("Rate multiplier $e^{h_k}$")
    for ax in axes[-1]:
        ax.set_xlabel("Cutting time (min)")
    save(fig, f"Fig7_rate_multiplier_{prot}")


def fig_toolchange(detail, prot="LOCO"):
    d = detail[(detail.protocol == prot) & (detail.method == "HWR-sel")]
    p = d.pivot(index="case", columns="policy", values="VB_true_at_decision")
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    x = np.arange(len(p))
    ax.vlines(x, p["upper"], p["point"], color=GRID, lw=1.5, zorder=1)
    ax.plot(x, p["point"], ls="none", marker="o", ms=4.5, mfc="white", mec=COL["HWR-sel"], mew=1.0,
            label="point estimate ≥ limit")
    ax.plot(x, p["upper"], ls="none", marker="o", ms=4.5, mfc=COL["HWR-sel"], mec="white", mew=0.4,
            label="upper 90 % bound ≥ limit")
    ax.axhline(VB_LIM, color=INK, lw=0.7, ls=(0, (2, 2)))
    ax.text(len(p) - 0.5, VB_LIM, " limit", va="bottom", ha="right", fontsize=7)
    ax.set_xticks(x, p.index)
    ax.set(xlabel="Held-out case", ylabel="Measured VB at tool change (mm)")
    ax.grid(axis="x", visible=False)
    fig.legend(loc="outside lower center", ncol=1)
    save(fig, f"Fig8_tool_change_{prot}")


PAPER_FIGURES = [
    ("fig1_dataset", "Figure 1 - Measured flank wear of the 15 inserts"),
    ("fig3_trajectories", "Figure 3 - Estimated wear of the held-out inserts with 90 % intervals"),
    ("fig4_monotonicity", "Figure 4 - Decreasing estimates: HWR against the reference models"),
    ("fig5_multiplier", "Figure 5 - Wear-rate multiplier learned by the network"),
]


def show_paper_figures():
    """Open the paper figures on screen, one window each."""
    import matplotlib.image as mpimg
    for name, title in PAPER_FIGURES:
        img = mpimg.imread(os.path.join(FIG, name + ".png"))
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(12, 12 * h / w), num=title)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
    plt.show()


def main():
    os.makedirs(FIG, exist_ok=True)
    cases = W.load_cases()
    pred = pd.read_csv(os.path.join(RAW, "predictions.csv"), dtype={"seed": str})
    iv = pd.read_csv(os.path.join(RAW, "intervals.csv"), dtype={"seed": str})
    ov = pd.read_csv(os.path.join(TAB, "T2_overall_metrics.csv"))
    detail = pd.read_csv(os.path.join(TAB, "T8_tool_change_per_case.csv"))
    fig_dataset(cases)
    fig_overall(ov)
    for prot in PROT:
        fig_scatter(iv, cases, prot)
        fig_trajectories(iv, cases, prot)
    fig_monotonicity(iv, ov)
    fig_multiplier(iv, cases)
    fig_toolchange(detail)
    print("figures written to", FIG)
    if "--no-show" not in sys.argv:
        show_paper_figures()


if __name__ == "__main__":
    main()
