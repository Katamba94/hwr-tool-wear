"""Step 3 - Turn raw predictions into the tables of the paper.

Input : results/raw/{predictions,inner_oof,selected_cfg}.csv
Output: results/tables/*.csv  and  results/raw/intervals.csv
Run   : python3 src/03_analyse.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wearlib as W  # noqa: E402

RAW = os.path.join(W.ROOT, "results", "raw")
TAB = os.path.join(W.ROOT, "results", "tables")
MAIN = ["EMP", "EMP3", "RF", "MLP", "GRU", "PINN-soft", "HWR", "HWR3", "HWR-sel"]
ABL = ["HWR", "HWR-noprior", "HWR-nofeedback", "HWR-labelled-only"]
ALPHA = 0.10          # 90 % intervals
A_SCALE = 0.10        # mm, offset of the normalised conformal score
VB_LIM = 0.50         # mm, tool-change criterion on flank wear


def add_selected(pred, oof, sel, a="HWR", b="HWR3", name="HWR-sel"):
    """HWR with the prior law treated as one more hyper-parameter: in each outer fold keep
    HWR (power law) or HWR3 (three-stage law), whichever has the lower inner-validation
    RMSE on the training cases. No test information is used."""
    w = sel[sel.method.isin([a, b])].pivot_table(index=["protocol", "fold"], columns="method",
                                                 values="inner_rmse")
    pick = (w[b] < w[a]).map({True: b, False: a}).rename("pick").reset_index()
    out = []
    for d in (pred, oof):
        e = d[d.method.isin([a, b])].merge(pick, on=["protocol", "fold"])
        e = e[e.method == e.pick].drop(columns="pick").assign(method=name)
        out.append(pd.concat([d, e], ignore_index=True))
    pick.to_csv(os.path.join(TAB, "T9b_prior_selected_per_fold.csv"), index=False)
    return out[0], out[1]


def m3(y, p):
    r = W.metrics(np.asarray(y), np.asarray(p))
    return r["RMSE"], r["MAE"], r["R2"]


def overall(pred):
    out = []
    lab = pred.dropna(subset=["VB"])
    for (prot, meth), g in lab.groupby(["protocol", "method"]):
        seeds = [m3(s.VB, s.pred) for k, s in g[g.seed != "ens"].groupby("seed")]
        ens = g[g.seed == "ens"]
        e = m3(ens.VB, ens.pred)
        # monotonicity on ALL passes (labelled or not) of the ensemble prediction
        allp = pred[(pred.protocol == prot) & (pred.method == meth) & (pred.seed == "ens")]
        d = np.concatenate([np.diff(c.sort_values("t").pred) for _, c in allp.groupby("case")])
        sd = np.std(seeds, 0) if len(seeds) > 1 else [0, 0, 0]
        out.append(dict(protocol=prot, method=meth,
                        RMSE_seed_mean=np.mean(seeds, 0)[0], RMSE_seed_std=sd[0],
                        MAE_seed_mean=np.mean(seeds, 0)[1], MAE_seed_std=sd[1],
                        RMSE_ens=e[0], MAE_ens=e[1], R2_ens=e[2],
                        decreasing_pairs_pct=100 * np.mean(d < -1e-6),
                        largest_drop_mm=float(max(0, -d.min()))))
    return pd.DataFrame(out)


def per_case(pred, prot):
    g = pred[(pred.protocol == prot) & (pred.seed == "ens")].dropna(subset=["VB"])
    t = g.groupby(["case", "method"]).apply(lambda s: m3(s.VB, s.pred)[0]).unstack("method")
    return t[[m for m in MAIN + ABL[1:] if m in t.columns]]


def per_material(pred, cases):
    g = pred[pred.seed == "ens"].dropna(subset=["VB"]).copy()
    g["material"] = g.case.map(lambda c: W.MATERIAL_NAME[cases[c]["material"]])
    out = []
    for (prot, mat, meth), s in g.groupby(["protocol", "material", "method"]):
        r = m3(s.VB, s.pred)
        out.append(dict(protocol=prot, material=mat, method=meth, RMSE=r[0], MAE=r[1], R2=r[2]))
    return pd.DataFrame(out)


def paired_tests(pred, refs=("HWR", "HWR3", "HWR-sel", "EMP3"), n_boot=5000, seed=0):
    """Each reference model against the others: Wilcoxon on per-case RMSE and a case
    bootstrap of the pooled-RMSE difference (cases, not passes, are the resampling unit).
    EMP3 is only compared with EMP (three-stage law against power law)."""
    rng = np.random.default_rng(seed)
    out = []
    for prot in pred.protocol.unique():
        g = pred[(pred.protocol == prot) & (pred.seed == "ens")].dropna(subset=["VB"])
        pc = per_case(pred, prot)
        ids = pc.index.to_numpy()
        se = {m: {c: ((s.pred - s.VB) ** 2).to_numpy() for c, s in g[g.method == m].groupby("case")}
              for m in pc.columns}
        pool = lambda m, cs: np.sqrt(np.concatenate([se[m][c] for c in cs]).mean())
        done = set()
        for ref in refs:
            if ref not in pc.columns:
                continue
            for m in (["EMP"] if ref == "EMP3" else pc.columns):
                if m == ref or (m, ref) in done:
                    continue
                done.add((ref, m))
                diffs = []
                for _ in range(n_boot):
                    bs = rng.choice(ids, len(ids))
                    diffs.append(pool(ref, bs) - pool(m, bs))
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                out.append(dict(protocol=prot, comparison=f"{ref} - {m}",
                                cases_ref_better=int((pc[ref] < pc[m]).sum()), n_cases=len(pc),
                                wilcoxon_p=wilcoxon(pc[ref], pc[m]).pvalue,
                                dRMSE_pooled=pool(ref, ids) - pool(m, ids),
                                boot_ci_lo=lo, boot_ci_hi=hi))
    return pd.DataFrame(out)


def conformal(pred, oof):
    """Split-conformal intervals. For each outer fold the calibration scores come from the
    inner out-of-fold predictions on the training cases (models that never saw the test
    case). Normalised score s = |y - yhat| / (yhat + A_SCALE): wider bands at higher wear."""
    out = []
    ens = pred[(pred.seed == "ens") & pred.method.isin(MAIN)]
    for (prot, fold, meth), te in ens.groupby(["protocol", "fold", "method"]):
        cal = oof[(oof.protocol == prot) & (oof.fold == fold) & (oof.method == meth)].dropna(
            subset=["VB"])
        yhat = np.maximum(cal.pred.to_numpy(), 0)
        s = np.sort(np.abs(cal.VB.to_numpy() - cal.pred.to_numpy()) / (yhat + A_SCALE))
        k = min(len(s), int(np.ceil((len(s) + 1) * (1 - ALPHA))))
        q = s[k - 1]
        te = te.copy()
        half = q * (np.maximum(te.pred, 0) + A_SCALE)
        te["lo"], te["hi"], te["q"], te["n_cal"] = np.maximum(te.pred - half, 0), te.pred + half, q, len(s)
        out.append(te)
    return pd.concat(out)


def interval_table(iv):
    lab = iv.dropna(subset=["VB"])
    out = []
    for (prot, meth), g in lab.groupby(["protocol", "method"]):
        cov = (g.VB >= g.lo) & (g.VB <= g.hi)
        out.append(dict(protocol=prot, method=meth, target_pct=100 * (1 - ALPHA),
                        coverage_pct=100 * cov.mean(),
                        min_case_coverage_pct=100 * cov.groupby(g.case).mean().min(),
                        mean_width_mm=(g.hi - g.lo).mean(), mean_q=g.q.mean()))
    return pd.DataFrame(out)


def tool_change(iv):
    """Replace the insert at the first pass where the criterion reaches VB_LIM.
    'point' uses the prediction, 'upper' the upper conformal bound.
    True wear at any time = linear interpolation of the measured VB (from VB(0) = 0)."""
    out = []
    for (prot, meth, case), g in iv.groupby(["protocol", "method", "case"]):
        g = g.sort_values("t")
        lab = g.dropna(subset=["VB"])
        tt, vv = np.r_[0.0, lab.t], np.r_[0.0, lab.VB]
        reach = vv.max() >= VB_LIM
        if reach:
            i = int(np.argmax(vv >= VB_LIM))
            t_true = tt[i - 1] + (VB_LIM - vv[i - 1]) / (vv[i] - vv[i - 1]) * (tt[i] - tt[i - 1])
        for policy, crit in (("point", g.pred), ("upper", g.hi)):
            hit = np.flatnonzero(crit.to_numpy() >= VB_LIM)
            t_dec = g.t.iloc[hit[0]] if len(hit) else np.nan
            vb_dec = np.interp(t_dec, tt, vv) if len(hit) else vv[-1]
            out.append(dict(protocol=prot, method=meth, case=case, policy=policy,
                            reaches_limit=reach, t_true=t_true if reach else np.nan,
                            t_decision=t_dec, VB_true_at_decision=vb_dec,
                            late=bool(vb_dec > VB_LIM + 1e-9),
                            life_used=(t_dec / t_true) if reach and len(hit) else np.nan))
    d = pd.DataFrame(out)
    s = d.groupby(["protocol", "method", "policy"]).agg(
        n_cases=("case", "count"), n_late=("late", "sum"),
        mean_VB_at_decision=("VB_true_at_decision", "mean"),
        max_VB_at_decision=("VB_true_at_decision", "max"),
        mean_life_used=("life_used", "mean")).reset_index()
    return d, s


def main():
    os.makedirs(TAB, exist_ok=True)
    pred = pd.read_csv(os.path.join(RAW, "predictions.csv"), dtype={"seed": str})
    oof = pd.read_csv(os.path.join(RAW, "inner_oof.csv"), dtype={"seed": str})
    sel = pd.read_csv(os.path.join(RAW, "selected_cfg.csv"))
    cases = W.load_cases()
    pred, oof = add_selected(pred, oof, sel)

    ov = overall(pred)
    order = {m: i for i, m in enumerate(MAIN + ABL[1:])}
    ov = ov.sort_values(["protocol", "method"], key=lambda s: s.map(order).fillna(-1) if s.name == "method" else s)
    ov[ov.method.isin(MAIN)].to_csv(os.path.join(TAB, "T2_overall_metrics.csv"), index=False)
    ov[ov.method.isin(ABL)].to_csv(os.path.join(TAB, "T6_ablation.csv"), index=False)
    for prot in ("LOCO", "LOCD"):
        per_case(pred, prot).to_csv(os.path.join(TAB, f"T3_per_case_RMSE_{prot}.csv"))
    per_material(pred, cases).to_csv(os.path.join(TAB, "T4_per_material.csv"), index=False)
    tests = paired_tests(pred)
    tests.to_csv(os.path.join(TAB, "T5_paired_tests.csv"), index=False)
    iv = conformal(pred, oof)
    iv.to_csv(os.path.join(RAW, "intervals.csv"), index=False)
    it = interval_table(iv)
    it.to_csv(os.path.join(TAB, "T7_conformal_intervals.csv"), index=False)
    detail, tc = tool_change(iv)
    detail.to_csv(os.path.join(TAB, "T8_tool_change_per_case.csv"), index=False)
    tc.to_csv(os.path.join(TAB, "T8_tool_change_summary.csv"), index=False)
    sel.groupby(["protocol", "method", "cfg"]).size().rename("n_folds").reset_index().to_csv(
        os.path.join(TAB, "T9_selected_configurations.csv"), index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    for name, t in (("OVERALL", ov), ("PAIRED TESTS (reference - other; negative = reference better)", tests),
                    ("CONFORMAL 90 % INTERVALS", it), (f"TOOL CHANGE AT VB = {VB_LIM} mm", tc)):
        print(f"\n=== {name} ===\n{t.to_string(index=False)}")
    print("\n=== PER-CASE RMSE, LOCO ===\n", per_case(pred, "LOCO").round(3).to_string())


if __name__ == "__main__":
    main()
