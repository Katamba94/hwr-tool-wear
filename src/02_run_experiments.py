"""Step 2 - Nested cross-validation of all models.

Two protocols:
  LOCO : leave-one-case-out          (15 folds; the replicate insert stays in training)
  LOCD : leave-one-condition-out     (8 folds; both inserts of a cutting condition held out)

For each outer fold and each method:
  1. every configuration of the small grid (wearlib.GRIDS) is scored by an inner
     leave-one-case-out on the TRAINING cases only (seed 0);
  2. the best configuration is refitted on all training cases with 5 seeds and applied
     to the held-out case(s);
  3. inner out-of-fold predictions of the selected configuration (5-seed ensemble) are
     stored: they are the calibration set of the conformal intervals (step 3).
The three HWR ablations reuse the configuration selected for HWR in the same fold.

Output (results/raw): predictions.csv, inner_oof.csv, selected_cfg.csv
Run:  python3 src/02_run_experiments.py        (about 30 min on 10 cores)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wearlib as W  # noqa: E402

OUT = os.path.join(W.ROOT, "results", "raw")
SEEDS = [0, 1, 2, 3, 4]
MAIN = ["EMP", "EMP3", "RF", "MLP", "GRU", "PINN-soft", "HWR", "HWR3"]
ABLATIONS = ["HWR-noprior", "HWR-nofeedback", "HWR-labelled-only"]


def folds(cases, protocol):
    ids = sorted(cases)
    if protocol == "LOCO":
        return [[c] for c in ids]
    conds = sorted({cases[c]["cond"] for c in ids})
    return [[c for c in ids if cases[c]["cond"] == k] for k in conds]


def rows(protocol, fold, method, seed, case, pred):
    return [dict(protocol=protocol, fold=fold, method=method, seed=seed, case=case["case"],
                 run=int(r), t=float(t), VB=float(v), pred=float(p))
            for r, t, v, p in zip(case["run"], case["t"], case["vb"], pred)]


def pooled_rmse(cs, preds):
    e = np.concatenate([(p - c["vb"])[~np.isnan(c["vb"])] for c, p in zip(cs, preds)])
    return float(np.sqrt((e ** 2).mean()))


def job(protocol, fold, test_ids, method):
    cases = W.load_cases()
    train = [cases[c] for c in sorted(cases) if c not in test_ids]
    test = [cases[c] for c in test_ids]
    seeds = [0] if method in ("EMP", "EMP3") else SEEDS

    def inner(cfg, seed):
        return [W.fit_predict(method, [c for c in train if c is not r], [r], seed, cfg)[0]
                for r in train]

    # 1. configuration selection on the training cases only
    scored = [(pooled_rmse(train, inner(cfg, 0)), i) for i, cfg in enumerate(W.GRIDS[method])]
    best_rmse, best = min(scored)
    cfg = W.GRIDS[method][best]
    sel = [dict(protocol=protocol, fold=fold, method=method, cfg=json.dumps(cfg),
                inner_rmse=best_rmse, grid_scores=json.dumps([round(s, 4) for s, _ in scored]))]

    # 2. refit on all training cases, predict the held-out case(s)
    pred_rows = []
    for m in [method] + (ABLATIONS if method == "HWR" else []):
        per_seed = [W.fit_predict(m, train, test, s, cfg) for s in seeds]
        for s, ps in zip(seeds, per_seed):
            for c, p in zip(test, ps):
                pred_rows += rows(protocol, fold, m, s, c, p)
        for i, c in enumerate(test):
            pred_rows += rows(protocol, fold, m, "ens", c, np.mean([ps[i] for ps in per_seed], 0))

    # 3. inner out-of-fold predictions of the selected configuration (seed ensemble)
    oof = np.mean([np.array(inner(cfg, s), dtype=object) for s in seeds], 0)
    oof_rows = []
    for c, p in zip(train, oof):
        oof_rows += rows(protocol, fold, method, "ens", c, p)
    return pred_rows, oof_rows, sel


def main():
    """With no argument every model is run. With model names as arguments
    (e.g. `02_run_experiments.py EMP3 HWR3`) only those are run and their rows replace
    the previous ones in the existing result files."""
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:]
    methods = only or MAIN
    cases = W.load_cases()
    jobs = [(prot, f, ids, m) for prot in ("LOCO", "LOCD")
            for f, ids in enumerate(folds(cases, prot)) for m in methods]
    jobs.sort(key=lambda j: not j[3].startswith("HWR"))          # longest jobs first
    t0 = time.time()
    res = Parallel(n_jobs=-1, verbose=5)(delayed(job)(*j) for j in jobs)
    for name, k in (("predictions", 0), ("inner_oof", 1), ("selected_cfg", 2)):
        new = pd.DataFrame([r for out in res for r in out[k]])
        path = os.path.join(OUT, name + ".csv")
        if only and os.path.exists(path):
            old = pd.read_csv(path, dtype={"seed": str})
            produced = set(new.method.unique())
            new = pd.concat([old[~old.method.isin(produced)], new.astype({"seed": str})
                             if "seed" in new else new], ignore_index=True)
        new.to_csv(path, index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
