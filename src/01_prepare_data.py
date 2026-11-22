"""Step 1 - Read the NASA Ames milling archive and build the per-pass feature table.

Input : data/raw/mill.mat          (Goebel & Agogino, NASA Ames / UC Berkeley)
Output: data/processed/features.csv         one row per usable pass
        data/processed/dataset_summary.csv  one row per case

Only numpy / scipy are needed. Run with:  python3 -I src/01_prepare_data.py
(-I = isolated mode, recommended because mill.mat is a downloaded file).
"""
import csv
import os

import numpy as np
import scipy.io as sio
from scipy.stats import kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mill.mat")
OUT = os.path.join(ROOT, "data", "processed")

FS = 250.0            # Hz, sampling rate after the analogue RMS stage (Readme.pdf)
N_NOMINAL = 9000      # samples per pass (36 s)
ADC_LIMIT = 9.99      # V, the DC current channel clips at +10 V
CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]


def steady_window(dc):
    """Indices of the steady-cutting part of a pass, found on the DC spindle current.

    The pass has an entry ramp, a plateau and an exit ramp. The plateau is the longest
    run of samples above 80 % of the (5th-95th percentile) range; 10 % is trimmed at
    each end to stay clear of the entry and exit transients.
    """
    k = 125  # 0.5 s moving average
    sm = np.convolve(dc, np.ones(k) / k, mode="same")
    lo, hi = np.percentile(sm, 5), np.percentile(sm, 95)
    above = sm > lo + 0.8 * (hi - lo)
    best, start, cur = (0, 0), None, 0
    for i, a in enumerate(np.append(above, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    a, b = best
    trim = int(0.1 * (b - a))
    return a + trim, b - trim


def pass_features(rec):
    sig = {c: rec[c].ravel().astype(float) for c in CHANNELS}
    a, b = steady_window(sig["smcDC"])
    s = {c: v[a:b] for c, v in sig.items()}
    f = {}
    ac = s["smcAC"] - s["smcAC"].mean()
    f["AC_rms"] = ac.std()
    f["AC_p2p"] = np.percentile(ac, 99.9) - np.percentile(ac, 0.1)
    f["AC_kurt"] = kurtosis(ac)
    f["DC_mean"] = s["smcDC"].mean()
    f["DC_std"] = s["smcDC"].std()
    f["DC_satfrac"] = float(np.mean(s["smcDC"] >= ADC_LIMIT))
    for c in ["vib_table", "vib_spindle", "AE_table", "AE_spindle"]:
        f[c + "_mean"] = s[c].mean()
        f[c + "_std"] = s[c].std()
        f[c + "_p95"] = np.percentile(s[c], 95)
    f["steady_len_s"] = (b - a) / FS
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    mill = sio.loadmat(RAW)["mill"][0]
    rows, dropped = [], []
    for rec in mill:
        g = lambda k: rec[k].ravel()
        case, run = int(g("case")[0]), int(g("run")[0])
        vb = float(g("VB")[0]) if g("VB").size else np.nan
        meta = dict(case=case, run=run, time_min=float(g("time")[0]), VB_mm=vb,
                    DOC_mm=float(g("DOC")[0]), feed_mm_rev=float(g("feed")[0]),
                    material=int(g("material")[0]))
        n = rec["smcAC"].shape[0]
        ok = n == N_NOMINAL and all(
            np.isfinite(rec[c]).all() and np.abs(rec[c]).max() < 20 for c in CHANNELS)
        if not ok:
            dropped.append((case, run, n, vb))
            continue
        meta.update(pass_features(rec))
        rows.append(meta)

    rows.sort(key=lambda r: (r["case"], r["time_min"], r["run"]))
    with open(os.path.join(OUT, "features.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    with open(os.path.join(OUT, "dataset_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "material", "DOC_mm", "feed_mm_rev", "n_passes", "n_labelled",
                    "n_unlabelled", "t_end_min", "VB_end_mm", "n_label_decreases"])
        for c in sorted({r["case"] for r in rows}):
            R = [r for r in rows if r["case"] == c]
            vb = np.array([r["VB_mm"] for r in R])
            lab = vb[~np.isnan(vb)]
            w.writerow([c, R[0]["material"], R[0]["DOC_mm"], R[0]["feed_mm_rev"], len(R),
                        len(lab), len(R) - len(lab), R[-1]["time_min"],
                        lab[-1] if len(lab) else "", int(np.sum(np.diff(lab) < 0))])

    lab = sum(not np.isnan(r["VB_mm"]) for r in rows)
    print(f"passes in archive : {len(mill)}")
    print(f"passes kept       : {len(rows)}  (labelled {lab}, unlabelled {len(rows) - lab})")
    print(f"passes dropped    : {dropped}  (case, run, n_samples, VB)")


if __name__ == "__main__":
    main()
