"""Models and helpers for the hybrid wear-rate study on the NASA Ames milling data.

Models (all predict flank wear VB in mm, pass by pass, causally):
  EMP        empirical power law  VB = C t^Et f^Ef ap^Eap, fitted per material      (scheme a)
  EMP3       three-stage law      VB = C tau^Et exp(G tau), tau = t f^a ap^b
  RF, MLP,
  GRU        purely data-driven regressors                                           (scheme b)
  PINN-soft  MLP + soft monotonicity penalty + soft wear-rate residual               (scheme c)
  HWR        proposed hybrid wear-rate model: VB_k = sum_j dVB_phys,j * exp(h_j),
             h_j from a small network -> non-decreasing by construction
  HWR3       HWR with the three-stage law as prior
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "processed", "features.csv")

# Signal descriptors chosen a priori (one level descriptor per sensor + two shape
# descriptors of the spindle current). Not selected with test labels.
LEVEL = ["AC_rms", "DC_mean", "vib_table_mean", "vib_spindle_mean",
         "AE_table_mean", "AE_spindle_mean"]
SHAPE = ["AC_kurt", "DC_satfrac"]
PARAMS = ["DOC_mm", "feed_mm_rev", "material"]
MATERIAL_NAME = {1: "cast iron", 2: "steel"}


# ----------------------------------------------------------------------------- data
def load_cases(min_passes=2):
    """Return {case: dict} with time-ordered arrays. Cases with <min_passes are dropped."""
    df = pd.read_csv(FEATURES_CSV)
    cases = {}
    for c, g in df.groupby("case"):
        g = g.sort_values(["time_min", "run"])
        if len(g) < min_passes:
            continue
        lvl = g[LEVEL].to_numpy(float)
        rel = np.log(lvl / lvl[0])                       # change w.r.t. first pass of the insert
        shp = g[SHAPE].to_numpy(float)
        cases[int(c)] = dict(
            case=int(c), run=g["run"].to_numpy(), t=g["time_min"].to_numpy(float),
            vb=g["VB_mm"].to_numpy(float),
            X=np.hstack([lvl, shp, rel, shp - shp[0]]),  # 16 signal inputs
            p=g[PARAMS].iloc[0].to_numpy(float),
            material=int(g["material"].iloc[0]),
            cond=(int(g["material"].iloc[0]), float(g["DOC_mm"].iloc[0]),
                  float(g["feed_mm_rev"].iloc[0])))
    return cases


def drop_unlabelled(case):
    m = ~np.isnan(case["vb"])
    out = dict(case)
    for k in ("run", "t", "vb", "X"):
        out[k] = case[k][m]
    return out


# ------------------------------------------------------------------ empirical law (a)
class PowerLaw:
    """log VB = log C + Et log t + Ef log f + Eap log ap, least squares, one fit per material."""

    def fit(self, cases):
        self.coef = {}
        for m in sorted({c["material"] for c in cases}):
            A, y = [], []
            for c in cases:
                if c["material"] != m:
                    continue
                ok = ~np.isnan(c["vb"]) & (c["vb"] > 0) & (c["t"] > 0)
                for t, vb in zip(c["t"][ok], c["vb"][ok]):
                    A.append([1.0, np.log(t), np.log(c["p"][1]), np.log(c["p"][0])])
                    y.append(np.log(vb))
            A, y = np.array(A), np.array(y)
            # tiny ridge keeps the fit defined when a held-out condition removes a level
            self.coef[m] = np.linalg.solve(A.T @ A + 1e-3 * np.eye(4), A.T @ y)
        return self

    def predict(self, case, t=None):
        t = case["t"] if t is None else t
        b = self.coef[case["material"]]
        scale = np.exp(b[0] + b[2] * np.log(case["p"][1]) + b[3] * np.log(case["p"][0]))
        return scale * np.power(np.maximum(t, 0.0), b[1])

    def increments(self, case):
        """Wear increment the law predicts between consecutive passes (first one from t=0)."""
        v = self.predict(case)
        return np.diff(np.concatenate([[0.0], v]))


class ThreeStageLaw(PowerLaw):
    """VB = C * tau^Et * exp(G * tau),  tau = t * f^a * ap^b  (equivalent cutting time).

    Et < 1 gives the decelerating run-in stage, G >= 0 the accelerating final stage, and
    the steady stage lies in between. G = 0 gives back the power law of PowerLaw exactly
    (Ef = a*Et, Eap = b*Et), so the two laws are nested. Et >= 0 and G >= 0 are imposed,
    hence d(VB)/dt >= 0. Fitted per material by bounded least squares on log VB.
    """

    def fit(self, cases):
        from scipy.optimize import least_squares
        start = PowerLaw().fit(cases).coef
        self.coef = {}
        for m, b0 in start.items():
            t, f, ap, y = [], [], [], []
            for c in cases:
                if c["material"] != m:
                    continue
                ok = ~np.isnan(c["vb"]) & (c["vb"] > 0) & (c["t"] > 0)
                t += list(c["t"][ok]); y += list(np.log(c["vb"][ok]))
                f += [c["p"][1]] * int(ok.sum()); ap += [c["p"][0]] * int(ok.sum())
            t, f, ap, y = map(np.array, (t, f, ap, y))

            def res(x):
                tau = t * f ** x[2] * ap ** x[3]
                return x[0] + x[1] * np.log(tau) + x[4] * tau - y

            et = float(np.clip(b0[1], 0.1, 3.0))
            x0 = [b0[0], et, np.clip(b0[2] / et, -4, 4), np.clip(b0[3] / et, -4, 4), 0.0]
            sol = least_squares(res, x0, bounds=([-20, 0.05, -5, -5, 0.0], [20, 3.0, 5, 5, 5.0]))
            self.coef[m] = sol.x
        return self

    def predict(self, case, t=None):
        t = case["t"] if t is None else t
        x = self.coef[case["material"]]
        tau = np.maximum(t, 0.0) * case["p"][1] ** x[2] * case["p"][0] ** x[3]
        return np.exp(x[0]) * np.power(tau, x[1]) * np.exp(x[4] * tau)


# ------------------------------------------------------------------------ tensors
class Scaler:
    def __init__(self, inputs="full"):
        self.inputs = inputs

    def fit(self, cases):
        X = np.vstack([c["X"] for c in cases])
        P = np.vstack([c["p"] for c in cases])
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-8
        self.mp, self.sp = P.mean(0), P.std(0) + 1e-8
        self.tmax = max(c["t"].max() for c in cases)
        return self

    def static(self, case):
        """Per-pass inputs, z-scored: signal descriptors + 3 cutting parameters.
        inputs="full": 16 descriptors (absolute levels and change since the first pass);
        inputs="rel" :  8 descriptors (change since the first pass only)."""
        x = (case["X"] - self.mx) / self.sx
        if self.inputs == "rel":
            x = x[:, 8:]
        p = np.tile((case["p"] - self.mp) / self.sp, (len(x), 1))
        return np.hstack([x, p])


def pad(arrs, fill=0.0):
    T = max(len(a) for a in arrs)
    out = np.full((len(arrs), T) + arrs[0].shape[1:], fill, dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return torch.tensor(out)


def batch(cases, scaler, law):
    S = pad([scaler.static(c) for c in cases])
    t = pad([c["t"][:, None] / scaler.tmax for c in cases])
    vb = pad([np.nan_to_num(c["vb"], nan=0.0) for c in cases])
    lab = pad([(~np.isnan(c["vb"])).astype(float) for c in cases])
    valid = pad([np.ones(len(c["t"])) for c in cases])
    dphys = pad([law.increments(c) for c in cases])
    return dict(S=S, t=t, vb=vb, lab=lab, valid=valid, dphys=dphys)


# ------------------------------------------------------------------------- networks
def mlp(n_in, hidden=16, n_out=1):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


class DirectMLP(nn.Module):
    """Scheme (b): VB = NN(signals, cutting parameters, time). Also the body of PINN-soft."""

    def __init__(self, n_static, hidden=16):
        super().__init__()
        self.net = mlp(n_static + 1, hidden)

    def forward(self, b):
        return self.net(torch.cat([b["S"], b["t"]], -1)).squeeze(-1), None


class DirectGRU(nn.Module):
    """Scheme (b), recurrent: causal GRU over the passes of one insert."""

    def __init__(self, n_static, hidden=16):
        super().__init__()
        self.gru = nn.GRU(n_static + 1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, b):
        h, _ = self.gru(torch.cat([b["S"], b["t"]], -1))
        return self.out(h).squeeze(-1), None


class HybridWearRate(nn.Module):
    """Proposed model. For pass k:
           dVB_k = dVB_phys,k * exp(h_k),   h_k = NN(signals_k, parameters, VB_{k-1})
           VB_k  = VB_{k-1} + dVB_k,        VB_0 = 0
    exp(.) > 0 and dVB_phys >= 0, so VB is non-decreasing whatever the weights are.
    The last layer starts at zero, so training starts exactly from the empirical law.
    prior=False replaces dVB_phys by dt * r0 (no physics);  feedback=False removes VB_{k-1}.
    """

    def __init__(self, n_static, hidden=16, prior=True, feedback=True, hmax=3.0):
        super().__init__()
        self.prior, self.feedback, self.hmax = prior, feedback, hmax
        self.net = mlp(n_static + (1 if feedback else 0), hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, b):
        base = b["dphys"] if self.prior else b["dt_r0"]
        B, T = base.shape
        vb = torch.zeros(B)
        out, hs = [], []
        for k in range(T):
            inp = b["S"][:, k]
            if self.feedback:
                inp = torch.cat([inp, vb[:, None]], -1)
            h = self.hmax * torch.tanh(self.net(inp).squeeze(-1) / self.hmax)
            vb = vb + base[:, k] * torch.exp(h)
            out.append(vb)
            hs.append(h)
        return torch.stack(out, 1), torch.stack(hs, 1)


# ------------------------------------------------------------------------- training
def train(model, b, epochs=400, lr=5e-3, wd=1e-4, lam_h=0.0, g_phys=0.0, g_mon=0.0):
    """Full-batch Adam. Loss = MSE on labelled passes
         + lam_h  * mean(h^2) over ALL passes (labelled or not)        [HWR]
         + g_phys * mean((dVB_pred - dVB_phys)^2) over all pass pairs  [PINN-soft]
         + g_mon  * mean(relu(VB_k - VB_{k+1}))                        [PINN-soft]
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pair = b["valid"][:, 1:] * b["valid"][:, :-1]
    for _ in range(epochs):
        opt.zero_grad()
        pred, h = model(b)
        loss = ((pred - b["vb"]) ** 2 * b["lab"]).sum() / b["lab"].sum()
        if lam_h and h is not None:
            loss = loss + lam_h * (h ** 2 * b["valid"]).sum() / b["valid"].sum()
        if g_phys or g_mon:
            d = pred[:, 1:] - pred[:, :-1]
            if g_phys:
                loss = loss + g_phys * ((d - b["dphys"][:, 1:]) ** 2 * pair).sum() / pair.sum()
            if g_mon:
                loss = loss + g_mon * (torch.relu(-d) * pair).sum() / pair.sum()
        loss.backward()
        opt.step()
    return model


METHODS = ["EMP", "EMP3", "RF", "MLP", "GRU", "PINN-soft", "HWR", "HWR3",
           "HWR-noprior", "HWR-nofeedback", "HWR-labelled-only"]
THREE_STAGE = ("EMP3", "HWR3")      # methods built on ThreeStageLaw instead of PowerLaw

DEFAULT = dict(inputs="full", hidden=16, wd=1e-4, lam_h=0.05, epochs=400, lr=5e-3,
               g_phys=1.0, g_mon=0.5, leaf=2)

# Small hyper-parameter grids, searched by inner leave-one-case-out on the training cases only.
GRIDS = {
    "EMP": [dict()],
    "EMP3": [dict()],
    "RF": [dict(inputs=i, leaf=l) for i in ("full", "rel") for l in (1, 3)],
    "MLP": [dict(inputs=i, wd=w) for i in ("full", "rel") for w in (1e-4, 1e-2)],
    "GRU": [dict(inputs=i, wd=w) for i in ("full", "rel") for w in (1e-4, 1e-2)],
    "PINN-soft": [dict(inputs=i, g_phys=g) for i in ("full", "rel") for g in (1.0, 10.0)],
    "HWR": [dict(inputs=i, lam_h=l) for i in ("full", "rel") for l in (0.02, 0.1, 0.5)],
}
for _m in ("HWR3", "HWR-noprior", "HWR-nofeedback", "HWR-labelled-only"):
    GRIDS[_m] = GRIDS["HWR"]


def fit_predict(method, train_cases, test_cases, seed=0, cfg=None):
    """Fit `method` on train_cases; return one array of VB predictions per test case."""
    torch.set_num_threads(1)
    cfg = {**DEFAULT, **(cfg or {})}
    law = (ThreeStageLaw() if method in THREE_STAGE else PowerLaw()).fit(train_cases)
    if method in ("EMP", "EMP3"):
        return [law.predict(c) for c in test_cases]
    scaler = Scaler(cfg["inputs"]).fit(train_cases)
    if method == "RF":
        def tab(cs):
            return np.vstack([np.hstack([scaler.static(c), c["t"][:, None]]) for c in cs])
        lab = np.concatenate([~np.isnan(c["vb"]) for c in train_cases])
        y = np.concatenate([c["vb"] for c in train_cases])
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=cfg["leaf"],
                                   random_state=seed)
        rf.fit(tab(train_cases)[lab], y[lab])
        return [rf.predict(tab([c])) for c in test_cases]

    torch.manual_seed(seed)
    tr_cases = [drop_unlabelled(c) for c in train_cases] if method == "HWR-labelled-only" \
        else train_cases
    btr, bte = batch(tr_cases, scaler, law), batch(test_cases, scaler, law)
    n_static = btr["S"].shape[-1]
    kw = dict(epochs=cfg["epochs"], lr=cfg["lr"], wd=cfg["wd"])
    if method == "MLP":
        model = DirectMLP(n_static, cfg["hidden"])
    elif method == "GRU":
        model = DirectGRU(n_static, cfg["hidden"])
    elif method == "PINN-soft":
        model = DirectMLP(n_static, cfg["hidden"])
        kw.update(g_phys=cfg["g_phys"], g_mon=cfg["g_mon"])
    else:
        model = HybridWearRate(n_static, cfg["hidden"], prior=method != "HWR-noprior",
                               feedback=method != "HWR-nofeedback")
        kw.update(lam_h=cfg["lam_h"])
        if method == "HWR-noprior":   # constant reference rate instead of the empirical law
            num = sum(np.nanmax(c["vb"]) for c in train_cases)
            r0 = num / sum(c["t"][~np.isnan(c["vb"])].max() for c in train_cases)
            for bb, cs in ((btr, tr_cases), (bte, test_cases)):
                bb["dt_r0"] = pad([np.diff(np.concatenate([[0.0], c["t"]])) * r0 for c in cs])
    train(model, btr, **kw)
    with torch.no_grad():
        pred = model(bte)[0].numpy()
    return [pred[i, :len(c["t"])].astype(float) for i, c in enumerate(test_cases)]


# -------------------------------------------------------------------------- metrics
def metrics(y, yhat):
    e = yhat - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(np.abs(e).mean()),
                R2=float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan)
