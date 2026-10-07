"""Main comparison: every model on one dataset, target and horizon.

Protocol
* Target: ``causal`` (backward-only tricube local-linear smoother over the last
  7 months), ``lowess_full`` (non-causal LOWESS, for comparison only) or ``raw``.
* Inputs: 6-month window of the charging features plus the causal SOH, so the
  last observed SOH s_{t-1} is an input.  Every model predicts the change from
  s_{t-1}; persistence (s_{t-1}) is reported as a baseline.
* Models: MQAttn-QLSTM-VQR (proposed), MQAttn-LSTM-VQR, QLSTM, equal-size LSTM,
  GRU and MLP, Seq2Seq, and a classical twin of each MQAttn model in which
  every circuit becomes a classical layer with the same size.
* Training: the same epochs, patience and learning-rate grid for every model;
  the validation loss selects the learning rate per fold.
* Evaluation: leave-vehicle-out (or leave-cell-out) folds; MAE, RMSE, skill,
  training-scaled MASE, and paired Wilcoxon tests over held-out units.

Examples (from the repository root):
    python -m experiments.run_comparison --target causal
    python -m experiments.run_comparison --target causal --horizon 3
    python -m experiments.run_comparison --dataset nasa --horizon 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qsoh.models as M                                    # noqa: E402
from qsoh.data import lowess_smooth             # noqa: E402

FEATURES = [
    "I_ave__mean", "I_std__mean", "Vpack_sum__mean", "Vpack_std__mean",
    "SOC_std__mean", "Tmax_sum__mean", "Vd_ave__mean", "Td_sum__mean",
    "Tmax_ave__mean", "Tmin_ave__mean",
]
LOOKBACK = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def trailing_smooth(y: np.ndarray, k: int = 7) -> np.ndarray:
    """Backward-only tricube local-linear smoother evaluated at the newest point.

    out[t] depends on y[max(0, t-k+1) : t+1] only, so it is computable on-line.
    k = 7 months matches the neighbourhood of LOWESS frac=0.25 on ~28 months.
    """
    out = np.empty_like(y, dtype=float)
    for t in range(len(y)):
        lo = max(0, t - k + 1)
        yy = y[lo:t + 1].astype(float)
        if len(yy) < 3:
            out[t] = yy.mean()
            continue
        x = np.arange(len(yy), dtype=float)
        d = (x[-1] - x) / (x[-1] - x[0] + 1.0)
        w = (1 - d ** 3) ** 3
        W = np.diag(w)
        A = np.stack([np.ones_like(x), x - x[-1]], 1)
        beta = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ yy, rcond=None)[0]
        out[t] = beta[0]
    return out


def make_series(monthly: pd.DataFrame, target: str) -> pd.DataFrame:
    monthly = monthly.sort_values(["vehicle_id", "year_month"]).reset_index(drop=True)
    g = monthly.groupby("vehicle_id")["soh_monthly"]
    monthly["soh_raw"] = monthly["soh_monthly"]
    if target == "causal":
        monthly["soh_t"] = g.transform(lambda s: trailing_smooth(s.values))
    elif target == "lowess_full":
        monthly["soh_t"] = g.transform(lambda s: lowess_smooth(s.values, frac=0.25))
    elif target == "raw":
        monthly["soh_t"] = monthly["soh_raw"]
    else:
        raise ValueError(target)
    return monthly


def windows(df: pd.DataFrame, L: int = LOOKBACK, h: int = 1):
    """X: months t-L..t-1 of [features, soh_t]; target y_{t+h-1}; anchor s_{t-1}."""
    Xs, ys, anc, raw, veh = [], [], [], [], []
    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("year_month")
        F = g[FEATURES].values.astype(np.float32)
        s = g["soh_t"].values.astype(np.float32)
        r = g["soh_raw"].values.astype(np.float32)
        Z = np.concatenate([F, s[:, None]], 1)
        for t in range(L, len(g) - h + 1):
            Xs.append(Z[t - L:t]); ys.append(s[t + h - 1]); anc.append(s[t - 1])
            raw.append(r[t + h - 1]); veh.append(vid)
    return (np.stack(Xs), np.array(ys), np.array(anc), np.array(raw),
            np.array(veh))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ClassicalTwin(nn.Module):
    """Drop-in classical replacement for one VQC: R^n -> [-1,1]^n.

    tanh(Wx+b) * a  has n^2 + 2n parameters = 24 at n=4, exactly the
    parameter count of the 2-layer, 4-qubit ansatz (2 x 4 x 3).
    """
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.lin = nn.Linear(n_qubits, n_qubits)
        self.scale = nn.Parameter(torch.ones(n_qubits))

    def forward(self, x):
        return torch.tanh(self.lin(x)) * self.scale


def build_twin(builder, **kw):
    orig = M._make_vqc
    M._make_vqc = lambda n_qubits, n_layers, name="": ClassicalTwin(n_qubits, n_layers)
    try:
        return builder(**kw)
    finally:
        M._make_vqc = orig


def model_zoo(in_dim: int, nq: int = 4, lv: int = 2):
    depth = dict(qlstm_layers=lv, attn_layers=lv, vqr_layers=lv)
    hyb = dict(in_dim=in_dim, hidden=8, n_qubits=nq, use_qlstm=False, **depth)
    full = dict(in_dim=in_dim, hidden=16, n_qubits=nq, **depth)
    return {
        # quantum models and their parameter-identical classical twins
        "MQAttn-LSTM-VQR": (lambda: M.MQAttn_QLSTM_VQR(**hyb), True),
        "MQAttn-LSTM-VQR classical twin": (lambda: build_twin(M.MQAttn_QLSTM_VQR, **hyb), False),
        "MQAttn-QLSTM-VQR": (lambda: M.MQAttn_QLSTM_VQR(**full), True),
        "MQAttn-QLSTM-VQR classical twin": (lambda: build_twin(M.MQAttn_QLSTM_VQR, **full), False),
        "QLSTM": (lambda: M.QLSTM(in_dim, hidden=16, n_qubits=nq, n_layers=lv), True),
        # parameter-matched (~1k) classical baselines
        "LSTM": (lambda: M.LSTMRegressor(in_dim, hidden=10), False),
        "GRU": (lambda: M.GRURegressor(in_dim, hidden=12), False),
        "MLP": (lambda: M.MLPRegressor(in_dim, hidden=14, window=LOOKBACK), False),
        # large classical reference
        "Seq2Seq": (lambda: M.Seq2SeqRegressor(in_dim, hidden=32), False),
    }


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training (identical budget for every model)
# ---------------------------------------------------------------------------
def train_eval(make, Xtr, ytr, Xva, yva, Xte, epochs, lr, seed, patience=15,
               return_model=False):
    torch.manual_seed(seed)
    model = make().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    warm = max(2, epochs // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm
        else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    loss_fn = nn.HuberLoss(delta=0.01)
    T = lambda a: torch.from_numpy(a.astype(np.float32)).to(DEVICE)  # noqa: E731
    Xtr_t, ytr_t, Xva_t, yva_t, Xte_t = T(Xtr), T(ytr), T(Xva), T(yva), T(Xte)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(len(Xtr_t), device=DEVICE)
        for s in range(0, len(idx), 64):
            b = idx[s:s + 64]
            opt.zero_grad()
            loss_fn(model(Xtr_t[b]), ytr_t[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva_t), yva_t).item()
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).cpu().numpy()
    if return_model:
        return best, pred, n_params(model), model
    return best, pred, n_params(model)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="causal", choices=["causal", "lowess_full", "raw"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lrs", default="1e-3,3e-3,1e-2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_veh", type=int, default=0,
                    help="cap on training vehicles per fold (small-data study); 0 = all")
    ap.add_argument("--models", default="", help="comma-separated subset of model names")
    ap.add_argument("--horizon", type=int, default=1, help="months ahead")
    ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--lv", type=int, default=2)
    ap.add_argument("--reupload", action="store_true")
    ap.add_argument("--backend", default="fast", choices=["fast", "pennylane"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="baticm", choices=["baticm", "nasa", "oxford"])
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    torch.set_num_threads(args.threads)
    if args.backend == "fast":
        from qsoh.simulator import use_fast_backend
        use_fast_backend(M, reupload=args.reupload)
    lrs = [float(x) for x in args.lrs.split(",")]

    global FEATURES
    pkl = {"baticm": "baticm_monthly.pkl", "nasa": "nasa_cycles.pkl",
           "oxford": "oxford_cycles.pkl"}[args.dataset]
    with open(os.path.join(ROOT, "data", "processed", pkl), "rb") as f:
        blob = pickle.load(f)
    FEATURES = blob.get("features", FEATURES)
    monthly = blob["monthly"].dropna(subset=FEATURES + ["soh_monthly"])
    monthly = make_series(monthly, args.target)
    vids = sorted(monthly["vehicle_id"].unique())
    n_folds = min(10, len(vids))
    n_val = 2 if len(vids) >= 15 else 1
    folds = list(KFold(n_folds, shuffle=True, random_state=args.seed).split(vids))
    zoo = model_zoo(len(FEATURES) + 1, args.nq, args.lv)
    if args.models:
        keep = [m.strip() for m in args.models.split(",")]
        zoo = {k: v for k, v in zoo.items() if k in keep}

    tag = (f"{args.dataset}_{args.target}_h{args.horizon}_q{args.nq}l{args.lv}"
           f"{'_ru' if args.reupload else ''}_ntr{args.n_train_veh or 'all'}_s{args.seed}{args.tag}")
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    rows = []                                # one row per (model, test vehicle window)
    params, chosen_lr = {}, {}

    for fi, (tr_i, te_i) in enumerate(folds):
        tr_v = [vids[i] for i in tr_i]
        te_v = [vids[i] for i in te_i]
        rng = np.random.default_rng(args.seed + fi)
        va_v = rng.choice(tr_v, size=n_val, replace=False).tolist()
        core = [v for v in tr_v if v not in va_v]
        if args.n_train_veh:
            core = rng.choice(core, size=args.n_train_veh, replace=False).tolist()
        Xtr, ytr, atr, _, _ = windows(monthly[monthly.vehicle_id.isin(core)], h=args.horizon)
        Xva, yva, ava, _, _ = windows(monthly[monthly.vehicle_id.isin(va_v)], h=args.horizon)
        Xte, yte, ate, rte, vte = windows(monthly[monthly.vehicle_id.isin(te_v)], h=args.horizon)
        F = Xtr.shape[-1]
        sc = StandardScaler().fit(Xtr.reshape(-1, F))
        z = lambda X: sc.transform(X.reshape(-1, F)).reshape(X.shape).astype(np.float32)  # noqa: E731
        Xtr_n, Xva_n, Xte_n = z(Xtr), z(Xva), z(Xte)
        # residual over persistence, centred on the training mean residual
        rtr, rva = ytr - atr, yva - ava
        mu = float(rtr.mean())
        naive_train = float(np.mean(np.abs(rtr)))        # Hyndman MASE scale (train)

        base = dict(fold=fi + 1, y=yte, anchor=ate, raw=rte, veh=vte,
                    naive_train=naive_train)
        rows.append(dict(base, model="Persistence", pred=ate))
        print(f"\n== fold {fi+1}/{n_folds}  train={len(core)} veh ({len(Xtr)} win)  "
              f"test={te_v}  persistence MAE={np.mean(np.abs(yte-ate))*100:.3f}%")

        for name, (make, is_q) in zoo.items():
            t0 = time.time()
            best = None
            for lr in lrs:
                v, pred, npar = train_eval(make, Xtr_n, rtr - mu, Xva_n, rva - mu,
                                           Xte_n, args.epochs, lr, args.seed + fi)
                if best is None or v < best[0]:
                    best = (v, pred, lr)
            yhat = ate + best[1] + mu
            params[name] = npar
            chosen_lr.setdefault(name, []).append(best[2])
            rows.append(dict(base, model=name, pred=yhat))
            print(f"   {name:26s} {npar:6d}p  lr={best[2]:.0e}  "
                  f"MAE={np.mean(np.abs(yte-yhat))*100:.3f}%  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ summary
    recs = []
    for r in rows:
        for y, p, a, raw, v in zip(r["y"], r["pred"], r["anchor"], r["raw"], r["veh"]):
            recs.append(dict(model=r["model"], fold=r["fold"], veh=v, y=y, pred=p,
                             anchor=a, raw=raw, naive_train=r["naive_train"]))
    df = pd.DataFrame(recs)
    df["ae"] = (df.y - df.pred).abs()
    df["ae_raw"] = (df.raw - df.pred).abs()
    df["ae_pers"] = (df.y - df.anchor).abs()
    df.to_csv(os.path.join(out_dir, f"{tag}_windows.csv"), index=False)

    per_veh = df.groupby(["model", "veh"]).agg(mae=("ae", "mean"),
                                                mae_pers=("ae_pers", "mean")).reset_index()
    summ = []
    for name, g in df.groupby("model"):
        fold_mae = g.groupby("fold")["ae"].mean()
        fold_mase = g.groupby("fold").apply(lambda h: h.ae.mean() / h.naive_train.iloc[0])
        summ.append(dict(
            model=name, params=params.get(name, 0),
            MAE_pct=fold_mae.mean() * 100, MAE_std=fold_mae.std() * 100,
            RMSE_pct=np.sqrt(g.groupby("fold").apply(lambda h: (h.ae ** 2).mean())).mean() * 100,
            MAE_vs_raw_pct=g.ae_raw.mean() * 100,
            skill_vs_persistence=1 - g.ae.mean() / g.ae_pers.mean(),
            MASE_train=fold_mase.mean(),
            lr_mode=(pd.Series(chosen_lr[name]).mode().iloc[0]
                     if name in chosen_lr else np.nan)))
    S = pd.DataFrame(summ).sort_values("MAE_pct")

    # vehicle-level paired Wilcoxon (n = 20 vehicles), quantum vs its twin etc.
    pv = per_veh.pivot(index="veh", columns="model", values="mae")
    pairs = [("MQAttn-LSTM-VQR", "MQAttn-LSTM-VQR classical twin"),
             ("MQAttn-QLSTM-VQR", "MQAttn-QLSTM-VQR classical twin"),
             ("MQAttn-LSTM-VQR", "LSTM"),
             ("MQAttn-LSTM-VQR", "Persistence"),
             ("MQAttn-LSTM-VQR", "Seq2Seq")]
    tests = []
    for a, b in pairs:
        if a in pv and b in pv:
            d = (pv[b] - pv[a]).dropna()
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            tests.append(dict(A=a, B=b, n_veh=len(d), mean_gain_pp=d.mean() * 100,
                              A_better_on=int((d > 0).sum()), p_two_sided=p))
    T = pd.DataFrame(tests)

    S.to_csv(os.path.join(out_dir, f"{tag}_summary.csv"), index=False)
    T.to_csv(os.path.join(out_dir, f"{tag}_vehicle_wilcoxon.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=========== SUMMARY", tag, "===========")
    print(S.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- vehicle-level paired Wilcoxon (gain>0 means A better) ---")
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
