"""Train MQAttn-QLSTM-VQR on one fold of the vehicle data (causal target,
1-month horizon) with the learning-rate selection of experiments.run_comparison,
and save everything needed to reuse it: weights, scaler, residual offset,
chosen learning rate and test MAE.

Example (from the repository root):
    python -m experiments.train_checkpoints --fold 3
Output: checkpoints/mqattn_qlstm_vqr_fold3.pt
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import qsoh.models as M
from qsoh import simulator as FQ
from experiments import run_comparison as P

ROOT = P.ROOT
MODEL = "MQAttn-QLSTM-VQR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    P.DEVICE = "cpu"
    FQ.use_fast_backend(M)

    with open(os.path.join(ROOT, "data", "processed", "baticm_monthly.pkl"), "rb") as f:
        monthly = pickle.load(f)["monthly"].dropna(subset=P.FEATURES + ["soh_monthly"])
    monthly = P.make_series(monthly, "causal")
    vids = sorted(monthly["vehicle_id"].unique())
    folds = list(KFold(10, shuffle=True, random_state=args.seed).split(vids))
    fi = args.fold - 1
    tr_i, te_i = folds[fi]
    tr_v = [vids[i] for i in tr_i]; te_v = [vids[i] for i in te_i]
    rng = np.random.default_rng(args.seed + fi)
    va_v = rng.choice(tr_v, size=2, replace=False).tolist()
    core = [v for v in tr_v if v not in va_v]
    Xtr, ytr, atr, _, _ = P.windows(monthly[monthly.vehicle_id.isin(core)])
    Xva, yva, ava, _, _ = P.windows(monthly[monthly.vehicle_id.isin(va_v)])
    Xte, yte, ate, _, _ = P.windows(monthly[monthly.vehicle_id.isin(te_v)])
    F = Xtr.shape[-1]
    sc = StandardScaler().fit(Xtr.reshape(-1, F))
    z = lambda X: sc.transform(X.reshape(-1, F)).reshape(X.shape).astype(np.float32)  # noqa: E731
    rtr, rva = ytr - atr, yva - ava
    mu = float(rtr.mean())
    make = P.model_zoo(F)[MODEL][0]

    best = None
    for lr in (1e-3, 3e-3, 1e-2):
        v, pred, npar, model = P.train_eval(make, z(Xtr), rtr - mu, z(Xva), rva - mu, z(Xte),
                                            60, lr, args.seed + fi, return_model=True)
        if best is None or v < best[0]:
            best = (v, pred, lr, model, npar)
    _, pred, lr, model, npar = best
    mae = float(np.mean(np.abs(yte - (ate + pred + mu))))

    out_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mqattn_qlstm_vqr_fold{args.fold}.pt")
    torch.save({
        "model": MODEL, "state_dict": model.state_dict(), "params": npar,
        "config": dict(in_dim=F, hidden=16, n_qubits=4, qlstm_layers=2, attn_layers=2, vqr_layers=2),
        "features": P.FEATURES + ["soh_t (lagged causal SOH)"], "lookback": P.LOOKBACK,
        "scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "residual_offset_mu": mu,
        "learning_rate": lr, "fold": args.fold, "seed": args.seed,
        "train_vehicles": core, "val_vehicles": va_v, "test_vehicles": te_v,
        "test_MAE": mae,
        "usage": "prediction = s_{t-1} + residual_offset_mu + model(standardised window)",
    }, path)
    print(f"fold {args.fold}: lr={lr:g} params={npar} test MAE={mae*100:.3f}% -> {path}")


if __name__ == "__main__":
    main()
