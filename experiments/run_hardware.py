"""Hardware realism of MQAttn-QLSTM-VQR (Table 5 and Fig. 7 of the paper).

For each fold of the vehicle data (causal target, 1-month horizon):
  (a) train with simulator gradients, then evaluate the same weights under
      finite shots (256, 1024, 4096), depolarising gate noise
      (p1 = p2 / 10) and a preset noise level (p1 = 2.5e-4, p2 = 3e-3,
      readout error 1e-2, 4096 shots) with and without readout correction;
  (b) train with parameter-shift gradients estimated from 1024 shots per
      circuit, as on quantum hardware, and evaluate under the same shots.

Example (from the repository root, one process per fold):
    python -m experiments.run_hardware --fold 1
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import qsoh.models as M                                       # noqa: E402
from qsoh import simulator as FQ                             # noqa: E402
from experiments import run_comparison as P                # noqa: E402

PRESET = dict(p1=2.5e-4, p2=3e-3, readout=1e-2)
CONDITIONS = [
    ("no noise", dict()),
    ("4096 shots", dict(shots=4096)),
    ("1024 shots", dict(shots=1024)),
    ("256 shots", dict(shots=256)),
    ("depolarising p2=1e-3", dict(p1=1e-4, p2=1e-3)),
    ("depolarising p2=1e-2", dict(p1=1e-3, p2=1e-2)),
    ("depolarising p2=3e-2", dict(p1=3e-3, p2=3e-2)),
    ("depolarising p2=1e-1", dict(p1=1e-2, p2=1e-1)),
    ("preset noise, 4096 shots", dict(**PRESET, shots=4096)),
    ("preset noise, 4096 shots, readout correction", dict(**PRESET, shots=4096, mitigate=True)),
]


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MQAttn-QLSTM-VQR")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--ps_shots", type=int, default=1024)
    ap.add_argument("--skip_ps", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3, help="repeats for stochastic (shot) conditions")
    ap.add_argument("--fold", type=int, default=0, help="run a single fold (1-10); 0 = all")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    P.DEVICE = "cpu"
    FQ.use_fast_backend(M)

    with open(os.path.join(ROOT, "data", "processed", "baticm_monthly.pkl"), "rb") as f:
        monthly = pickle.load(f)["monthly"].dropna(subset=P.FEATURES + ["soh_monthly"])
    monthly = P.make_series(monthly, "causal")
    vids = sorted(monthly["vehicle_id"].unique())
    folds = list(KFold(10, shuffle=True, random_state=0).split(vids))
    make = P.model_zoo(len(P.FEATURES) + 1)[args.model][0]

    recs = []
    for fi, (tr_i, te_i) in enumerate(folds):
        if args.fold and fi + 1 != args.fold:
            continue
        tr_v = [vids[i] for i in tr_i]; te_v = [vids[i] for i in te_i]
        rng = np.random.default_rng(fi)
        va_v = rng.choice(tr_v, size=2, replace=False).tolist()
        core = [v for v in tr_v if v not in va_v]
        Xtr, ytr, atr, _, _ = P.windows(monthly[monthly.vehicle_id.isin(core)])
        Xva, yva, ava, _, _ = P.windows(monthly[monthly.vehicle_id.isin(va_v)])
        Xte, yte, ate, _, _ = P.windows(monthly[monthly.vehicle_id.isin(te_v)])
        F = Xtr.shape[-1]
        sc = StandardScaler().fit(Xtr.reshape(-1, F))
        z = lambda X: sc.transform(X.reshape(-1, F)).reshape(X.shape).astype(np.float32)  # noqa: E731
        Xtr_n, Xva_n, Xte_n = z(Xtr), z(Xva), z(Xte)
        rtr, rva = ytr - atr, yva - ava
        mu = float(rtr.mean())
        pers = float(np.mean(np.abs(yte - ate)))

        # (a) noiseless training, noisy evaluation
        FQ.set_noise(); FQ.set_grad_mode("backprop")
        t0 = time.time()
        _, _, _, model = P.train_eval(make, Xtr_n, rtr - mu, Xva_n, rva - mu, Xte_n,
                                      args.epochs, args.lr, fi, return_model=True)
        print(f"fold {fi+1}: trained (backprop) in {time.time()-t0:.0f}s, persistence MAE {pers*100:.3f}%")
        for cname, cfg in CONDITIONS:
            reps = args.repeats if cfg.get("shots") else 1
            maes = []
            for r in range(reps):
                torch.manual_seed(100 + r)
                FQ.set_noise(**cfg)
                yhat = ate + predict(model, Xte_n) + mu
                maes.append(np.mean(np.abs(yte - yhat)))
            FQ.set_noise()
            recs.append(dict(fold=fi + 1, training="simulator gradients", condition=cname,
                             MAE=float(np.mean(maes)), persistence=pers))
            print(f"    {cname:45s} MAE={np.mean(maes)*100:.3f}%")

        # (b) hardware-compatible training: parameter-shift + finite shots
        if not args.skip_ps:
            FQ.set_noise(shots=args.ps_shots); FQ.set_grad_mode("shift")
            t0 = time.time()
            _, _, _, model_ps = P.train_eval(make, Xtr_n, rtr - mu, Xva_n, rva - mu, Xte_n,
                                             args.epochs, args.lr, fi, return_model=True)
            FQ.set_grad_mode("backprop")
            maes = []
            for r in range(args.repeats):
                torch.manual_seed(200 + r)
                maes.append(np.mean(np.abs(yte - (ate + predict(model_ps, Xte_n) + mu))))
            FQ.set_noise()
            ideal = np.mean(np.abs(yte - (ate + predict(model_ps, Xte_n) + mu)))
            recs.append(dict(fold=fi + 1, training=f"parameter shift, {args.ps_shots} shots",
                             condition=f"{args.ps_shots} shots", MAE=float(np.mean(maes)),
                             persistence=pers))
            recs.append(dict(fold=fi + 1, training=f"parameter shift, {args.ps_shots} shots",
                             condition="no noise", MAE=float(ideal), persistence=pers))
            print(f"    [PS+{args.ps_shots} shots training, {time.time()-t0:.0f}s] "
                  f"eval@shots MAE={np.mean(maes)*100:.3f}%  eval@ideal MAE={ideal*100:.3f}%")

    df = pd.DataFrame(recs)
    tag = f"_fold{args.fold}" if args.fold else ""
    out = os.path.join(ROOT, "results", f"hardware{tag}.csv")
    df.to_csv(out, index=False)
    S = (df.groupby(["training", "condition"], sort=False)
           .agg(MAE_pct=("MAE", lambda s: s.mean() * 100), MAE_std=("MAE", lambda s: s.std() * 100),
                persistence_pct=("persistence", lambda s: s.mean() * 100)).reset_index())
    S["skill_vs_persistence"] = 1 - S.MAE_pct / S.persistence_pct
    S.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + S.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
