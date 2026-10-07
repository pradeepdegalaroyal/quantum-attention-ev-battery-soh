"""NASA and Oxford cell datasets in the same long format as the vehicle data
(``vehicle_id``, ``year_month`` as the ordering index, feature columns,
``soh_monthly``).  Each cell plays the role of a vehicle, so leave-one-cell-out
validation mirrors the leave-vehicle-out protocol.

NASA (Saha and Goebel, 2007): 18650 cells rated 2 Ah, constant-current /
    constant-voltage charging.  One record per discharge cycle.  Features come
    from the charge before each discharge plus the discharge and ambient
    temperatures.  A Hampel filter removes corrupted capacities; the cells with
    at least 40 valid cycles remain (10 cells at 4, 22 and 24 degC).
Oxford Battery Degradation Dataset 1 (Birkl, 2017): eight 740 mAh pouch cells
    aged with an urban drive-cycle profile at 40 degC and characterised every
    100 cycles.  Features come from the 1C characterisation charge and the
    pseudo-OCV curve.

Place the raw files in data/external/ (see README) and run
    python -m qsoh.cell_datasets
which writes data/processed/{nasa,oxford}_cycles.pkl.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018",           # 24 C
              "B0042", "B0043", "B0044",                    # 22 C
              "B0046", "B0047", "B0048",                    # 4 C
              "B0029", "B0030", "B0031", "B0032"]           # 43 C
NASA_FEATURES = ["chg_I_mean", "chg_I_std", "chg_V_mean", "chg_V_std",
                 "chg_T_mean", "chg_T_max", "chg_duration", "dis_T_mean",
                 "ambient_T"]
OXF_FEATURES = ["chg_V_mean", "chg_V_std", "chg_T_mean", "chg_T_max",
                "chg_duration", "ocv_V_mean", "ocv_dVdq_mean"]


def _hampel(x, k=5, t=3.0):
    x = pd.Series(x)
    med = x.rolling(2 * k + 1, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(2 * k + 1, center=True, min_periods=1).median()
    return (x - med).abs() <= t * 1.4826 * mad + 1e-9


def load_nasa() -> tuple[pd.DataFrame, list[str]]:
    base = os.path.join(EXT, "nasa_pcoe", "cleaned_dataset")
    meta = pd.read_csv(os.path.join(base, "metadata.csv"))
    rows = []
    for ci, cell in enumerate(NASA_CELLS):
        m = meta[(meta.battery_id == cell) & meta.type.isin(["charge", "discharge"])]
        m = m.sort_values("test_id")
        last_chg = None
        k = 0
        for _, r in m.iterrows():
            df = pd.read_csv(os.path.join(base, "data", r.filename))
            if r.type == "charge":
                c = df[df.Current_measured > 0.05]
                if len(c) < 10:
                    continue
                last_chg = dict(chg_I_mean=c.Current_measured.mean(),
                                chg_I_std=c.Current_measured.std(),
                                chg_V_mean=c.Voltage_measured.mean(),
                                chg_V_std=c.Voltage_measured.std(),
                                chg_T_mean=c.Temperature_measured.mean(),
                                chg_T_max=c.Temperature_measured.max(),
                                chg_duration=c.Time.iloc[-1] - c.Time.iloc[0])
            else:
                cap = pd.to_numeric(r.Capacity, errors="coerce")
                if last_chg is None or not np.isfinite(cap) or cap <= 0.2:
                    continue
                rows.append(dict(vehicle_id=ci + 1, cell=cell, year_month=k,
                                 capacity=cap, soh_monthly=cap / 2.0,
                                 dis_T_mean=df.Temperature_measured.mean(),
                                 ambient_T=r.ambient_temperature, **last_chg))
                k += 1
    out = pd.DataFrame(rows)
    keep = out.groupby("vehicle_id")["capacity"].transform(lambda s: _hampel(s.values).values)
    out = out[keep.astype(bool)].copy()
    out["year_month"] = out.groupby("vehicle_id").cumcount()
    out = out[out.groupby("vehicle_id")["year_month"].transform("size") >= 40]
    return out.reset_index(drop=True), NASA_FEATURES


def load_oxford() -> tuple[pd.DataFrame, list[str]]:
    import scipy.io as sio
    path = os.path.join(EXT, "oxford_degradation_1", "Oxford_Battery_Degradation_Dataset_1.mat")
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    rows = []
    for ci in range(1, 9):
        cell = d[f"Cell{ci}"]
        for k, name in enumerate(sorted(cell._fieldnames)):
            cy = getattr(cell, name)
            try:
                dc, ch, ocv = cy.C1dc, cy.C1ch, cy.OCVch
                cap = float(np.max(np.abs(np.asarray(dc.q, float))))
                v, T, t = (np.asarray(ch.v, float), np.asarray(ch.T, float),
                           np.asarray(ch.t, float))
                ov, oq = np.asarray(ocv.v, float), np.asarray(ocv.q, float)
            except AttributeError:
                continue
            dvdq = np.diff(ov) / np.maximum(np.diff(oq), 1e-6)
            rows.append(dict(vehicle_id=ci, cell=f"Cell{ci}", year_month=k,
                             cycle=int(name[3:]), capacity=cap,
                             soh_monthly=cap / 740.0,
                             chg_V_mean=v.mean(), chg_V_std=v.std(),
                             chg_T_mean=T.mean(), chg_T_max=T.max(),
                             chg_duration=(t[-1] - t[0]) * (86400 if t[-1] < 10 else 1),
                             ocv_V_mean=ov.mean(),
                             ocv_dVdq_mean=float(np.median(dvdq[np.isfinite(dvdq)]))))
    out = pd.DataFrame(rows).sort_values(["vehicle_id", "cycle"])
    out["year_month"] = out.groupby("vehicle_id").cumcount()
    return out.reset_index(drop=True), OXF_FEATURES


if __name__ == "__main__":
    for name, fn in [("nasa", load_nasa), ("oxford", load_oxford)]:
        df, feats = fn()
        with open(os.path.join(ROOT, "data", "processed", f"{name}_cycles.pkl"), "wb") as f:
            pickle.dump({"monthly": df, "features": feats}, f)
        g = df.groupby("vehicle_id").agg(n=("soh_monthly", "size"),
                                         soh0=("soh_monthly", "first"),
                                         soh_end=("soh_monthly", "last"))
        print(f"\n{name}: {len(df)} records, {df.vehicle_id.nunique()} cells")
        print(g.to_string())
        print(df[feats].describe().T[["mean", "std", "min", "max"]].to_string())
