"""Preprocessing of the BatICM on-road charging data (Deng et al., Applied Energy, 2023).

Steps: charging-event slicing (time gap > 10 s), filtering of short events and
SOC jumps, capacity extraction with the SOC-normalised Ampere integral,
SOH = capacity / 145 Ah, and monthly aggregation (mean, sum and standard
deviation of every event statistic).  ``lowess_smooth`` provides the
non-causal LOWESS target that the study uses only for comparison.

Place the vehicle CSV files in data/BatICM/vehicles/ and run
    python -m qsoh.data
which writes data/processed/baticm_monthly.pkl.
"""
from __future__ import annotations

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

NOMINAL_CAPACITY_AH = 145.0          # BAIC EU500 with CATL NCM 90s pack
SAMPLING_INTERVAL_S = 8              # paper reports 8 s
CHARGE_GAP_THRESHOLD_S = 10          # > 10 s gap => new charging event
MIN_EVENT_LEN = 100                  # samples
SOC_DELTA_MIN = 20.0                 # need >=20% SOC change for a usable event
CURRENT_LOW_BAND = -75.0             # paper retains [-75,-65] A core charging
CURRENT_HIGH_BAND = -65.0

COLUMNS = [
    "idx", "record_time", "soc", "pack_voltage", "charge_current",
    "max_cell_voltage", "min_cell_voltage", "max_temperature",
    "min_temperature", "available_energy", "available_capacity",
]


def _load_vehicle_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    df.columns = COLUMNS
    df = df.drop(columns=["idx"])
    df["record_time"] = pd.to_datetime(df["record_time"], format="%Y%m%d%H%M%S",
                                       errors="coerce")
    df = df.dropna(subset=["record_time"]).sort_values("record_time").reset_index(drop=True)
    return df


def slice_charging_events(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a vehicle's recording into individual charging events."""
    dt = df["record_time"].diff().dt.total_seconds().fillna(0).values
    gap_idx = np.where(dt > CHARGE_GAP_THRESHOLD_S)[0]
    starts = np.r_[0, gap_idx]
    ends = np.r_[gap_idx, len(df)]
    events: list[pd.DataFrame] = []
    for s, e in zip(starts, ends):
        if e - s < MIN_EVENT_LEN:
            continue
        seg = df.iloc[s:e].reset_index(drop=True)
        d_soc = np.diff(seg["soc"].values)
        if np.any(d_soc > 2) or np.any(d_soc < -0.1):
            continue
        events.append(seg)
    return events


def event_capacity(seg: pd.DataFrame) -> float:
    """SOC-normalized Ampere integral (Eq. 1 from Deng 2023 / Wang & Kebede 2026)."""
    cur = seg["charge_current"].values.astype(float)
    if np.isnan(cur).sum() > len(cur) * 0.1:
        return 0.0
    cur = pd.Series(cur).ffill().bfill().values
    t_sec = (seg["record_time"] - seg["record_time"].iloc[0]).dt.total_seconds().values
    q_acc = -trapezoid(cur, t_sec) / 3600.0
    d_soc = seg["soc"].iloc[-1] - seg["soc"].iloc[0]
    if d_soc < SOC_DELTA_MIN:
        return 0.0
    return float(q_acc / d_soc * 100.0)


def event_features(seg: pd.DataFrame) -> dict:
    """Compute per-event statistical features (subset used across both papers)."""
    cell_vd = seg["max_cell_voltage"] - seg["min_cell_voltage"]
    cell_td = seg["max_temperature"] - seg["min_temperature"]
    return dict(
        time_start=seg["record_time"].iloc[0],
        time_end=seg["record_time"].iloc[-1],
        soc_s=float(seg["soc"].iloc[0]),
        soc_e=float(seg["soc"].iloc[-1]),
        I_ave=float(seg["charge_current"].mean()),
        I_std=float(seg["charge_current"].std()),
        I_sum=float(seg["charge_current"].sum()),
        Vpack_ave=float(seg["pack_voltage"].mean()),
        Vpack_std=float(seg["pack_voltage"].std()),
        Vpack_sum=float(seg["pack_voltage"].sum()),
        SOC_std=float(seg["soc"].std()),
        SOC_sum=float(seg["soc"].sum()),
        Vmax_sum=float(seg["max_cell_voltage"].sum()),
        Vmin_sum=float(seg["min_cell_voltage"].sum()),
        Tmax_ave=float(seg["max_temperature"].mean()),
        Tmin_ave=float(seg["min_temperature"].mean()),
        Tmax_sum=float(seg["max_temperature"].sum()),
        Vd_ave=float(cell_vd.mean()),
        Td_sum=float(cell_td.sum()),
        Vstart=float(seg["pack_voltage"].iloc[0]),
        Vend=float(seg["pack_voltage"].iloc[-1]),
    )


def process_vehicle(csv_path: str) -> pd.DataFrame:
    df = _load_vehicle_csv(csv_path)
    events = slice_charging_events(df)
    rows = []
    for seg in events:
        cap = event_capacity(seg)
        if cap <= 0 or cap > 200:
            continue
        feat = event_features(seg)
        feat["capacity"] = cap
        feat["soh"] = cap / NOMINAL_CAPACITY_AH
        rows.append(feat)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values("time_end").reset_index(drop=True)
    return out


def aggregate_monthly(events_df: pd.DataFrame) -> pd.DataFrame:
    """Deng 2023: aggregate per-event features into monthly mean/sum/std."""
    if len(events_df) == 0:
        return events_df
    df = events_df.copy()
    df["year_month"] = df["time_end"].dt.to_period("M").dt.to_timestamp()
    cap_med = df.groupby("year_month")["capacity"].median()
    feat_cols = [c for c in df.columns if c not in {"time_start", "time_end",
                                                    "year_month", "capacity", "soh"}]
    agg = df.groupby("year_month")[feat_cols].agg(["mean", "std", "sum"])
    agg.columns = [f"{c[0]}__{c[1]}" for c in agg.columns]
    agg["capacity_monthly"] = cap_med
    agg["soh_monthly"] = cap_med / NOMINAL_CAPACITY_AH
    agg = agg.reset_index()
    return agg


def lowess_smooth(y: np.ndarray, frac: float = 0.08) -> np.ndarray:
    """Tricube-weighted local-linear smoothing (Wang & Kebede 2026)."""
    if len(y) < 5:
        return y
    x = np.arange(len(y), dtype=float)
    z = lowess(y, x, frac=frac, it=2, return_sorted=False)
    return z


def run_all(vehicles_dir: str, out_path: str) -> pd.DataFrame:
    """Process all 20 vehicles and concatenate monthly-aggregated features."""
    paths = sorted(glob.glob(os.path.join(vehicles_dir, "#*.csv")),
                   key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))
    all_events = []
    all_monthly = []
    for p in paths:
        vid = int(os.path.basename(p)[1:].split(".")[0])
        ev = process_vehicle(p)
        if len(ev) == 0:
            print(f"[veh {vid:02d}] no valid events")
            continue
        ev["vehicle_id"] = vid
        mo = aggregate_monthly(ev)
        mo["vehicle_id"] = vid
        all_events.append(ev)
        all_monthly.append(mo)
        print(f"[veh {vid:02d}] {len(ev)} events, {len(mo)} months, "
              f"SOH {ev['soh'].min():.3f}..{ev['soh'].max():.3f}")
    events_df = pd.concat(all_events, ignore_index=True)
    monthly_df = pd.concat(all_monthly, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"events": events_df, "monthly": monthly_df}, f)
    print(f"Saved -> {out_path}  events={len(events_df)}  monthly={len(monthly_df)}")
    return monthly_df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    veh_dir = os.path.join(base, "data", "BatICM", "vehicles")
    out = os.path.join(base, "data", "processed", "baticm_monthly.pkl")
    run_all(veh_dir, out)
