"""Build every table (LaTeX) and figure (PDF) of the paper from the files in results/.

Example (from the repository root):
    python -m experiments.make_tables_figures --out outputs
Writes outputs/tables/*.tex and outputs/figures/*.pdf.  Missing inputs are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from scipy.stats import wilcoxon                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = FIG = ""                                                # set in main()

PROPOSED = "MQAttn-QLSTM-VQR"
TWIN = "MQAttn-QLSTM-VQR classical twin"
VARIANT = "MQAttn-LSTM-VQR"
VARIANT_TWIN = "MQAttn-LSTM-VQR classical twin"
S2S, PERS = "Seq2Seq", "Persistence"
ORDER = [PROPOSED, TWIN, "QLSTM", VARIANT, VARIANT_TWIN, "LSTM", "GRU", "MLP", S2S, PERS]
LABEL = {PROPOSED: "MQAttn-QLSTM-VQR", TWIN: "Classical twin of MQAttn-QLSTM-VQR",
         "QLSTM": "QLSTM \\cite{wang2026}", VARIANT: "MQAttn-LSTM-VQR",
         VARIANT_TWIN: "Classical twin of MQAttn-LSTM-VQR", "LSTM": "LSTM", "GRU": "GRU",
         "MLP": "MLP", S2S: "Seq2Seq \\cite{deng2023}", PERS: "Persistence"}
SETTINGS = [  # key, column label, result-file stems (several stems = several seeds)
    ("b_h1", "Vehicles, 1\\,mo", ["baticm_causal_h1_q4l2_ntrall_s0", "baticm_causal_h1_q4l2_ntrall_s1"]),
    ("b_h3", "Vehicles, 3\\,mo", ["baticm_causal_h3_q4l2_ntrall_s0"]),
    ("b_h6", "Vehicles, 6\\,mo", ["baticm_causal_h6_q4l2_ntrall_s0"]),
    ("b_raw", "Vehicles, raw SOH", ["baticm_raw_h1_q4l2_ntrall_s0"]),
    ("b_n4", "Vehicles, 4 train", ["baticm_causal_h1_q4l2_ntr4_s0"]),
    ("n_h1", "NASA, 1 cyc", ["nasa_causal_h1_q4l2_ntrall_s0"]),
    ("n_h10", "NASA, 10 cyc", ["nasa_causal_h10_q4l2_ntrall_s0"]),
    ("o_h1", "Oxford, 100 cyc", ["oxford_causal_h1_q4l2_ntrall_s0"]),
    ("o_h5", "Oxford, 500 cyc", ["oxford_causal_h5_q4l2_ntrall_s0"]),
]

# colours: the first three slots of a colour-blind-safe categorical palette + neutral ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def _p(name):
    return os.path.join(RES, name)


def _write(name, lines):
    open(os.path.join(TAB, name), "w").write("\n".join(lines))
    print(name)


def load_windows(stems):
    dfs = [pd.read_csv(_p(f"{s}_windows.csv")).assign(seed=i)
           for i, s in enumerate(stems) if os.path.exists(_p(f"{s}_windows.csv"))]
    if not dfs:
        return None
    W = pd.concat(dfs)
    W["ae"] = (W.y - W.pred).abs()
    return W


def setting_stats(stems):
    """Fold-mean MAE (averaged over seeds), MASE, skill and unit-level p-values."""
    S = [pd.read_csv(_p(f"{s}_summary.csv")) for s in stems if os.path.exists(_p(f"{s}_summary.csv"))]
    if not S:
        return None
    S = pd.concat(S).groupby("model").agg(params=("params", "first"), MAE=("MAE_pct", "mean"),
                                          MASE=("MASE_train", "mean"),
                                          skill=("skill_vs_persistence", "mean"))
    unit = load_windows(stems).groupby(["model", "veh"]).ae.mean().unstack(0)
    pv = {}
    for b in unit.columns:
        if b == PROPOSED:
            continue
        d = (unit[b] - unit[PROPOSED]).dropna()
        pv[b] = (wilcoxon(d).pvalue if (d != 0).any() else 1.0, int((d > 0).sum()), len(d))
    return S, pv


def fmt_p(p):
    return f"{p:.2f}" if p >= 0.01 else (f"{p:.3f}" if p >= 0.001 else "$<$0.001")


def fmt_delta(x):
    """Signed difference; values that round to zero print as 0.000."""
    return "0.000" if abs(x) < 0.0005 else f"${x:+.3f}$"


def fmt_skill(x):
    return f"$-${abs(x):.0f}\\%" if x < -0.5 else f"{x:.0f}\\%"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table_main():
    stats = {k: setting_stats(st) for k, _, st in SETTINGS}
    keys = [k for k, _, _ in SETTINGS if stats[k] is not None]
    lab = {k: l for k, l, _ in SETTINGS}
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Mean absolute error (MAE, \% of SOH) of every model under the causal "
             r"target.  Each column gives a dataset and a forecast horizon (mo: months; cyc: "
             r"cycles; raw SOH: unsmoothed target; 4 train: four training vehicles).  The "
             r"``Vehicles, 1\,mo'' column is the mean of two seeds; the other columns use one seed.  "
             r"Bold marks the lowest error in each column.  ``Params'' gives the parameter count on "
             r"the vehicle data.  The bottom block gives, for MQAttn-QLSTM-VQR, its rank among the "
             r"nine learned models, its skill over persistence "
             r"($1-\mathrm{MAE}/\mathrm{MAE}_{\mathrm{pers}}$), its MASE scaled on the training data "
             r"\cite{hyndman2006mase}, and paired Wilcoxon $p$-values over the held-out vehicles or "
             r"cells against persistence, its classical twin and Seq2Seq.  The numbers in brackets "
             r"count the held-out units in which MQAttn-QLSTM-VQR has the lower error.}",
             r"\label{tab:main}", r"\resizebox{\textwidth}{!}{",
             r"\begin{tabular}{lr" + "c" * len(keys) + "}", r"\toprule",
             "Model & Params & " + " & ".join(lab[k] for k in keys) + r" \\", r"\midrule"]
    best = {k: stats[k][0].drop(index=PERS).MAE.min() for k in keys}
    for m in ORDER:
        if m not in stats["b_h1"][0].index:
            continue
        cells = []
        for k in keys:
            v = stats[k][0].MAE.get(m, np.nan)
            cells.append(f"\\textbf{{{v:.3f}}}" if np.isclose(v, best[k]) else f"{v:.3f}")
        if m == PERS:
            lines.append(r"\midrule")
        lines.append(f"{LABEL[m]} & {int(stats['b_h1'][0].params.get(m, 0)):,} & "
                     + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    ranks, skills, mases, pp, pt, ps = [], [], [], [], [], []
    for k in keys:
        S, pv = stats[k]
        learned = S.drop(index=PERS).MAE.sort_values()
        ranks.append(f"{list(learned.index).index(PROPOSED) + 1}/{len(learned)}")
        skills.append(f"{S.skill[PROPOSED]*100:.0f}\\%")
        mases.append(f"{S.MASE[PROPOSED]:.2f}")
        pp.append(f"{fmt_p(pv[PERS][0])} ({pv[PERS][1]}/{pv[PERS][2]})")
        pt.append(fmt_p(pv[TWIN][0]))
        ps.append(fmt_p(pv[S2S][0]))
    for name, row in [("Rank", ranks), ("Skill over persistence", skills),
                      ("MASE (training-scaled)", mases),
                      ("$p$ vs persistence (units better)", pp),
                      ("$p$ vs classical twin", pt), ("$p$ vs Seq2Seq", ps)]:
        lines.append(f"{name} & & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    _write("tab_main.tex", lines)
    return stats


def table_protocol():
    stems = ["baticm_lowess_full_h1_q4l2_ntrall_s0", "baticm_causal_h1_q4l2_ntrall_s0",
             "baticm_raw_h1_q4l2_ntrall_s0"]
    S = [pd.read_csv(_p(f"{v}_summary.csv")).set_index("model") for v in stems]
    rows = [PROPOSED, TWIN, "QLSTM", "LSTM", S2S, PERS]
    names = {PROPOSED: "MQAttn-QLSTM-VQR", TWIN: "Classical twin", "QLSTM": "QLSTM",
             "LSTM": "LSTM", S2S: "Seq2Seq", PERS: "Persistence"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{MAE (\% of SOH) and MASE under three targets: the non-causal LOWESS "
             r"target of the earlier version, the causal target, and the raw monthly SOH (vehicle "
             r"data, 1-month horizon, seed 0).  All learned models use the last observed SOH as "
             r"an input.  ``Classical twin'' is the classical twin of MQAttn-QLSTM-VQR.}",
             r"\label{tab:protocol}", r"{\fontsize{8}{9.5}\selectfont\setlength{\tabcolsep}{2.2pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Non-causal} & \multicolumn{2}{c}{Causal} & \multicolumn{2}{c}{Raw} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & MAE & MASE & MAE & MASE & MAE & MASE \\", r"\midrule"]
    for m in rows:
        c = []
        for s in S:
            c += [f"{s.loc[m].MAE_pct:.3f}", f"{s.loc[m].MASE_train:.2f}"]
        lines.append(names[m] + " & " + " & ".join(c) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write("tab_protocol.tex", lines)


def table_ablation():
    fs = sorted(glob.glob(_p("baticm_causal_h1_q4l2_ntrall_s*_ablation_summary.csv")))
    if not fs:
        print("[skip] tab_ablation.tex"); return
    full = "Full model"
    A = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)])
    g = A.groupby("model").agg(params=("params", "first"), MAE=("MAE_pct", "mean"),
                               sd=("MAE_pct", "std"), MASE=("MASE_train", "mean"))
    W = pd.concat([pd.read_csv(f.replace("_summary", "_windows")) for f in fs])
    W["ae"] = (W.y - W.pred).abs()
    unit = W.groupby(["model", "veh"]).ae.mean().unstack(0)
    rest = [m for m in g.sort_values("MAE").index if m not in (full, PERS)]
    groups = [("Reference", [full]),
              ("Component removal", [m for m in rest if m.startswith("Without")]),
              ("Classical substitution (equal size)", [m for m in rest if not m.startswith("Without")]),
              ("Baseline", [PERS])]
    lines = [r"\begin{table*}[!t]", r"\centering",
             rf"\caption{{Ablation of MQAttn-QLSTM-VQR (vehicle data, causal target, 1-month "
             rf"horizon, {len(fs)} seeds $\times$ 10 folds).  Each variant removes a component or "
             rf"replaces a group of circuits with classical layers of the same size.  All variants "
             rf"use the same learning rate and training budget.  MAE is the mean $\pm$ standard "
             rf"deviation over seeds.  $\Delta$ is the MAE change from the full model (positive: "
             rf"the variant is worse).  ``Worse on'' counts the held-out vehicles in which the "
             rf"variant has the higher error.  $p$ comes from a paired two-sided Wilcoxon test "
             rf"over the 20 held-out vehicles, with each vehicle's error averaged over seeds.}}",
             r"\label{tab:abl}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrccccc}", r"\toprule",
             r"Type & Variant & Params & MAE (\%) & MASE & $\Delta$ (points) & Worse on & $p$ \\",
             r"\midrule"]
    for gi, (gname, ms) in enumerate(groups):
        for k, m in enumerate(ms):
            r = g.loc[m]
            head = f"{gname if k == 0 else ''} & {m} & {int(r.params)} & {r.MAE:.3f}$\\pm${r.sd:.3f} & {r.MASE:.2f}"
            if m == full:
                lines.append(head + r" & -- & -- & -- \\")
                continue
            d = (unit[m] - unit[full]).dropna()
            lines.append(head + f" & {fmt_delta(r.MAE - g.loc[full].MAE)} & "
                                f"{int((d > 0).sum())}/{len(d)} & {fmt_p(wilcoxon(d).pvalue)} \\\\")
        if gi < len(groups) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    _write("tab_ablation.tex", lines)


def table_grid():
    rows = []
    for nq, lv in [(2, 10), (3, 8), (4, 5), (4, 2)]:
        f = _p(f"baticm_causal_h1_q{nq}l{lv}_ntrall_s0_summary.csv")
        if not os.path.exists(f):
            print(f"[skip] grid ({nq},{lv})"); continue
        r = pd.read_csv(f).set_index("model").loc[PROPOSED]
        rows.append((nq, lv, int(r.params), r.MAE_pct, r.MAE_std, r.skill_vs_persistence, r.MASE_train))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{MAE of MQAttn-QLSTM-VQR for different numbers of qubits $n_q$ and "
             r"variational layers $L_v$ (vehicle data, causal target, 1-month horizon, 10 folds, "
             r"seed 0).  Skill $=1-\mathrm{MAE}/\mathrm{MAE}_{\mathrm{pers}}$.}",
             r"\label{tab:grid}", r"{\fontsize{8}{9.5}\selectfont",
             r"\begin{tabular}{ccrccc}", r"\toprule",
             r"$n_q$ & $L_v$ & Params & MAE (\%) & Skill & MASE \\", r"\midrule"]
    for nq, lv, par, mae, sd, sk, ma in rows:
        lines.append(f"{nq} & {lv} & {par} & {mae:.3f}$\\pm${sd:.3f} & {sk*100:.0f}\\% & {ma:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write("tab_grid.tex", lines)


HW_LABEL = {"no noise": "No noise", "4096 shots": "4096 shots", "1024 shots": "1024 shots",
            "256 shots": "256 shots",
            "depolarising p2=1e-3": "Depol.\\ $p_2=10^{-3}$",
            "depolarising p2=1e-2": "Depol.\\ $p_2=10^{-2}$",
            "depolarising p2=3e-2": "Depol.\\ $p_2=3\\times10^{-2}$",
            "depolarising p2=1e-1": "Depol.\\ $p_2=10^{-1}$",
            "preset noise, 4096 shots": "Preset noise, 4096 shots",
            "preset noise, 4096 shots, readout correction": "Preset noise, 4096 shots, readout correction"}


def hardware_summary():
    fs = [f for f in sorted(glob.glob(_p("hardware_fold*.csv"))) if not f.endswith("_summary.csv")]
    if len(fs) < 10:
        print(f"[skip] hardware: {len(fs)}/10 folds"); return None
    D = pd.concat([pd.read_csv(f) for f in fs])
    return (D.groupby(["training", "condition"], sort=False)
             .agg(MAE=("MAE", lambda s: s.mean() * 100),
                  pers=("persistence", lambda s: s.mean() * 100)).reset_index())


def table_hardware(S):
    if S is None:
        return
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Accuracy of MQAttn-QLSTM-VQR under finite shots, gate noise and "
             r"readout error (vehicle data, causal target, 1-month horizon, mean of 10 folds). "
             r"Depol.: depolarising noise with two-qubit error $p_2$ and single-qubit error "
             r"$p_1=p_2/10$. Preset noise: $p_1=2.5\times10^{-4}$, $p_2=3\times10^{-3}$, readout "
             r"error $10^{-2}$. Skill $=1-\mathrm{MAE}/\mathrm{MAE}_{\mathrm{pers}}$, with "
             r"persistence MAE " + f"{S.pers.iloc[0]:.3f}" + r"\%.}",
             r"\label{tab:hw}", r"\resizebox{\columnwidth}{!}{",
             r"\begin{tabular}{llcc}", r"\toprule",
             r"Training & Evaluation & MAE (\%) & Skill \\", r"\midrule"]
    for _, r in S.iterrows():
        tr = "Simulator gradients" if r.training.startswith("simulator") else "Parameter shift, 1024 shots"
        lines.append(f"{tr} & {HW_LABEL.get(r.condition, r.condition)} & {r.MAE:.3f} & "
                     f"{fmt_skill((1 - r.MAE / r.pers) * 100)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write("tab_hardware.tex", lines)


def table_cost():
    J = json.load(open(_p("cost.json")))
    order = ["Seq2Seq", "LSTM", "QLSTM", "MQAttn-QLSTM-VQR", "MQAttn-LSTM-VQR"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Cost of one prediction (6-month window, 11 inputs). Weights use 32-bit "
             r"floating-point numbers. FLOPs count floating-point operations; for the quantum "
             r"models they include the exact classical simulation of every 4-qubit circuit "
             + f"({J['flops_per_circuit']:,} FLOPs per circuit). " +
             r"Training on quantum hardware with the parameter-shift rule needs "
             + f"{J['parameter_shift']['circuit_evaluations_per_window_and_step']:,} circuit "
             + r"evaluations per window and update step.}",
             r"\label{tab:cost}", r"\resizebox{\columnwidth}{!}{",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Parameters & Weights (kB) & FLOPs & Circuits \\", r"\midrule"]
    for k in order:
        d = J["models"][k]
        lines.append(f"{k} & {d['params']:,} & {d['weight_kB_fp32']:.1f} & "
                     f"{int(d['flops_per_prediction']):,} & {d['circuits_per_prediction']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write("tab_cost.tex", lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_horizon(stats):
    hz, keys = [1, 3, 6], ["b_h1", "b_h3", "b_h6"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    series = [(PROPOSED, "MQAttn-QLSTM-VQR (988 parameters)", C1, "o", -0.12),
              (TWIN, "Classical twin (988 parameters)", C2, "s", 0.0),
              (S2S, "Seq2Seq (33,345 parameters)", C3, "^", 0.12),
              (PERS, "Persistence", MUTED, "D", 0.0)]
    for m, lab, c, mk, dx in series:
        ax.plot([h + dx for h in hz], [stats[k][0].MAE[m] for k in keys], color=c, lw=1.5,
                marker=mk, ms=5, mec="white", mew=1, label=lab, zorder=4 if m == PROPOSED else 3)
    ax.text(3.6, 0.55, "learned models statistically\nequal ($p\\geq0.41$)", fontsize=6.5,
            color=INK2, va="top")
    ax.set_xticks(hz); ax.set_xticklabels(["1", "3", "6"])
    ax.set_xlabel("Forecast horizon (months)"); ax.set_ylabel("MAE (% of SOH)")
    ax.set_ylim(0, None); ax.legend(loc="upper left", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_horizon.pdf")); plt.close(fig)
    print("fig_horizon.pdf")


def fig_data_efficiency(stats):
    a, b = stats["b_h1"][0], stats["b_n4"][0]
    ms = sorted([m for m in ORDER if m not in (PERS, "MLP") and m in b.index], key=lambda m: b.MAE[m])
    short = {PROPOSED: "MQAttn-QLSTM-VQR", TWIN: "Classical twin of MQAttn-QLSTM-VQR",
             "QLSTM": "QLSTM", VARIANT: "MQAttn-LSTM-VQR",
             VARIANT_TWIN: "Classical twin of MQAttn-LSTM-VQR", "LSTM": "LSTM", "GRU": "GRU",
             S2S: "Seq2Seq"}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(ms))[::-1]
    for yi, m in zip(y, ms):
        ax.plot([a.MAE[m], b.MAE[m]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter([a.MAE[m] for m in ms], y, s=30, color=C1, edgecolor="white", lw=1, zorder=3,
               label="16 training vehicles")
    ax.scatter([b.MAE[m] for m in ms], y, s=30, color=C2, edgecolor="white", lw=1, zorder=3,
               marker="s", label="4 training vehicles")
    ax.axvline(b.MAE[PERS], color=MUTED, lw=1, zorder=0)
    ax.text(b.MAE[PERS], len(ms) - 0.35, " Persistence", color=INK2, fontsize=7, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([short[m] for m in ms])
    ax.set_xlabel("MAE (% of SOH)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.0, 0.93))
    fig.savefig(os.path.join(FIG, "fig_data_efficiency.pdf")); plt.close(fig)
    print("fig_data_efficiency.pdf")


def fig_noise(S):
    if S is None:
        return
    bp = S[S.training.str.startswith("simulator")].set_index("condition")
    pers = S.pers.iloc[0]
    x = [1e-3, 1e-2, 3e-2, 1e-1]
    y = [bp.loc[f"depolarising p2={c}"].MAE for c in ("1e-3", "1e-2", "3e-2", "1e-1")]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhline(pers, color=MUTED, lw=1)
    ax.text(x[0], pers, "Persistence", color=INK2, fontsize=7, va="bottom")
    ax.axhline(bp.loc["no noise"].MAE, color=C1, lw=1, alpha=0.5)
    ax.text(x[0], bp.loc["no noise"].MAE, "No noise", color=INK2, fontsize=7, va="top")
    ax.plot(x, y, color=C1, lw=2, marker="o", ms=5, mec="white", mew=1,
            label="Depolarising noise", zorder=3)
    ax.scatter([3e-3], [bp.loc["preset noise, 4096 shots, readout correction"].MAE], color=C2,
               marker="s", s=36, edgecolor="white", lw=1, zorder=4,
               label="Preset noise, 4096 shots, readout correction")
    ax.set_ylim(bp.loc["no noise"].MAE - 0.05, None)
    ax.set_xscale("log"); ax.set_xlabel("Two-qubit gate error $p_2$ ($p_1=p_2/10$)")
    ax.set_ylabel("MAE (% of SOH)"); ax.legend(loc="upper left", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_noise.pdf")); plt.close(fig)
    print("fig_noise.pdf")


def fig_trajectory():
    W = load_windows(["baticm_causal_h1_q4l2_ntrall_s0"])
    W = W[W.fold == 1]
    vehs = sorted(W.veh.unique())
    fig, axs = plt.subplots(1, len(vehs), figsize=(7.0, 2.2), sharey=True)
    for ax, v in zip(np.atleast_1d(axs), vehs):
        g = W[W.veh == v]
        p = g[g.model == PROPOSED].reset_index(drop=True)
        s = g[g.model == S2S].reset_index(drop=True)
        t = np.arange(len(p)) + 7
        ax.scatter(t, p.raw * 100, s=10, color=MUTED, label="Measured monthly SOH", zorder=2)
        ax.plot(t, p.y * 100, color=INK, lw=1.2, label="Causal target", zorder=3)
        ax.plot(t, p.anchor * 100, color=MUTED, lw=1, label="Persistence", zorder=2)
        ax.plot(t, s.pred * 100, color=C3, lw=2, label="Seq2Seq", zorder=3)
        ax.plot(t, p.pred * 100, color=C1, lw=2, label="MQAttn-QLSTM-VQR", zorder=4)
        ax.set_title(f"Vehicle #{v}", fontsize=8, color=INK)
        ax.set_xlabel("Month index")
    np.atleast_1d(axs)[0].set_ylabel("SOH (%)")
    np.atleast_1d(axs)[-1].legend(loc="upper right", fontsize=6.5)
    fig.savefig(os.path.join(FIG, "fig_trajectory.pdf")); plt.close(fig)
    print("fig_trajectory.pdf")


def main():
    global TAB, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()
    TAB, FIG = os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    stats = table_main()
    table_protocol()
    table_ablation()
    table_grid()
    S = hardware_summary()
    table_hardware(S)
    table_cost()
    fig_horizon(stats)
    fig_data_efficiency(stats)
    fig_noise(S)
    fig_trajectory()


if __name__ == "__main__":
    main()
