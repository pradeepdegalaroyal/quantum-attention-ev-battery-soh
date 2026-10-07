"""Exact cost of one prediction (Table 6 of the paper).

1. A plain NumPy state-vector simulation of the 4-qubit circuit (16 complex
   amplitudes) reproduces PennyLane, so a classical processor can run the
   trained model exactly.
2. For every model: parameters, weight memory (32-bit floats), floating-point
   operations (FLOPs) per prediction, and circuit evaluations per prediction.
3. Circuit evaluations per window and update step for training with the
   parameter-shift rule.

Only exact counts are reported; the script does not estimate run times.

Example (from the repository root):
    python -m experiments.run_cost
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import qsoh.models as M                                    # noqa: E402

NQ, LV, L, F = 4, 2, 6, 11          # 10 charging features + last observed SOH


# ---------------------------------------------------------------------------
# NumPy state-vector implementation of the circuit in qsoh/models.py::_make_vqc
# ---------------------------------------------------------------------------
def _rx(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -1j * s], [-1j * s, c]])
def _ry(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]], complex)
def _rz(t): return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def _apply_1q(psi, U, q, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def _apply_cnot(psi, c, t, n):
    psi = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[c] = 1
    sub = psi[tuple(idx)]
    tt = t if t < c else t - 1
    psi[tuple(idx)] = np.flip(sub, axis=tt)
    return psi.reshape(-1)


def vqc_numpy(x, W, n=NQ):
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for q in range(n):
        psi = _apply_1q(psi, _ry(x[q]), q, n)
    for layer in W:
        for q in range(n):
            U = _rz(layer[q, 2]) @ _ry(layer[q, 1]) @ _rx(layer[q, 0])   # fused per qubit
            psi = _apply_1q(psi, U, q, n)
        for q in range(n):
            psi = _apply_cnot(psi, q, (q + 1) % n, n)
    p = np.abs(psi.reshape([2] * n)) ** 2
    return np.array([p.take(0, q).sum() - p.take(1, q).sum() for q in range(n)])


def vqc_flops(n=NQ, lv=LV):
    """Real FLOPs for one circuit with fused single-qubit gates.

    A 2x2 complex gate on a 2^n state = 2^(n-1) pairs x (4 cmul + 2 cadd)
    = 2^(n-1) x 28 real FLOPs.  CNOTs are permutations (0 FLOPs).
    Gate-matrix construction ~ 40 FLOPs incl. trig per qubit per layer.
    Readout: |amp|^2 (3 FLOPs each) + n signed sums.
    """
    per_gate = (2 ** (n - 1)) * 28
    gates = n + n * lv                    # encoding RY + fused RX.RY.RZ
    build = 40 * gates
    readout = 3 * 2 ** n + n * 2 ** n
    return gates * per_gate + build + readout


def linear_flops(i, o): return 2 * i * o


def lstm_flops(i, h, steps): return steps * (2 * 4 * h * (i + h) + 10 * h)


def gru_flops(i, h, steps): return steps * (2 * 3 * h * (i + h) + 10 * h)


def proposed_flops(h=16, in_dim=F, steps=L, quantum_rnn=True):
    c = vqc_flops()
    if quantum_rnn:
        rnn = steps * (linear_flops(in_dim + h, NQ) + 4 * c + 4 * linear_flops(NQ, h) + 10 * h)
    else:
        rnn = lstm_flops(in_dim, h, steps)
    qa = steps * (linear_flops(h, NQ) + 3 * c + linear_flops(NQ, h)) + 2 * steps * steps * NQ
    head = linear_flops(h, h // 2) + linear_flops(h // 2, 1)
    vqr = linear_flops(4, NQ) + c + linear_flops(NQ, 8) + linear_flops(8, 1)
    return rnn + qa + head + vqr


def circuits_per_prediction(quantum_rnn=True, steps=L):
    return (4 * steps if quantum_rnn else 0) + 3 * steps + 1


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)

    # 1) the NumPy state-vector simulation equals PennyLane
    vqc = M._make_vqc(NQ, LV)
    W = vqc.qlayer.weights.detach().numpy()
    errs = []
    for _ in range(50):
        x = rng.uniform(-1, 1, NQ)
        ref = vqc(torch.tensor(x, dtype=torch.float32)[None]).detach().numpy()[0]
        errs.append(np.max(np.abs(ref - vqc_numpy(x, W))))
    max_err = float(np.max(errs))
    print(f"NumPy state vector vs PennyLane: max |diff| over 50 inputs = {max_err:.2e}")

    # 2) exact counts per prediction
    c1 = vqc_flops()
    models = {
        "Seq2Seq": (M.Seq2SeqRegressor(F, hidden=32),
                    2 * lstm_flops(F, 32, L) + lstm_flops(F, 64, 1) + 2 * (64 * 32 + 32), 0),
        "LSTM": (M.LSTMRegressor(F, hidden=10), lstm_flops(F, 10, L) + 2 * (10 * 5 + 5), 0),
        "QLSTM": (M.QLSTM(F, hidden=16, n_qubits=NQ, n_layers=LV),
                  L * (linear_flops(F + 16, NQ) + 4 * c1 + 4 * linear_flops(NQ, 16) + 160)
                  + linear_flops(16, 1), 4 * L),
        "MQAttn-QLSTM-VQR": (M.MQAttn_QLSTM_VQR(F, hidden=16, n_qubits=NQ), proposed_flops(),
                             circuits_per_prediction(True)),
        "MQAttn-LSTM-VQR": (M.MQAttn_QLSTM_VQR(F, hidden=8, n_qubits=NQ, use_qlstm=False),
                            proposed_flops(h=8, quantum_rnn=False), circuits_per_prediction(False)),
    }
    out = {"numpy_vs_pennylane_max_abs_diff": max_err, "flops_per_circuit": c1, "models": {}}
    print(f"\nFLOPs per 4-qubit, 2-layer circuit (exact classical simulation): {c1}")
    print(f"\n{'model':18s} {'params':>7s} {'weights':>9s} {'FLOPs':>9s} {'circuits':>9s}")
    for name, (m, fl, ncirc) in models.items():
        npar = sum(p.numel() for p in m.parameters())
        out["models"][name] = dict(params=npar, weight_kB_fp32=npar * 4 / 1024,
                                   flops_per_prediction=fl, circuits_per_prediction=ncirc)
        print(f"{name:18s} {npar:7d} {npar*4/1024:7.1f}kB {fl:9,d} {ncirc:9d}")

    # 3) parameter-shift training: two evaluations per angle occurrence
    per_circuit = LV * NQ * 3
    occurrences = (4 * L + 3 * L + 1) * per_circuit
    out["parameter_shift"] = dict(variational_parameters=8 * per_circuit,
                                  circuit_evaluations_per_window_and_step=2 * occurrences)
    print(f"\nVariational parameters: {8 * per_circuit}")
    print(f"Parameter shift: {2 * occurrences} circuit evaluations per window and update step")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "cost.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
