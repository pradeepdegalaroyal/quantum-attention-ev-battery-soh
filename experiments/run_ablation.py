"""Ablation of MQAttn-QLSTM-VQR (Table 3 of the paper).

Each variant removes one component or replaces one group of circuits with
classical layers of the same size.  All variants use the same learning rate,
epochs and patience.  Circuit depth is not part of the ablation; see the
--nq/--lv options of experiments.run_comparison.

Example (from the repository root, one run per seed):
    python -m experiments.run_ablation --target causal --lrs 1e-2 --seed 0
"""
from __future__ import annotations

import sys

import qsoh.models as M
from experiments import run_comparison as P

FULL = dict(hidden=16, n_qubits=4)


def _classical(model, attention=False, residual=False, recurrence=False):
    """Replace groups of circuits by classical layers of the same size."""
    if recurrence and isinstance(model.rnn_cell, M.QLSTMCell):
        for k in ("vqc_f", "vqc_i", "vqc_g", "vqc_o"):
            setattr(model.rnn_cell, k, P.ClassicalTwin(4, 2))
    if attention and model.attn is not None:
        for k in ("vqc_q", "vqc_k", "vqc_v"):
            setattr(model.attn, k, P.ClassicalTwin(4, 2))
    if residual and model.vqr is not None:
        model.vqr.vqc = P.ClassicalTwin(4, 2)
    return model


def ablation_zoo(in_dim: int, nq: int = 4, lv: int = 2):
    mk = lambda **kw: M.MQAttn_QLSTM_VQR(in_dim=in_dim, **{**FULL, **kw})  # noqa: E731
    return {
        "Full model": (lambda: mk(), True),
        "Without quantum attention": (lambda: mk(use_attention=False), True),
        "Without VQR": (lambda: mk(use_vqr=False), True),
        "Without quantum attention and VQR": (lambda: mk(use_attention=False, use_vqr=False), True),
        "Classical QLSTM gates": (lambda: _classical(mk(), recurrence=True), True),
        "Classical attention circuits": (lambda: _classical(mk(), attention=True), True),
        "Classical VQR circuit": (lambda: _classical(mk(), residual=True), True),
        "All circuits classical": (lambda: _classical(mk(), attention=True, residual=True,
                                                      recurrence=True), False),
    }


if __name__ == "__main__":
    P.model_zoo = ablation_zoo
    if "--tag" not in sys.argv:
        sys.argv += ["--tag", "_ablation"]
    P.main()
