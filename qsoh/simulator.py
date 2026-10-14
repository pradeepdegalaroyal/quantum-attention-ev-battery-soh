"""Exact batched simulator for the circuit of ``qsoh.models._make_vqc``.

``FastVQC`` reproduces the outputs and gradients of PennyLane ``default.qubit``
for the same circuit (RY angle encoding, n_layers x [RX RY RZ on every qubit +
CNOT ring], Pauli-Z readout) and evaluates a whole batch at once.

Options
    reupload=True   repeat the RY encoding before every variational layer
    set_noise(...)  density-matrix evaluation with depolarising noise
                    (p1 after every single-qubit gate, p2 on both qubits after
                    every CNOT), symmetric readout error, finite shots, and
                    optional readout correction z / (1 - 2e)
    set_grad_mode("shift")  gradients by the parameter-shift rule, computed only
                    from circuit evaluations, as on quantum hardware

Wire convention follows PennyLane: wire 0 is the most significant bit.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

_NOISE = dict(p1=0.0, p2=0.0, readout=0.0, shots=0, mitigate=False)
_GRAD = dict(mode="backprop")          # "backprop" (simulator only) or "shift" (hardware-compatible)


def set_noise(p1=0.0, p2=0.0, readout=0.0, shots=0, mitigate=False):
    """mitigate=True applies standard readout-error inversion z / (1 - 2 e)."""
    _NOISE.update(p1=p1, p2=p2, readout=readout, shots=shots, mitigate=mitigate)


def set_grad_mode(mode: str):
    assert mode in ("backprop", "shift")
    _GRAD["mode"] = mode


class _ParamShift(torch.autograd.Function):
    """Gradients of <Z_q> w.r.t. weights and input angles by the parameter-shift
    rule, dE/dθ = [E(θ+π/2) - E(θ-π/2)] / 2, i.e. only circuit evaluations --
    exactly what a QPU can provide.  Each shifted evaluation goes through the
    module's (possibly noisy / finite-shot) forward."""

    @staticmethod
    def forward(ctx, x, weights, module):
        ctx.module = module
        ctx.save_for_backward(x, weights)
        with torch.no_grad():
            return module._expect(x, weights)

    @staticmethod
    def backward(ctx, g):                       # g: (B, n)
        x, W = ctx.saved_tensors
        m, s = ctx.module, math.pi / 2
        gW = torch.zeros_like(W)
        gx = torch.zeros_like(x)
        with torch.no_grad():
            for idx in np.ndindex(*W.shape):
                Wp, Wm = W.clone(), W.clone()
                Wp[idx] += s; Wm[idx] -= s
                dE = (m._expect(x, Wp) - m._expect(x, Wm)) / 2      # (B, n)
                gW[idx] = (g * dE).sum()
            for q in range(x.shape[1]):
                xp, xm = x.clone(), x.clone()
                xp[:, q] += s; xm[:, q] -= s
                dE = (m._expect(xp, W) - m._expect(xm, W)) / 2
                gx[:, q] = (g * dE).sum(1)
        return gx, gW, None


def _bits(n):
    idx = torch.arange(2 ** n)
    return torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], 1)  # (2^n, n)


def _ring_perm(n):
    """Basis permutation implementing CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0) in order."""
    b = _bits(n).clone()
    for q in range(n):
        c, t = q, (q + 1) % n
        b[:, t] = b[:, t] ^ b[:, c]
    w = torch.tensor([2 ** (n - 1 - q) for q in range(n)])
    out = (b * w).sum(1)                      # basis i -> out[i]
    perm = torch.empty_like(out)
    perm[out] = torch.arange(2 ** n)          # new[j] = old[perm[j]]
    return perm


def _cplx(re, im):
    return torch.complex(re, im)


def _rx(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([_cplx(c, z), _cplx(z, -s)], -1),
                        torch.stack([_cplx(z, -s), _cplx(c, z)], -1)], -2)


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([_cplx(c, z), _cplx(-s, z)], -1),
                        torch.stack([_cplx(s, z), _cplx(c, z)], -1)], -2)


def _rz(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([_cplx(c, -s), _cplx(z, z)], -1),
                        torch.stack([_cplx(z, z), _cplx(c, s)], -1)], -2)


def _apply_1q(psi, U, q, n):
    """psi (B, 2^n) complex; U (2,2) or (B,2,2)."""
    B = psi.shape[0]
    p = psi.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
    if U.dim() == 2:
        p = torch.einsum("ij,akjb->akib", U, p)
    else:
        p = torch.einsum("aij,akjb->akib", U, p)
    return p.reshape(B, 2 ** n)


def _full_1q(U, q, n):
    """Embed a single-qubit unitary (...,2,2) into (...,2^n,2^n)."""
    eye_l = torch.eye(2 ** q, dtype=U.dtype, device=U.device)
    eye_r = torch.eye(2 ** (n - q - 1), dtype=U.dtype, device=U.device)
    if U.dim() == 2:
        return torch.kron(torch.kron(eye_l, U), eye_r)
    return torch.stack([torch.kron(torch.kron(eye_l, u), eye_r) for u in U])


_PAULI = None


def _depol_1q(rho, q, n, p):
    if p <= 0:
        return rho
    global _PAULI
    dev = rho.device
    X = torch.tensor([[0, 1], [1, 0]], dtype=rho.dtype, device=dev)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=rho.dtype, device=dev)
    Zm = torch.tensor([[1, 0], [0, -1]], dtype=rho.dtype, device=dev)
    acc = 0
    for P in (X, Y, Zm):
        F = _full_1q(P, q, n)
        acc = acc + F @ rho @ F.conj().T
    return (1 - p) * rho + (p / 3) * acc


class FastVQC(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, reupload: bool = False):
        super().__init__()
        self.n, self.L, self.reupload = n_qubits, n_layers, reupload
        # PennyLane TorchLayer default init: U(0, 2*pi)
        self.weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * 2 * math.pi)
        self.register_buffer("perm", _ring_perm(n_qubits), persistent=False)
        self.register_buffer("zsign", (1 - 2 * _bits(n_qubits)).float(), persistent=False)

    # ------------------------------------------------------------------ exact
    def _gate_tensor(self, W=None):
        """All fused single-qubit gates RZ.RY.RX at once: (L, n, 2, 2) complex."""
        W = self.weights if W is None else W
        h = W / 2
        c, s = torch.cos(h), torch.sin(h)                 # (L, n, 3)
        ca, cb, cc = c.unbind(-1)
        sa, sb, sc = s.unbind(-1)
        z = torch.zeros_like(ca)
        rx = torch.stack([torch.stack([torch.complex(ca, z), torch.complex(z, -sa)], -1),
                          torch.stack([torch.complex(z, -sa), torch.complex(ca, z)], -1)], -2)
        ry = torch.stack([torch.stack([torch.complex(cb, z), torch.complex(-sb, z)], -1),
                          torch.stack([torch.complex(sb, z), torch.complex(cb, z)], -1)], -2)
        rz = torch.stack([torch.stack([torch.complex(cc, -sc), torch.complex(z, z)], -1),
                          torch.stack([torch.complex(z, z), torch.complex(cc, sc)], -1)], -2)
        return rz @ ry @ rx

    def _layer_unitaries(self, W=None):
        G = self._gate_tensor(W)
        return [[G[l, q] for q in range(self.n)] for l in range(self.L)]

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, self.n)
        if _GRAD["mode"] == "shift" and torch.is_grad_enabled():
            z = _ParamShift.apply(x, self.weights, self)
        else:
            z = self._expect(x, self.weights)
        return z.reshape(*shp[:-1], self.n)

    def _expect(self, x, W):
        if any(_NOISE[k] for k in ("p1", "p2", "readout")):
            z = self._forward_noisy(x, W)
        else:
            z = self._forward_exact(x, W)
        if _NOISE["shots"]:
            pr = ((1 + z.detach()) / 2).clamp(0, 1)
            zs = 2 * torch.distributions.Binomial(_NOISE["shots"], pr).sample() / _NOISE["shots"] - 1
            z = z + (zs - z).detach()          # sampled value, straight-through gradient
        if _NOISE["mitigate"] and _NOISE["readout"]:
            z = z / (1 - 2 * _NOISE["readout"])
        return z

    def _encode(self, psi, x):
        for q in range(self.n):
            psi = _apply_1q(psi, _ry(x[:, q]), q, self.n)
        return psi

    def _layer_matrices(self, W=None):
        """Each variational layer as one 2^n x 2^n unitary: P_ring . (U_0 x ... x U_{n-1})."""
        G = self._gate_tensor(W)                 # (L, n, 2, 2)
        K = G[:, 0]
        for q in range(1, self.n):               # batched Kronecker over layers
            Lr, a, b = K.shape
            K = torch.einsum("lij,lkm->likjm", K, G[:, q]).reshape(Lr, a * 2, b * 2)
        K = K[:, self.perm]                      # row permutation = CNOT ring after rotations
        return list(K.unbind(0))

    def _forward_exact(self, x, W=None):
        n = self.n
        c, s = torch.cos(x / 2), torch.sin(x / 2)      # RY(x)|0> = [cos, sin]
        psi = torch.stack([c[:, 0], s[:, 0]], -1)
        for q in range(1, n):                          # product state, wire 0 = MSB
            psi = (psi[:, :, None] * torch.stack([c[:, q], s[:, q]], -1)[:, None, :]).reshape(x.shape[0], -1)
        psi = psi.to(torch.complex64)
        mats = self._layer_matrices(W)
        if self.reupload:
            E = None
            for l, Ml in enumerate(mats):
                if l > 0:
                    if E is None:
                        E = self._encoding_unitary(x)
                    psi = torch.bmm(E, psi[:, :, None])[:, :, 0]
                psi = psi @ Ml.T
        else:
            Mtot = mats[0]
            for Ml in mats[1:]:
                Mtot = Ml @ Mtot
            psi = psi @ Mtot.T
        probs = psi.real ** 2 + psi.imag ** 2
        return probs @ self.zsign

    def _encoding_unitary(self, x):
        K = _ry(x[:, 0])
        for q in range(1, self.n):
            U = _ry(x[:, q])
            B, a, b = K.shape
            K = torch.einsum("bij,bkl->bikjl", K, U).reshape(B, a * 2, b * 2)
        return K

    # ------------------------------------------------------------------ noisy
    def _forward_noisy(self, x, W=None):
        n, p1, p2, ro = self.n, _NOISE["p1"], _NOISE["p2"], _NOISE["readout"]
        B = x.shape[0]
        D = 2 ** n
        rho = torch.zeros(B, D, D, dtype=torch.complex64, device=x.device)
        rho[:, 0, 0] = 1
        Us = self._layer_unitaries(W)

        def enc(rho):
            for q in range(n):
                U = _full_1q(_ry(x[:, q]), q, n)
                rho = U @ rho @ U.conj().transpose(-1, -2)
                rho = _depol_1q(rho, q, n, p1)
            return rho

        rho = enc(rho)
        for l in range(self.L):
            if self.reupload and l > 0:
                rho = enc(rho)
            for q in range(n):
                U = _full_1q(Us[l][q], q, n)
                rho = U @ rho @ U.conj().T
                for _ in range(3):                      # RX, RY, RZ each noisy
                    rho = _depol_1q(rho, q, n, p1)
            for q in range(n):                          # ring of CNOTs, noisy
                c, t = q, (q + 1) % n
                Pq = torch.eye(D, dtype=rho.dtype, device=x.device)[_single_cnot_perm(n, c, t).to(x.device)]
                rho = Pq @ rho @ Pq.T
                rho = _depol_1q(rho, c, n, p2)
                rho = _depol_1q(rho, t, n, p2)
        probs = torch.diagonal(rho, dim1=-2, dim2=-1).real
        z = probs @ self.zsign
        return (1 - 2 * ro) * z


def _single_cnot_perm(n, c, t):
    b = _bits(n).clone()
    b[:, t] = b[:, t] ^ b[:, c]
    w = torch.tensor([2 ** (n - 1 - q) for q in range(n)])
    out = (b * w).sum(1)
    perm = torch.empty_like(out)
    perm[out] = torch.arange(2 ** n)
    return perm


def make_fast_vqc(n_qubits, n_layers, name="", reupload=False):
    return FastVQC(n_qubits, n_layers, reupload)


def use_fast_backend(models_module, reupload=False):
    """Make every model built afterwards use FastVQC instead of PennyLane."""
    models_module._make_vqc = lambda n_qubits, n_layers, name="": FastVQC(n_qubits, n_layers, reupload)
