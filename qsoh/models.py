"""Models used in the study.

Classical baselines
    LSTMRegressor     single-layer LSTM with a two-layer head
    GRURegressor      single-layer GRU with a two-layer head
    MLPRegressor      MLP on the flattened input window
    Seq2SeqRegressor  encoder-decoder LSTM (Deng et al., Applied Energy, 2023)

Quantum models
    QLSTM             quantum long short-term memory with a shared embedding
                      (Wang and Kebede, ISA Transactions, 2026)
    MQAttn_QLSTM_VQR  the proposed model: multi-qubit quantum attention (MQAttn)
                      over a QLSTM, with a variational quantum residual (VQR) head.
                      With ``use_qlstm=False`` a classical LSTM cell replaces the
                      QLSTM cell (the MQAttn-LSTM-VQR variant).

Every variational quantum circuit (VQC) uses the same hardware-efficient circuit:
RY angle encoding, ``n_layers`` layers of RX-RY-RZ rotations followed by a ring
of CNOT gates, and Pauli-Z readout.  ``_make_vqc`` builds it with PennyLane;
``qsoh.simulator.use_fast_backend`` replaces it with an exact, faster simulator.
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 12):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 14, window: int = 6):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim * window, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Seq2SeqRegressor(nn.Module):
    """Encoder-decoder LSTM of Deng et al. (2023)."""
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.dec = nn.LSTM(in_dim, 2 * hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        _, (h, c) = self.enc(x)
        h0 = h.transpose(0, 1).contiguous().view(1, x.size(0), -1)
        c0 = c.transpose(0, 1).contiguous().view(1, x.size(0), -1)
        y, _ = self.dec(x[:, -1:], (h0, c0))
        return self.head(y[:, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# Variational quantum circuit
# ---------------------------------------------------------------------------
def _make_vqc(n_qubits: int, n_layers: int, name: str = "vqc"):
    """PennyLane implementation of the circuit; returns a torch module R^n -> [-1, 1]^n."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for q in range(n_qubits):
            qml.RY(inputs[..., q], wires=q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RX(weights[layer, q, 0], wires=q)
                qml.RY(weights[layer, q, 1], wires=q)
                qml.RZ(weights[layer, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    layer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})

    class _VQC(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = layer

        def forward(self, x):
            out = self.qlayer(x.to("cpu"))
            if isinstance(out, (list, tuple)):
                out = torch.stack(out, dim=-1)
            return out.to(x.device)

    return _VQC()


# ---------------------------------------------------------------------------
# Quantum recurrent cell
# ---------------------------------------------------------------------------
class QLSTMCell(nn.Module):
    """QLSTM cell with a shared embedding (Wang and Kebede, 2026).

        z_t = tanh(W_e [x_t; h_{t-1}] + b_e)                (n_qubits values)
        f_t, i_t, o_t = sigmoid(W_k VQC_k(z_t) + b_k),   g_t = tanh(W_g VQC_g(z_t) + b_g)
        c_t = f_t * c_{t-1} + i_t * g_t,                  h_t = tanh(c_t) * tanh(o_t)
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_qubits: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_emb = nn.Linear(in_dim + hidden_dim, n_qubits)
        self.vqc_f = _make_vqc(n_qubits, n_layers, "f")
        self.vqc_i = _make_vqc(n_qubits, n_layers, "i")
        self.vqc_g = _make_vqc(n_qubits, n_layers, "g")
        self.vqc_o = _make_vqc(n_qubits, n_layers, "o")
        self.proj_f = nn.Linear(n_qubits, hidden_dim)
        self.proj_i = nn.Linear(n_qubits, hidden_dim)
        self.proj_g = nn.Linear(n_qubits, hidden_dim)
        self.proj_o = nn.Linear(n_qubits, hidden_dim)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        z = torch.tanh(self.shared_emb(torch.cat([x_t, h_prev], dim=-1)))
        f = torch.sigmoid(self.proj_f(self.vqc_f(z)))
        i = torch.sigmoid(self.proj_i(self.vqc_i(z)))
        g = torch.tanh(self.proj_g(self.vqc_g(z)))
        o = torch.sigmoid(self.proj_o(self.vqc_o(z)))
        c = f * c_prev + i * g
        h = torch.tanh(c) * torch.tanh(o)
        return h, c


class QLSTM(nn.Module):
    """QLSTM baseline with a linear output layer."""
    def __init__(self, in_dim: int, hidden: int = 16, n_qubits: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.cell = QLSTMCell(in_dim, hidden, n_qubits, n_layers)
        self.head = nn.Linear(hidden, 1)
        self.hidden_dim = hidden

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        c = x.new_zeros(B, self.hidden_dim)
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Proposed model
# ---------------------------------------------------------------------------
class QuantumAttention(nn.Module):
    """Quantum attention (QA) over the hidden states.

    Three VQCs compute the query, key and value vectors from the projected
    hidden states.  The similarity Q K^T / sqrt(n_qubits) and the softmax are
    classical; the layer does not estimate quantum state overlaps.
    """
    def __init__(self, hidden: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(hidden, n_qubits)
        self.vqc_q = _make_vqc(n_qubits, n_layers, "q")
        self.vqc_k = _make_vqc(n_qubits, n_layers, "k")
        self.vqc_v = _make_vqc(n_qubits, n_layers, "v")
        self.scale = 1.0 / math.sqrt(n_qubits)
        self.merge = nn.Linear(n_qubits, hidden)

    def forward(self, H):                                    # B x T x hidden
        B, T, _ = H.shape
        z = torch.tanh(self.proj(H)).reshape(B * T, -1)
        Q = self.vqc_q(z).reshape(B, T, -1)
        K = self.vqc_k(z).reshape(B, T, -1)
        V = self.vqc_v(z).reshape(B, T, -1)
        A = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return self.merge(A @ V)


class VariationalQuantumResidual(nn.Module):
    """Variational quantum residual (VQR): a VQC and a small MLP that correct
    the primary prediction from the last month's charging statistics."""
    def __init__(self, in_dim: int = 4, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.pre = nn.Linear(in_dim, n_qubits)
        self.vqc = _make_vqc(n_qubits, n_layers, "res")
        self.post = nn.Sequential(nn.Linear(n_qubits, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, r):
        return self.post(self.vqc(torch.tanh(self.pre(r)))).squeeze(-1)


class MQAttn_QLSTM_VQR(nn.Module):
    """MQAttn-QLSTM-VQR: multi-qubit quantum attention, quantum long short-term
    memory and variational quantum residual.

    Forward pass for a window x (B x T x F):
      1. a QLSTM cell (or a classical LSTM cell if ``use_qlstm=False``) reads x;
      2. quantum attention reweights the hidden states:  H <- H + alpha_qa * QA(H);
      3. an MLP head maps the last hidden state to the primary output y_p;
      4. the VQR adds alpha_vqr * r_hat, computed from the last month's first
         three features (charging-current mean and standard deviation, summed
         pack voltage) and y_p.
    The training code adds the persistence anchor s_{t-1} + mu to the output.
    """
    def __init__(self, in_dim: int, hidden: int = 16, n_qubits: int = 4,
                 qlstm_layers: int = 2, attn_layers: int = 2, vqr_layers: int = 2,
                 use_qlstm: bool = True, use_attention: bool = True,
                 use_vqr: bool = True, residual_feat_idx: tuple = (0, 1, 2)):
        super().__init__()
        self.hidden_dim = hidden
        self.residual_feat_idx = residual_feat_idx
        self.rnn_cell = (QLSTMCell(in_dim, hidden, n_qubits, qlstm_layers) if use_qlstm
                         else nn.LSTMCell(in_dim, hidden))
        if use_attention:
            self.attn = QuantumAttention(hidden, n_qubits, attn_layers)
            self.attn_alpha = nn.Parameter(torch.tensor(0.3))
        else:
            self.attn = None
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))
        if use_vqr:
            self.vqr = VariationalQuantumResidual(len(residual_feat_idx) + 1,
                                                  n_qubits, vqr_layers)
            self.vqr_alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.vqr = None

    def forward(self, x):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        c = x.new_zeros(B, self.hidden_dim)
        hs = []
        for t in range(T):
            h, c = self.rnn_cell(x[:, t], (h, c))
            hs.append(h)
        H = torch.stack(hs, dim=1)
        if self.attn is not None:
            H = H + self.attn_alpha * self.attn(H)
        primary = self.head(H[:, -1]).squeeze(-1)
        if self.vqr is None:
            return primary
        last = x[:, -1, list(self.residual_feat_idx)]
        return primary + self.vqr_alpha * self.vqr(torch.cat([last, primary.unsqueeze(-1)], dim=-1))
