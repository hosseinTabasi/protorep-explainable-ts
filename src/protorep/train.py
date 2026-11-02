"""Training loops for ProtoRep and baselines (all CPU-friendly)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import ProtoRep, info_nce, augment, proto_losses

DEFAULT_LAMBDAS = dict(con=0.5, clu=0.5, evi=0.1, div=0.5, sp=0.01, weak=1.0)


def set_seed(s: int) -> None:
    torch.manual_seed(s); np.random.seed(s)


def train_protorep(X: np.ndarray, n_prototypes: int = 16, epochs: int = 20, seed: int = 0,
                   sensor_gates: bool = True, lambdas: dict | None = None, weak: np.ndarray | None = None,
                   d: int = 32, bs: int = 128, lr: float = 3e-3) -> ProtoRep:
    """X (N, C, L) normal (or unlabelled) windows. weak: optional (N,) in {-1, 0, 1}."""
    set_seed(seed)
    lam = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    Xt = torch.tensor(X, dtype=torch.float32)
    W = torch.tensor(weak if weak is not None else -np.ones(len(X)), dtype=torch.long)
    m = ProtoRep(X.shape[1], X.shape[2], n_prototypes, d, sensor_gates)
    with torch.no_grad():  # initialise prototypes on random real windows
        m.prototypes.data = m.enc(Xt[torch.randperm(len(Xt))[:n_prototypes]]).clone()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            idx = perm[s:s + bs]
            if len(idx) < 8:
                continue
            x = Xt[idx]
            o = m(x)
            loss = F.mse_loss(o["recon"], x)
            if lam["con"] > 0:
                loss = loss + lam["con"] * info_nce(m(augment(x))["z"], m(augment(x))["z"])
            pl = proto_losses(m, o)
            loss = loss + lam["clu"] * pl["clu"] + lam["evi"] * pl["evi"] + lam["div"] * pl["div"] + lam["sp"] * pl["sp"]
            w = W[idx]
            if (w == 1).any():
                loss = loss + lam["weak"] * F.relu(2.0 - o["dist"][w == 1].min(1).values).mean()
                loss = loss + lam["weak"] * o["dist"][w == 0].min(1).values.mean() if (w == 0).any() else loss
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    m.push_prototypes(Xt[W != 1] if (W == 1).any() else Xt)
    return m


def batched(fn, X: np.ndarray, bs: int = 2048) -> np.ndarray:
    out = []
    for s in range(0, len(X), bs):
        out.append(fn(torch.tensor(X[s:s + bs], dtype=torch.float32)).detach().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- baselines
class ConvAE(nn.Module):
    def __init__(self, C: int, L: int, z: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(C, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
                                 nn.Flatten(), nn.Linear(32 * L, z))
        self.dec = nn.Sequential(nn.Linear(z, 32 * L), nn.GELU(), nn.Unflatten(1, (32, L)), nn.Conv1d(32, C, 5, padding=2))

    def forward(self, x):
        return self.dec(self.enc(x))


class LSTMAE(nn.Module):
    def __init__(self, C: int, L: int, z: int = 32):
        super().__init__()
        self.L = L
        self.e = nn.LSTM(C, z, batch_first=True); self.d = nn.LSTM(z, z, batch_first=True); self.o = nn.Linear(z, C)

    def enc(self, x):
        return self.e(x.transpose(1, 2))[1][0][-1]

    def forward(self, x):
        h = self.enc(x)[:, None].expand(-1, self.L, -1)
        return self.o(self.d(h)[0]).transpose(1, 2)


def train_ae(X: np.ndarray, kind: str = "conv", epochs: int = 20, seed: int = 0, bs: int = 128) -> nn.Module:
    set_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    m = (ConvAE if kind == "conv" else LSTMAE)(X.shape[1], X.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            x = Xt[perm[s:s + bs]]
            loss = F.mse_loss(m(x), x); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


class TS2VecLite(nn.Module):
    """Compact TS2Vec-style encoder: dilated conv stack, hierarchical (multi-scale) contrastive
    loss over two random overlapping crops, combining instance-wise and temporal negatives."""

    def __init__(self, C: int, d: int = 64, depth: int = 4):
        super().__init__()
        self.inp = nn.Linear(C, d)
        self.blocks = nn.ModuleList([nn.Conv1d(d, d, 3, padding=2 ** i, dilation=2 ** i) for i in range(depth)])

    def forward(self, x):  # (B, C, L) -> (B, d, L)
        h = self.inp(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            h = h + F.gelu(b(h))
        return h

    def embed(self, x):
        return self.forward(x).amax(-1)


def _hier_loss(z1, z2):
    loss, n = 0.0, 0
    while z1.shape[-1] > 1:
        B, d, T = z1.shape
        a, b = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        # instance contrast at each timestamp
        zi = torch.cat([a, b], 0).permute(2, 0, 1)        # (T, 2B, d)
        s = zi @ zi.transpose(1, 2); s = s - 1e4 * torch.eye(2 * B)[None]
        y = torch.cat([torch.arange(B, 2 * B), torch.arange(B)])
        loss = loss + F.cross_entropy(s.reshape(-1, 2 * B) / 0.2, y.repeat(T)); n += 1
        # temporal contrast inside each instance
        zt = torch.cat([a, b], 2).transpose(1, 2)         # (B, 2T, d)
        s = zt @ zt.transpose(1, 2); s = s - 1e4 * torch.eye(2 * T)[None]
        y = torch.cat([torch.arange(T, 2 * T), torch.arange(T)])
        loss = loss + F.cross_entropy(s.reshape(-1, 2 * T) / 0.2, y.repeat(B)); n += 1
        z1, z2 = F.max_pool1d(z1, 2), F.max_pool1d(z2, 2)
    return loss / max(1, n)


def train_ts2vec(X: np.ndarray, epochs: int = 20, seed: int = 0, bs: int = 64) -> TS2VecLite:
    set_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    m = TS2VecLite(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    L = X.shape[2]; crop = max(4, int(0.75 * L))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            x = Xt[perm[s:s + bs]]
            if len(x) < 4:
                continue
            o = L - crop
            a = int(torch.randint(0, o // 2 + 1, (1,))); b = a + int(torch.randint(0, o // 2 + 1, (1,)))
            ov = crop - (b - a)
            z1 = m(augment(x[..., a:a + crop]))[..., b - a:]
            z2 = m(augment(x[..., b:b + crop]))[..., :ov]
            loss = _hier_loss(z1, z2); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()
