"""ProtoRep model: per-sensor temporal encoder -> sparse sensor gates -> prototype bottleneck.

For a window x in R^{C x L}:
  h_c = f(x_c) + e_c                       per-sensor embedding (shared conv, sensor id embedding)
  g_c = sigmoid(w^T h_c + b)               sensor gate (sparse via L1)
  d_k(x) = sum_c g_c ||h_c - P_{k,c}||^2 / sum_c g_c     gated distance to prototype k
  score(x) = min_k d_k(x)                  anomaly score (distance to nearest normal prototype)
  a_c(x) = g_c ||h_c - P_{k*,c}||^2        per-sensor attribution for the nearest prototype k*
Training loss (all unsupervised; optional weak labels):
  L = recon + lam_con * InfoNCE + lam_clu * mean_x min_k d_k + lam_evi * mean_k min_x d_k
      + lam_div * mean_{i<j} relu(margin - ||P_i - P_j||^2 / C) + lam_sp * mean g
      + lam_weak * mean_{x: weak label = 1} relu(margin - min_k d_k)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorEncoder(nn.Module):
    def __init__(self, n_sensors: int, d: int = 32, hidden: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, d, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(d, d, 3, padding=4, dilation=4))
        self.emb = nn.Parameter(torch.zeros(n_sensors, d))
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, C, L) -> (B, C, d)
        B, C, L = x.shape
        z = self.conv(x.reshape(B * C, 1, L))
        z = 0.5 * (z.mean(-1) + z.amax(-1))
        return z.reshape(B, C, self.d) + self.emb


class ProtoRep(nn.Module):
    def __init__(self, n_sensors: int, length: int, n_prototypes: int = 16, d: int = 32,
                 sensor_gates: bool = True):
        super().__init__()
        self.C, self.L, self.K, self.d = n_sensors, length, n_prototypes, d
        self.enc = SensorEncoder(n_sensors, d)
        self.gate = nn.Linear(d, 1)
        self.use_gates = sensor_gates
        self.prototypes = nn.Parameter(0.1 * torch.randn(n_prototypes, n_sensors, d))
        self.dec = nn.Linear(n_sensors * d, n_sensors * length)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.register_buffer("proto_src", torch.full((n_prototypes,), -1, dtype=torch.long))

    def gates(self, h: torch.Tensor) -> torch.Tensor:
        if not self.use_gates:
            return torch.ones(h.shape[:2], device=h.device)
        return torch.sigmoid(self.gate(h)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.enc(x)                                   # (B, C, d)
        g = self.gates(h)                                 # (B, C)
        diff = ((h[:, None] - self.prototypes[None]) ** 2).sum(-1)  # (B, K, C)
        dist = (diff * g[:, None]).sum(-1) / (g.sum(-1, keepdim=True) + 1e-6)  # (B, K)
        recon = self.dec((h * g[..., None]).reshape(len(x), -1)).reshape(x.shape)
        z = self.proj((h * g[..., None]).sum(1) / (g.sum(1, keepdim=True) + 1e-6))
        return dict(h=h, g=g, diff=diff, dist=dist, recon=recon, z=z)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["dist"].min(1).values

    @torch.no_grad()
    def attribution(self, x: torch.Tensor) -> torch.Tensor:
        o = self.forward(x)
        k = o["dist"].argmin(1)
        return o["g"] * o["diff"][torch.arange(len(x)), k]   # (B, C)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Explainable representation: gated per-sensor embedding + prototype similarities."""
        o = self.forward(x)
        return torch.cat([(o["h"] * o["g"][..., None]).reshape(len(x), -1), -o["dist"]], 1)

    @torch.no_grad()
    def push_prototypes(self, X: torch.Tensor, bs: int = 1024) -> None:
        """ProtoPNet-style projection: each prototype becomes the embedding of its nearest
        real training window, so every prototype is a real sensor snippet."""
        best = torch.full((self.K,), float("inf")); src = torch.zeros(self.K, dtype=torch.long)
        H = []
        for s in range(0, len(X), bs):
            o = self.forward(X[s:s + bs]); H.append(o["h"])
            v, i = o["dist"].min(0)
            upd = v < best; best[upd] = v[upd]; src[upd] = i[upd] + s
        H = torch.cat(H)
        self.prototypes.data = H[src].clone(); self.proto_src = src


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = z1 @ z2.T / tau
    y = torch.arange(len(z1))
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Jitter + per-sensor scaling + random time shift (circular is avoided: edge padding)."""
    x = x * (1 + 0.1 * torch.randn(x.shape[0], x.shape[1], 1)) + 0.05 * torch.randn_like(x)
    s = int(torch.randint(-2, 3, (1,)))
    if s > 0:
        x = torch.cat([x[..., :1].expand(-1, -1, s), x[..., :-s]], -1)
    elif s < 0:
        x = torch.cat([x[..., -s:], x[..., -1:].expand(-1, -1, -s)], -1)
    return x


def proto_losses(model: ProtoRep, o: dict, margin: float = 1.0) -> dict:
    P = model.prototypes
    pd_ = ((P[:, None] - P[None]) ** 2).sum((-1, -2)) / model.C
    iu = torch.triu_indices(model.K, model.K, 1)
    return dict(clu=o["dist"].min(1).values.mean(), evi=o["dist"].min(0).values.mean(),
                div=F.relu(margin - pd_[iu[0], iu[1]]).mean(), sp=o["g"].mean())
