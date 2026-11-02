import numpy as np
import pytest
import torch
from protorep.model import ProtoRep, info_nce, augment, proto_losses
from protorep.train import train_protorep, train_ae, train_ts2vec, batched, TS2VecLite
from protorep.windows import sliding, windows_from
from protorep.metrics import point_adjust, detection_report

torch.set_num_threads(1)


def toy(n=300, C=4, L=16, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, L)
    X = np.stack([np.stack([np.sin(t + rng.uniform(0, 0.3)) * (c + 1) / C for c in range(C)]) for _ in range(n)])
    return (X + 0.05 * rng.normal(size=X.shape)).astype(np.float32)


def test_forward_shapes():
    m = ProtoRep(5, 12, 7, 8); o = m(torch.randn(3, 5, 12))
    assert o["h"].shape == (3, 5, 8) and o["dist"].shape == (3, 7) and o["recon"].shape == (3, 5, 12) and o["g"].shape == (3, 5)


def test_gates_off_are_ones():
    m = ProtoRep(3, 10, 4, 8, sensor_gates=False)
    assert torch.allclose(m(torch.randn(2, 3, 10))["g"], torch.ones(2, 3))


def test_attribution_sums_to_weighted_distance():
    m = ProtoRep(4, 10, 5, 8); x = torch.randn(6, 4, 10)
    o = m(x); a = m.attribution(x)
    assert torch.allclose(a.sum(1) / o["g"].sum(1), o["dist"].min(1).values, atol=1e-5)


def test_info_nce_lower_for_aligned_pairs():
    z = torch.randn(32, 8)
    assert info_nce(z, z).item() < info_nce(z, torch.randn(32, 8)).item()


def test_diversity_loss_zero_for_far_prototypes():
    m = ProtoRep(2, 8, 3, 4)
    m.prototypes.data = torch.tensor([0.0, 10.0, 20.0])[:, None, None].expand(3, 2, 4).clone()
    o = m(torch.randn(2, 2, 8))
    assert proto_losses(m, o)["div"].item() == 0.0


def test_training_pushes_prototypes_to_real_windows_and_detects_anomaly():
    X = toy()
    m = train_protorep(X, n_prototypes=4, epochs=3, seed=0)
    src = m.proto_src.numpy(); assert (src >= 0).all()
    H = batched(lambda x: m(x)["h"], X[src])
    assert np.allclose(H, m.prototypes.detach().numpy(), atol=1e-4)
    bad = X[:20].copy(); bad[:, 1] += 3.0
    assert np.median(batched(m.score, bad)) > np.quantile(batched(m.score, X), 0.95)


def test_weak_label_hook_runs():
    X = toy(120); w = -np.ones(120, int); w[:5] = 1; w[5:50] = 0
    m = train_protorep(X, n_prototypes=3, epochs=1, seed=0, weak=w)
    assert np.isfinite(batched(m.score, X)).all()


def test_baselines_train():
    X = toy(80)
    for kind in ["conv", "lstm"]:
        a = train_ae(X, kind, 1)
        assert a(torch.tensor(X[:2])).shape == (2, 4, 16)
    tv = train_ts2vec(X, 1)
    assert batched(tv.embed, X[:3]).shape == (3, 64)


def test_augment_preserves_shape():
    x = torch.randn(4, 3, 20)
    assert augment(x).shape == x.shape


def test_windows_and_metrics():
    w = windows_from([np.zeros((10, 2)), np.ones((3, 2))], 4)
    assert w.shape == (7, 2, 4)
    y = np.array([0, 1, 1, 0]); assert point_adjust(np.array([0, 1, 0, 0], bool), y).tolist() == [False, True, True, False]
    assert detection_report(np.array([0.1, 0.9, 0.8, 0.2]), y)["auc_roc"] == pytest.approx(1.0)
