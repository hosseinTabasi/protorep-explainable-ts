"""ProtoRep experiments E1-E5 (+E4b weak supervision). Writes experiments/results and figures."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, yaml, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari, f1_score, accuracy_score
from protorep.data import load_skab, load_tep, load_tep_train, SKAB_SENSORS
from protorep.windows import sliding, windows_from
from protorep.train import train_protorep, train_ae, train_ts2vec, batched
from protorep.metrics import detection_report

torch.set_num_threads(2)
CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
RES, FIG = os.path.join(ROOT, "experiments", "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
K, EP, SEEDS, SP = CFG["n_prototypes"], CFG["epochs"], CFG["seeds"], CFG["sensor_sparsity"]
LAM = dict(sp=SP)


def dataset(name):
    if name == "SKAB":
        segs = load_skab(); L = CFG["window"]
        X = windows_from([s.ref for s in segs], L, 2)
        XT = np.concatenate([sliding(np.concatenate([s.ref[-(L - 1):], s.test]), L) for s in segs])
        y = np.concatenate([s.labels for s in segs]); ep = EP
    else:
        ref, segs = load_tep(); L = CFG["window_tep"]
        X = sliding(ref, L, 1)
        XT = np.concatenate([sliding(np.concatenate([ref[-(L - 1):], s.test]), L) for s in segs])
        y = np.concatenate([s.labels for s in segs]); ep = EP * 3  # tiny training set
    return X, XT, y, ep, segs


def knn_score(E, ET):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=1).fit(E).kneighbors(ET)[0][:, 0]


def e1_detection():
    rows, models = [], {}
    for ds in ["SKAB", "TEP"]:
        X, XT, y, ep, _ = dataset(ds)
        Xf, XTf = X.reshape(len(X), -1), XT.reshape(len(XT), -1)
        p = PCA(0.9).fit(Xf); rec = p.inverse_transform(p.transform(XTf))
        rows.append(dict(dataset=ds, method="PCA reconstruction", seed=0, **detection_report(((XTf - rec) ** 2).mean(1), y)))
        for sd in SEEDS:
            t = time.perf_counter(); m = train_protorep(X, K, ep, sd, lambdas=LAM); tt = time.perf_counter() - t
            models[(ds, sd)] = m
            rows.append(dict(dataset=ds, method="ProtoRep (ours)", seed=sd, train_s=tt, **detection_report(batched(m.score, XT), y)))
            for kind, nm in [("conv", "Conv autoencoder"), ("lstm", "LSTM autoencoder")]:
                t = time.perf_counter(); a = train_ae(X, kind, ep, sd); tt = time.perf_counter() - t
                rows.append(dict(dataset=ds, method=nm, seed=sd, train_s=tt, **detection_report(batched(lambda x: ((a(x) - x) ** 2).mean((1, 2)), XT), y)))
            t = time.perf_counter(); tv = train_ts2vec(X, ep, sd); tt = time.perf_counter() - t
            rows.append(dict(dataset=ds, method="TS2Vec-lite + 1-NN", seed=sd, train_s=tt, **detection_report(knn_score(batched(tv.embed, X), batched(tv.embed, XT)), y)))
            f = IsolationForest(n_estimators=200, random_state=sd).fit(Xf)
            rows.append(dict(dataset=ds, method="Isolation Forest", seed=sd, **detection_report(-f.score_samples(XTf), y)))
            print("E1", ds, sd, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e1_detection_raw.csv", index=False)
    cols = ["auc_roc", "auc_pr", "f1_best", "f1_best_pa"]
    g = df.groupby(["dataset", "method"])[cols].agg(["mean", "std"]).round(4)
    g.columns = ["_".join(c) for c in g.columns]; g.reset_index().to_csv(f"{RES}/e1_detection_summary.csv", index=False)
    torch.save({f"{k[0]}_{k[1]}": v.state_dict() for k, v in models.items()}, "/tmp/protorep_models.pt")
    plot_prototypes(models[("SKAB", 0)], dataset("SKAB")[0])
    return models


def plot_prototypes(m, X):
    src = m.proto_src.numpy()
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    for a, k in zip(axs.ravel(), range(8)):
        w = X[src[k]]; a.plot(w.T, lw=0.8); a.set_title(f"prototype {k} (window #{src[k]})", fontsize=8); a.set_xticks([])
    axs[0, 0].legend([s[:10] for s in SKAB_SENSORS], fontsize=5)
    fig.suptitle("ProtoRep prototypes after projection = real SKAB snippets (8 sensors)"); fig.tight_layout()
    fig.savefig(f"{FIG}/prototypes_skab.png", dpi=130); plt.close()


def grad_attr(m, x):
    x = x.clone().requires_grad_(True)
    s = m(x)["dist"].min(1).values.sum(); s.backward()
    return (x.grad * x).abs().sum(-1).detach()


def e2_faithfulness(models):
    rows, curves = [], {}
    for ds in ["SKAB", "TEP"]:
        X, XT, y, ep, _ = dataset(ds)
        for sd in SEEDS:
            m = models[(ds, sd)]
            ref_s = batched(m.score, X); sc = batched(m.score, XT)
            sel = np.where((y == 1) & (sc > np.quantile(ref_s, 0.99)))[0]
            rng = np.random.default_rng(sd); sel = rng.choice(sel, min(400, len(sel)), replace=False)
            x = torch.tensor(XT[sel]); C = x.shape[1]
            base = m.score(x)
            attrs = {"ProtoRep attribution": m.attribution(x), "Gradient x input": grad_attr(m, x),
                     "Random": torch.tensor(rng.random((len(sel), C)), dtype=torch.float32)}
            for nm, A in attrs.items():
                order = torch.argsort(A, 1, descending=True)
                dele, ins = [], []
                for k in range(C + 1):
                    mask = torch.zeros_like(A); 
                    if k: mask.scatter_(1, order[:, :k], 1.0)
                    dele.append((m.score(x * (1 - mask[..., None])) / base).mean().item())
                    ins.append((m.score(x * mask[..., None]) / base).mean().item())
                curves[(ds, nm, sd)] = (dele, ins)
                rows.append(dict(dataset=ds, attribution=nm, seed=sd, n_windows=len(sel),
                                 deletion_auc=float(np.mean(dele)), insertion_auc=float(np.mean(ins)),
                                 score_after_top1_deleted=dele[1], score_after_top3_deleted=dele[min(3, C)]))
        print("E2", ds, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e2_faithfulness_raw.csv", index=False)
    g = df.groupby(["dataset", "attribution"]).agg(["mean", "std"]).drop(columns=["seed", "n_windows"], level=0).round(4)
    g.columns = ["_".join(c) for c in g.columns]; g.reset_index().to_csv(f"{RES}/e2_faithfulness_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for a, ds in zip(axs, ["SKAB", "TEP"]):
        for nm in ["ProtoRep attribution", "Gradient x input", "Random"]:
            d = np.mean([curves[(ds, nm, s)][0] for s in SEEDS], 0); a.plot(d, label=nm)
        a.set_title(f"{ds}: deletion curve (lower = more faithful)"); a.set_xlabel("# sensors removed"); a.set_ylabel("score / original")
    axs[0].legend(); fig.tight_layout(); fig.savefig(f"{FIG}/e2_deletion.png", dpi=140); plt.close()


def tep_pool(L, stride=2):
    xs, ids = load_tep_train()
    X = np.concatenate([sliding(x, L, stride) for x in xs])
    yl = np.concatenate([np.full(len(sliding(x, L, stride)), i) for x, i in zip(xs, ids)])
    ref, segs = load_tep()
    from protorep.data import DATA_ROOT
    x0 = np.loadtxt(os.path.join(DATA_ROOT, "TEP_Braatz", "d00_te.dat")); x0 = (x0 - x0[:0].sum(0)) if False else x0
    mu, sd = np.loadtxt(os.path.join(DATA_ROOT, "TEP_Braatz", "d00.dat")).T.mean(0), np.loadtxt(os.path.join(DATA_ROOT, "TEP_Braatz", "d00.dat")).T.std(0) + 1e-8
    test = [sliding((x0 - mu) / sd, L, stride)] + [sliding(s.test[160:], L, stride) for s in segs]
    XT = np.concatenate(test); yt = np.concatenate([np.full(len(t), i) for t, i in zip(test, [0] + list(range(1, 22)))])
    return X, yl, XT, yt


def e3_e4(models_unused=None):
    L = CFG["window_tep"]; X, yl, XT, yt = tep_pool(L)
    Xf, XTf = X.reshape(len(X), -1), XT.reshape(len(XT), -1)
    rows3, rows4 = [], []
    for sd in SEEDS:
        m = train_protorep(X, 22, EP, sd, lambdas=LAM)
        ae = train_ae(X, "conv", EP, sd); tv = train_ts2vec(X, EP, sd)
        pca = PCA(20, random_state=sd).fit(Xf)
        reps = {"Raw window": (Xf, XTf), "PCA-20": (pca.transform(Xf), pca.transform(XTf)),
                "Conv-AE latent": (batched(ae.enc, X), batched(ae.enc, XT)),
                "TS2Vec-lite": (batched(tv.embed, X), batched(tv.embed, XT)),
                "ProtoRep embedding (ours)": (batched(m.embed, X), batched(m.embed, XT))}
        assign = batched(lambda x: m(x)["dist"].argmin(1), X)
        rows3.append(dict(method="ProtoRep nearest prototype (ours, K=22)", seed=sd, nmi=nmi(yl, assign), ari=ari(yl, assign), n_used=len(np.unique(assign))))
        for nm, (E, ET) in reps.items():
            sc = StandardScaler().fit(E); E, ET = sc.transform(E), sc.transform(ET)
            lab = KMeans(22, n_init=4, random_state=sd).fit_predict(E)
            rows3.append(dict(method=f"k-means(22) on {nm}", seed=sd, nmi=nmi(yl, lab), ari=ari(yl, lab), n_used=22))
            rng = np.random.default_rng(sd)
            for frac in [0.01, 0.05, 0.10, 1.0]:
                idx = np.concatenate([rng.choice(np.where(yl == c)[0], max(1, int(round(frac * (yl == c).sum()))), replace=False) for c in np.unique(yl)])
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(E[idx], yl[idx]); p = clf.predict(ET)
                rows4.append(dict(representation=nm, label_frac=frac, n_labels=len(idx), seed=sd,
                                  macro_f1=f1_score(yt, p, average="macro"), accuracy=accuracy_score(yt, p)))
        print("E3/E4 seed", sd, flush=True)
    d3 = pd.DataFrame(rows3); d3.to_csv(f"{RES}/e3_purity_raw.csv", index=False)
    g3 = d3.groupby("method")[["nmi", "ari"]].agg(["mean", "std"]).round(4); g3.columns = ["_".join(c) for c in g3.columns]
    g3.reset_index().sort_values("nmi_mean", ascending=False).to_csv(f"{RES}/e3_purity_summary.csv", index=False)
    d4 = pd.DataFrame(rows4); d4.to_csv(f"{RES}/e4_label_efficiency_raw.csv", index=False)
    g4 = d4.groupby(["representation", "label_frac"])[["macro_f1", "accuracy"]].agg(["mean", "std"]).round(4); g4.columns = ["_".join(c) for c in g4.columns]
    g4 = g4.reset_index(); g4.to_csv(f"{RES}/e4_label_efficiency_summary.csv", index=False)
    plt.figure(figsize=(6, 4))
    for nm, g in g4.groupby("representation"):
        plt.errorbar(g.label_frac * 100, g.macro_f1_mean, g.macro_f1_std, fmt="o-", label=nm)
    plt.xscale("log"); plt.xlabel("% labelled training windows"); plt.ylabel("macro-F1 (22 TEP classes)"); plt.legend(fontsize=7)
    plt.title("E4 label efficiency (linear probe, TEP)"); plt.tight_layout(); plt.savefig(f"{FIG}/e4_label_efficiency.png", dpi=140); plt.close()


def e4b_weak():
    """Weak supervision: reveal labels for 10% of windows of the 'other' SKAB runs; evaluate on held-out valve1+valve2 runs."""
    segs = load_skab(); L = CFG["window"]
    tr = [s for s in segs if s.group == "other"]; te = [s for s in segs if s.group != "other"]
    Xn = windows_from([s.ref for s in segs], L, 2)
    Xo = np.concatenate([sliding(np.concatenate([s.ref[-(L - 1):], s.test]), L, 2) for s in tr])
    yo = np.concatenate([s.labels[::2][:len(sliding(np.concatenate([s.ref[-(L - 1):], s.test]), L, 2))] for s in tr])
    XT = np.concatenate([sliding(np.concatenate([s.ref[-(L - 1):], s.test]), L) for s in te]); y = np.concatenate([s.labels for s in te])
    rows = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd); keep = rng.random(len(Xo)) < 0.10
        X = np.concatenate([Xn, Xo[keep]]); w = np.concatenate([np.zeros(len(Xn)), yo[keep]]).astype(int)
        for nm, weak, data in [("ProtoRep unsupervised (normal refs only)", None, Xn), ("ProtoRep + weak labels (10% of 'other' runs)", w, X)]:
            m = train_protorep(data, K, EP, sd, lambdas=LAM, weak=weak)
            rows.append(dict(method=nm, seed=sd, n_weak_pos=int((w == 1).sum()) if weak is not None else 0, **detection_report(batched(m.score, XT), y)))
        print("E4b", sd, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e4b_weak_raw.csv", index=False)
    g = df.groupby("method")[["auc_roc", "auc_pr", "f1_best", "f1_best_pa"]].agg(["mean", "std"]).round(4); g.columns = ["_".join(c) for c in g.columns]
    g.reset_index().to_csv(f"{RES}/e4b_weak_summary.csv", index=False)


def e5_ablation():
    X, XT, y, ep, _ = dataset("SKAB"); rows = []
    variants = {"full (K=16)": dict(), "no diversity loss": dict(lambdas=dict(div=0.0)), "no sensor gates": dict(sensor_gates=False),
                "no contrastive": dict(lambdas=dict(con=0.0)), "K=4": dict(n_prototypes=4), "K=8": dict(n_prototypes=8), "K=32": dict(n_prototypes=32)}
    for nm, kw in variants.items():
        for sd in SEEDS:
            lam = {**LAM, **kw.get("lambdas", {})}
            m = train_protorep(X, kw.get("n_prototypes", K), ep, sd, sensor_gates=kw.get("sensor_gates", True), lambdas=lam)
            P = m.prototypes.detach(); pd_ = ((P[:, None] - P[None]) ** 2).sum((-1, -2)) / m.C
            uniq = len(np.unique(m.proto_src.numpy()))
            gm = batched(lambda x: m(x)["g"], XT).mean(0)
            rows.append(dict(variant=nm, seed=sd, unique_prototypes=uniq, mean_gate=float(gm.mean()), gate_entropy_bits=float(-(gm / gm.sum() * np.log2(gm / gm.sum())).sum()),
                             **detection_report(batched(m.score, XT), y)))
        print("E5", nm, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e5_ablation_raw.csv", index=False)
    g = df.groupby("variant").agg(["mean", "std"]).drop(columns="seed", level=0).round(4); g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index(); g.to_csv(f"{RES}/e5_ablation_summary.csv", index=False)
    plt.figure(figsize=(7, 4)); plt.bar(g.variant, g.auc_pr_mean, yerr=g.auc_pr_std); plt.xticks(rotation=25, ha="right"); plt.ylabel("AUC-PR (SKAB)")
    plt.ylim(g.auc_pr_mean.min() - 0.05, g.auc_pr_mean.max() + 0.03); plt.title("E5 ablations (3 seeds)"); plt.tight_layout(); plt.savefig(f"{FIG}/e5_ablation.png", dpi=140); plt.close()


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4b", "e5"]
    meta = {}; models = None
    for w in which:
        t = time.perf_counter()
        if w == "e1": models = e1_detection()
        elif w == "e2": e2_faithfulness(models)
        elif w == "e3": e3_e4()
        elif w == "e4b": e4b_weak()
        elif w == "e5": e5_ablation()
        meta[w] = time.perf_counter() - t; print(w, "done", round(meta[w], 1), flush=True)
    json.dump(meta, open(f"{RES}/runtime_{'_'.join(which)}.json", "w"), indent=1)
