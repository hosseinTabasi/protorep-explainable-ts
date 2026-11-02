"""Loaders for SKAB and Tennessee Eastman (Braatz) with per-file reference normalisation."""
from __future__ import annotations
import glob
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("INDUSTRIAL_TS_DATA", "/workspace/industrial-ts-data")
SKAB_SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
                "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
EXCLUDED_SKAB = {"other/2.csv"}  # anomaly starts at t=104, inside the 400-sample reference prefix


@dataclass
class Segment:
    name: str
    group: str          # operating regime / file family (e.g. valve1, other, TEP fault id)
    ref: np.ndarray     # (T_ref, C) normal reference, standardised
    test: np.ndarray    # (T_test, C) standardised with reference statistics
    labels: np.ndarray  # (T_test,) 0/1


def _standardise(ref: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = ref.mean(0), ref.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (ref - mu) / sd, (x - mu) / sd


def load_skab(root: str | None = None, ref_len: int = 400) -> list[Segment]:
    """Each SKAB experiment file: first `ref_len` samples are the normal reference
    (fit scaler + train model), the remainder is the test stream."""
    root = os.path.join(root or DATA_ROOT, "SKAB", "data")
    segs = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
        rel = os.path.relpath(f, root)
        if rel.startswith("anomaly-free") or rel in EXCLUDED_SKAB:
            continue
        d = pd.read_csv(f, sep=";")
        x = d[SKAB_SENSORS].to_numpy(float)
        y = d["anomaly"].to_numpy(int)
        r, t = _standardise(x[:ref_len], x)
        segs.append(Segment(rel, rel.split("/")[0], r, t[ref_len:], y[ref_len:]))
    return segs


def load_skab_anomaly_free(root: str | None = None) -> np.ndarray:
    root = os.path.join(root or DATA_ROOT, "SKAB", "data")
    x = pd.read_csv(os.path.join(root, "anomaly-free", "anomaly-free.csv"), sep=";")[SKAB_SENSORS].to_numpy(float)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def load_tep(root: str | None = None, faults: list[int] | None = None, onset: int = 160) -> tuple[np.ndarray, list[Segment]]:
    """Braatz TEP: d00.dat (normal training, stored transposed) is the reference;
    dNN_te.dat test runs have the fault introduced after sample `onset`."""
    root = os.path.join(root or DATA_ROOT, "TEP_Braatz")
    ref = np.loadtxt(os.path.join(root, "d00.dat"))
    if ref.shape[0] == 52:
        ref = ref.T
    faults = faults or list(range(1, 22))
    mu, sd = ref.mean(0), ref.std(0) + 1e-8
    refz = (ref - mu) / sd
    segs = []
    for k in faults:
        x = np.loadtxt(os.path.join(root, f"d{k:02d}_te.dat"))
        y = (np.arange(len(x)) >= onset).astype(int)
        segs.append(Segment(f"fault{k:02d}", f"fault{k:02d}", refz, (x - mu) / sd, y))
    return refz, segs


def load_tep_train(root: str | None = None, faults: list[int] | None = None) -> tuple[list[np.ndarray], list[int]]:
    """Training runs dNN.dat (fault active from sample 20 of 480) standardised with d00 stats."""
    root = os.path.join(root or DATA_ROOT, "TEP_Braatz")
    ref = np.loadtxt(os.path.join(root, "d00.dat"))
    ref = ref.T if ref.shape[0] == 52 else ref
    mu, sd = ref.mean(0), ref.std(0) + 1e-8
    xs, ids = [(ref - mu) / sd], [0]
    for k in faults or list(range(1, 22)):
        x = np.loadtxt(os.path.join(root, f"d{k:02d}.dat"))
        xs.append((x[20:] - mu) / sd)
        ids.append(k)
    return xs, ids
