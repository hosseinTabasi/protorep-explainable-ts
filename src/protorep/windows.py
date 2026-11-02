from __future__ import annotations
import numpy as np


def sliding(x: np.ndarray, L: int, stride: int = 1) -> np.ndarray:
    """(T, C) -> (N, C, L) float32 windows; window i ends at sample i*stride + L - 1."""
    w = np.lib.stride_tricks.sliding_window_view(np.asarray(x, np.float32), L, axis=0)
    return np.ascontiguousarray(w[::stride])


def windows_from(xs: list[np.ndarray], L: int, stride: int = 1) -> np.ndarray:
    return np.concatenate([sliding(x, L, stride) for x in xs if len(x) >= L])
