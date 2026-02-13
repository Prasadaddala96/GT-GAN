
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch



# Reproducibility

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Scaling + windowing

def minmax_fit(x_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit MinMax on [T, D]."""
    min_val = np.min(x_2d, axis=0)
    max_val = np.max(x_2d, axis=0)
    denom = (max_val - min_val)
    denom[denom == 0] = 1.0
    return min_val, denom


def minmax_transform(x_2d: np.ndarray, min_val: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Transform [T, D]."""
    return (x_2d - min_val) / denom


def make_windows(x_2d: np.ndarray, start: int, end: int, seq_len: int) -> np.ndarray:
    """
    x_2d: [T, D]
    returns: [N, L, D]
    """
    if end - start <= seq_len:
        raise ValueError(f"Segment too short: start={start}, end={end}, seq_len={seq_len}")
    windows = [x_2d[i : i + seq_len] for i in range(start, end - seq_len)]
    return np.asarray(windows, dtype=np.float32)


def ett_split_indices(seq_len: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    ETT split used in Autoformer family:
      train: 12 months
      val: 4 months
      test: 4 months
    With overlap by seq_len for val/test context.
    """
    month = 30 * 24
    train_end = 12 * month
    val_end = train_end + 4 * month
    test_end = train_end + 8 * month
    train = (0, train_end)
    val = (train_end - seq_len, val_end)
    test = (val_end - seq_len, test_end)
    return train, val, test


def fallback_split(T: int, seq_len: int, train_ratio=0.7, val_ratio=0.1) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    train_end = int(train_ratio * T)
    val_end = int((train_ratio + val_ratio) * T)
    tr = (0, train_end)
    va = (max(0, train_end - seq_len), val_end)
    te = (max(0, val_end - seq_len), T)
    return tr, va, te


def add_time_channel(windows_3d: np.ndarray) -> np.ndarray:
    """
    Convert (N, L, D) -> (N, L, D+1) with last channel = 0..L-1
    """
    N, L, D = windows_3d.shape
    t = np.arange(L, dtype=np.float32).reshape(1, L, 1)
    t = np.repeat(t, N, axis=0)
    return np.concatenate([windows_3d, t], axis=-1).astype(np.float32)


def to_list_of_sequences(windows_3d: np.ndarray) -> list[np.ndarray]:
    """
    GTGAN batch_generator expects a python list.
    """
    return [windows_3d[i] for i in range(windows_3d.shape[0])]


# Data loading

def load_csv_series(
    csv_path: str,
    features: List[str],
    datetime_col: Optional[str] = "date",
) -> np.ndarray:
    """
    Load CSV -> return [T, D] float32
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if datetime_col and datetime_col in df.columns:
        df = df.drop(columns=[datetime_col])

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    df = df[features]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df.values.astype(np.float32)



# Standard output directory + metadata

def make_run_dir(
    save_root: str,
    model_name: str,
    dataset_name: str,
    seq_len: int,
    D: int,
    seed: int,
    tag: str | None = None,
) -> Path:
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    parts = [save_root, model_name, dataset_name, f"seq{seq_len}", f"D{D}", f"seed{seed}"]
    if tag:
        parts.append(tag)
    parts.append(stamp)
    run_dir = Path(*parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def touch_done(run_dir: Path, text: str = "DONE\n") -> None:
    (run_dir / "DONE.txt").write_text(text, encoding="utf-8")



# Build train/test datasets for GT-GAN (ETT split)

@dataclass
class WindowedData:
    train_windows_t: np.ndarray   # (N, L, D+1)
    test_windows_t: np.ndarray    # (N, L, D+1)
    train_list: list[np.ndarray]
    test_list: list[np.ndarray]
    min_val: np.ndarray
    denom: np.ndarray
    splits: dict


def build_etth1_windows_for_gtgan(
    csv_path: str,
    features: List[str],
    seq_len: int,
    datetime_col: str = "date",
) -> WindowedData:
    x = load_csv_series(csv_path, features, datetime_col=datetime_col)
    T, D = x.shape

    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = ett_split_indices(seq_len)
    if T < te_e:
        tr, va, te = fallback_split(T, seq_len)
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = tr, va, te

    # scale train-only
    min_val, denom = minmax_fit(x[tr_s:tr_e])
    x_scaled = minmax_transform(x, min_val, denom)

    # windows
    train_w = make_windows(x_scaled, tr_s, tr_e, seq_len)
    test_w = make_windows(x_scaled, te_s, te_e, seq_len)

    # add time channel
    train_w_t = add_time_channel(train_w)
    test_w_t = add_time_channel(test_w)

    return WindowedData(
        train_windows_t=train_w_t,
        test_windows_t=test_w_t,
        train_list=to_list_of_sequences(train_w_t),
        test_list=to_list_of_sequences(test_w_t),
        min_val=min_val,
        denom=denom,
        splits={
            "train": [int(tr_s), int(tr_e)],
            "val": [int(va_s), int(va_e)],
            "test": [int(te_s), int(te_e)],
            "T": int(T),
            "D": int(D),
        },
    )
