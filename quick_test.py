import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch
print("DEBUG torch loaded:", torch.__version__)
import re
import sys
import random
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

# USER SETTINGS

SEED = 42
ROOT_PATH = r"D:\TimeSeries-Generative-Modeling-RnD\data"
DATA_PATH = r"ETT-small\ETTh1.csv"
ETT_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
DATETIME_COL = "date"
SEQ_LEN = 24
NUM_EPOCHS = 1
BATCH_SIZE = 128
HIDDEN_DIM = 24
MAX_STEPS = 300          # training iterations i
MAX_STEPS_METRIC = 3     # metric repeats 

SAVE_DIR = "./results_quicktest_gtgan_stocks"

# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# Scaling functions

def minmax_fit(x_2d: np.ndarray):
    """Fit MinMax on [T, D]."""
    min_val = np.min(x_2d, axis=0)
    max_val = np.max(x_2d, axis=0)
    denom = (max_val - min_val)
    denom[denom == 0] = 1.0
    return min_val, denom


def minmax_transform(x_2d: np.ndarray, min_val, denom):
    """Transform [T, D]."""
    return (x_2d - min_val) / denom


def make_windows(x_2d: np.ndarray, start: int, end: int, seq_len: int):


    #   output shape: (N, seq_len, D)

    if end - start <= seq_len:
        raise ValueError(f"Segment too short: start={start}, end={end}, seq_len={seq_len}")

    windows = []
    for i in range(start, end - seq_len):
        windows.append(x_2d[i : i + seq_len])
    return np.asarray(windows, dtype=np.float32)


def ett_split_indices(seq_len: int):
    """
    ETT split used in Autoformer family repos:
      train: 12 months
      val: 4 months
      test: 4 months
    With overlap by seq_len for val context.
    """
    month = 30 * 24
    train_end = 12 * month
    val_end = train_end + 4 * month
    test_end = train_end + 8 * month

    train = (0, train_end)
    val = (train_end - seq_len, val_end)
    test = (val_end - seq_len, test_end)
    return train, val, test


def load_etth1(csv_path: str) -> np.ndarray:
    """
    Load ETTh1.csv -> return x [T, 7]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # drop datetime if present
    if DATETIME_COL in df.columns:
        df = df.drop(columns=[DATETIME_COL])

    # enforce exact columns
    missing = [c for c in ETT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ETTh1.csv missing expected columns: {missing}")

    df = df[ETT_COLUMNS]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df.values.astype(np.float32)


def add_time_channel(windows_3d: np.ndarray) -> np.ndarray:
    """
    GTGAN_stocks expects each sequence to have an extra last channel
    that behaves like a “time index” per step.
    So we convert:
      (N, L, D) -> (N, L, D+1)
    where last channel is 0..L-1.
    """
    N, L, D = windows_3d.shape
    t = np.arange(L, dtype=np.float32).reshape(1, L, 1)
    t = np.repeat(t, N, axis=0)  # (N, L, 1)
    return np.concatenate([windows_3d, t], axis=-1).astype(np.float32)


def to_list_of_sequences(windows_3d: np.ndarray):
    """
    GTGAN batch_generator indexes dataset like a python list.
    So we convert (N, L, D) array -> list of (L, D) arrays.
    """
    return [windows_3d[i] for i in range(windows_3d.shape[0])]


# Runtime patch loader for GTGAN_stocks.py CPU
def load_patched_gtgan_stocks(gtgan_dir: str):
    """
    Patch GTGAN_stocks.py to be:
      - CPU-safe (no hardcoded cuda / .cuda())
      - checkpoint load safe on CPU (map_location=device)
      - seq_len-safe for time vector + final index
      - spline-safe (time length matches x length)
      - visualization removed
      - numpy export safe (detach before numpy)
    """
    import re
    import sys
    import os
    from pathlib import Path
    import importlib.util

    src_path = os.path.join(gtgan_dir, "GTGAN_stocks.py")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"GTGAN_stocks.py not found in: {gtgan_dir}")


    if gtgan_dir not in sys.path:
        sys.path.insert(0, gtgan_dir)

    txt = Path(src_path).read_text(encoding="utf-8", errors="ignore")


    # Remove visualization import + calls CPU version

    txt = txt.replace("from metrics.visualization_metrics import visualization\n", "")
    txt = txt.replace("from metrics.visualization_metrics import visualization\r\n", "")
    txt = re.sub(r"^\s*visualization\s*\(.*\)\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*visualization\s*\([\s\S]*?\)\s*$", "", txt, flags=re.MULTILINE)


    # Seq-len safe time vector + final index and remove cuda hardcoding

    txt = txt.replace(
        "torch.FloatTensor(list(range(24))).cuda()",
        "torch.FloatTensor(list(range(args.seq_len))).to(device)"
    )
    txt = txt.replace(
        "(torch.ones(batch_size) * 23).cuda()",
        "(torch.ones(batch_size, device=device) * (args.seq_len - 1))"
    )
    txt = txt.replace(
        "(torch.ones(dataset_size) * 23).cuda()",
        "(torch.ones(dataset_size, device=device) * (args.seq_len - 1))"
    )

    # Replace remaining .cuda() calls
    txt = txt.replace(".cuda()", ".to(device)")
    # Replace explicit to('cuda')
    txt = txt.replace(".to('cuda')", ".to(device)")
    txt = txt.replace('.to("cuda")', ".to(device)")


    #Use the incoming tensor device instead (HH.device / out.device).

    txt = re.sub(
        r"torch\.zeros\(\s*HH\.shape\[0\]\s*,\s*self\.hidden_size\s*\)\.to\(device\)",
        "torch.zeros(HH.shape[0], self.hidden_size, device=HH.device)",
        txt
    )

    #if there are variants using out.shape[0]
    txt = re.sub(
        r"torch\.zeros\(\s*out\.shape\[0\]\s*,\s*self\.hidden_size\s*\)\.to\(device\)",
        "torch.zeros(out.shape[0], self.hidden_size, device=out.device)",
        txt
    )

  
    # CPU checkpoint loading (CUDA-saved .pt files)
  
    for name in ["embedder", "generator", "supervisor", "recovery", "discriminator"]:
        txt = txt.replace(
            f'torch.load(path/"{name}.pt")',
            f'torch.load(path/"{name}.pt", map_location=device)'
        )
    # Spline coeffs call: time length must equal x time dim
    pattern = r"^(\s*)train_coeffs\s*=\s*controldiffeq\.natural_cubic_spline_coeffs\(time,\s*x\)\s*$"

    def repl(m):
        indent = m.group(1)
        return (
            f"{indent}time = time[:x.size(1)]\n"
            f"{indent}train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, x)"
        )

    txt = re.sub(pattern, repl, txt, flags=re.MULTILINE)

    # numpy export at end of training (detach before numpy)

    txt = txt.replace(
        "generated_data_curr = x_hat.cpu().numpy()",
        "generated_data_curr = x_hat.detach().cpu().numpy()"
    )


    #patched module

    patched_path = os.path.join(gtgan_dir, "_GTGAN_stocks_patched.py")
    Path(patched_path).write_text(txt, encoding="utf-8")

    # Load it as a module
    spec = importlib.util.spec_from_file_location("gtgan_stocks_patched", patched_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Main quick test
def main():
    set_seed(SEED)

    gtgan_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    csv_path = os.path.join(ROOT_PATH, DATA_PATH)

    print("=== GT-GAN (stocks) Quick Test on ETTh1 ===")
    print("GT-GAN dir:", gtgan_dir)
    print("CSV:", csv_path)
    print("SEQ_LEN:", SEQ_LEN)
    print("Columns:", ETT_COLUMNS)

    # Load raw
    x = load_etth1(csv_path)
    T, D = x.shape
    print(f"Raw series shape: T={T}, D={D}")

    # Split indices
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = ett_split_indices(SEQ_LEN)
    if T < te_e:
        print(f"WARNING: T={T} < expected test_end={te_e}. Falling back to 70/10/20 split.")
        train_end = int(0.7 * T)
        val_end = int(0.8 * T)
        tr_s, tr_e = 0, train_end
        va_s, va_e = max(0, train_end - SEQ_LEN), val_end
        te_s, te_e = max(0, val_end - SEQ_LEN), T

    print("Split indices:")
    print(f"  Train: {tr_s} -> {tr_e}")
    print(f"  Val:   {va_s} -> {va_e}")
    print(f"  Test:  {te_s} -> {te_e}")

    # Scale using TRAIN only (no leakage)
    min_val, denom = minmax_fit(x[tr_s:tr_e])
    x_scaled = minmax_transform(x, min_val, denom)

    #  Make windows
    train_windows = make_windows(x_scaled, tr_s, tr_e, SEQ_LEN)
    test_windows = make_windows(x_scaled, te_s, te_e, SEQ_LEN)
    print("\nWindowed shapes (N, L, D):")
    print("  train_windows:", train_windows.shape)
    print("  test_windows: ", test_windows.shape)

    # Add time channel for GTGAN_stocks
    train_windows_t = add_time_channel(train_windows)  # (N, L, D+1)
    test_windows_t = add_time_channel(test_windows)    # (N, L, D+1)

    print("\nAfter adding time channel (N, L, D+1):")
    print("  train_windows_t:", train_windows_t.shape)
    print("  test_windows_t: ", test_windows_t.shape)

    # Convert to python lists (GTGAN uses batch_generator(dataset, batch_size) with indexing)
    train_dataset = to_list_of_sequences(train_windows_t)
    test_dataset = to_list_of_sequences(test_windows_t)

    # Load patched GTGAN_stocks module
    gt = load_patched_gtgan_stocks(gtgan_dir)

    # Monkeypatch TimeDataset + np.load used inside gt.main (so we can reuse its training+metrics flow)
    #    - First TimeDataset call -> train_dataset
    #    - Second TimeDataset call -> test_dataset
    call_state = {"count": 0}

    def fake_timedataset(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return train_dataset
        return test_dataset

    def fake_npload(*args, **kwargs):
        return np.asarray(test_windows_t, dtype=np.float32)

    gt.TimeDataset = fake_timedataset
    gt.np.load = fake_npload

    #  Run their main() with CLI args (no need to manually build the giant args object)
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "quick_test.py",
        "--use_cpu",
        "--train",
        "--data", "stock",
        "--seq-len", str(SEQ_LEN),
        "--batch-size", str(BATCH_SIZE),
        "--num_epochs", str(NUM_EPOCHS),
        "--max-steps", str(MAX_STEPS),
        "--max-steps-metric", str(MAX_STEPS_METRIC),
        "--save_dir", str(save_dir),
    ]


    print("\n=== Running patched GTGAN_stocks.main() ===")
    print("Args:", " ".join(sys.argv[1:]))

    gt.main()

    done_path = save_dir / "DONE.txt"
    done_path.write_text("GTGAN quick_test finished.\n", encoding="utf-8")
    print(f"\nSaved outputs under: {save_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
