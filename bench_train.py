
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import re
from pathlib import Path
import importlib.util
import argparse
import numpy as np
import torch

from bench_utils import (
    set_seed,
    build_etth1_windows_for_gtgan,
    make_run_dir,
    write_meta,
    touch_done,
)

print("DEBUG torch loaded:", torch.__version__)



# Runtime patch loader 

def load_patched_gtgan_stocks(gtgan_dir: str):
    src_path = os.path.join(gtgan_dir, "GTGAN_stocks.py")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"GTGAN_stocks.py not found in: {gtgan_dir}")

    if gtgan_dir not in sys.path:
        sys.path.insert(0, gtgan_dir)

    txt = Path(src_path).read_text(encoding="utf-8", errors="ignore")

    # Remove visualization 
    txt = txt.replace("from metrics.visualization_metrics import visualization\n", "")
    txt = txt.replace("from metrics.visualization_metrics import visualization\r\n", "")
    txt = re.sub(r"^\s*visualization\s*\(.*\)\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*visualization\s*\([\s\S]*?\)\s*$", "", txt, flags=re.MULTILINE)

    # Seq-len safe time vector + final index (original code hardcoded 24 / 23)
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

    # Remove CUDA hardcoding
    txt = txt.replace(".cuda()", ".to(device)")
    txt = txt.replace(".to('cuda')", ".to(device)")
    txt = txt.replace('.to("cuda")', ".to(device)")

    # Safer zeros allocation (avoid referencing global 'device' wrongly)
    txt = re.sub(
        r"torch\.zeros\(\s*HH\.shape\[0\]\s*,\s*self\.hidden_size\s*\)\.to\(device\)",
        "torch.zeros(HH.shape[0], self.hidden_size, device=HH.device)",
        txt
    )
    txt = re.sub(
        r"torch\.zeros\(\s*out\.shape\[0\]\s*,\s*self\.hidden_size\s*\)\.to\(device\)",
        "torch.zeros(out.shape[0], self.hidden_size, device=out.device)",
        txt
    )

    # CPU checkpoint loading (if CUDA-saved)
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

    # numpy export safe
    txt = txt.replace(
        "generated_data_curr = x_hat.cpu().numpy()",
        "generated_data_curr = x_hat.detach().cpu().numpy()"
    )

    patched_path = os.path.join(gtgan_dir, "_GTGAN_stocks_patched.py")
    Path(patched_path).write_text(txt, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("gtgan_stocks_patched", patched_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod



# Bench Train entry
def parse_args():
    p = argparse.ArgumentParser(description="GT-GAN BenchTrain (ETTh1 first)")
    p.add_argument("--data-root", type=str, required=True, help="Repo data root, e.g. D:\\...\\data")
    p.add_argument("--csv-rel", type=str, required=True, help="CSV relative path under data-root")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--datetime-col", type=str, default="date")

    p.add_argument("--features", type=str, required=True, help="Comma-separated features (D columns)")
    p.add_argument("--seq-len", type=int, default=24)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--max-steps-metric", type=int, default=3)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default=None)

    # explicit cpu option
    p.add_argument("--use-cpu", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    gtgan_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(args.data_root, args.csv_rel)

    features = [c.strip() for c in args.features.split(",") if c.strip()]
    if not features:
        raise ValueError("Empty --features list")

    # Build windowed datasets (train/test), add GTGAN time channel
    w = build_etth1_windows_for_gtgan(
        csv_path=csv_path,
        features=features,
        seq_len=args.seq_len,
        datetime_col=args.datetime_col,
    )

    # Standard run directory 
    run_dir = make_run_dir(
        save_root=args.save_root,
        model_name="gtgan",
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        D=len(features),
        seed=args.seed,
        tag=args.tag,
    )

    # basic dataset info for reproducibility
    write_meta(run_dir, {
        "model": "GT-GAN",
        "impl": "GTGAN_stocks.py (runtime patched)",
        "dataset": args.dataset,
        "csv": csv_path,
        "features": features,
        "seq_len": args.seq_len,
        "D": len(features),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "max_steps_metric": args.max_steps_metric,
        "splits": w.splits,
        "notes": "Train/test windows scaled using TRAIN only; time-channel appended as last dim.",
    })

    print("=== GT-GAN BenchTrain ===")
    print("CSV:", csv_path)
    print("SEQ_LEN:", args.seq_len)
    print("Features:", features)
    print("Train windows_t:", w.train_windows_t.shape, " Test windows_t:", w.test_windows_t.shape)
    print("Run dir:", run_dir.resolve())

    # Load patched GTGAN module
    gt = load_patched_gtgan_stocks(gtgan_dir)

    # Monkeypatch TimeDataset + np.load used inside gt.main()
    # - First TimeDataset call -> train
    # - Second TimeDataset call -> test
    call_state = {"count": 0}
    def fake_timedataset(*_a, **_k):
        call_state["count"] += 1
        return w.train_list if call_state["count"] == 1 else w.test_list

    def fake_npload(*_a, **_k):
        # GTGAN_stocks expects np.load(...) to return test array
        return np.asarray(w.test_windows_t, dtype=np.float32)

    gt.TimeDataset = fake_timedataset
    gt.np.load = fake_npload

    # run GTGAN_stocks.main() 
    sys.argv = [
        "bench_train.py",
        "--use_cpu" if args.use_cpu else "",
        "--train",
        "--data", "stock",
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--num_epochs", str(args.epochs),
        "--max-steps", str(args.max_steps),
        "--max-steps-metric", str(args.max_steps_metric),
        "--save_dir", str(run_dir),
    ]
    sys.argv = [x for x in sys.argv if x != ""]

    print("\n=== Calling patched GTGAN_stocks.main() ===")
    print("Args:", " ".join(sys.argv[1:]))

    gt.main()

    touch_done(run_dir, "GTGAN bench_train finished.\n")
    print("\nDONE:", run_dir.resolve())


if __name__ == "__main__":
    main()
