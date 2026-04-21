#!/usr/bin/env python3
"""
main.py - Tensorized MinHash: full pipeline runner

Produces all three deliverables:
  1. Tensor-Aware Hashing Module (runs and validates)
  2. Computational Resource Profile (memory + RAM + speed tables)
  3. Local Scalability Prototype (PySpark demo or multiprocessing fallback)

Usage:
    python main.py             # full run
    python main.py --quick     # reduced sizes, faster
    python main.py --spark     # enable PySpark (requires pyspark installed)
    python main.py --only 2    # run only deliverable 2
"""

import argparse
import logging
import os
import sys

# Repo root is on sys.path for log_setup; add tensorized_minhash/ for engine imports.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.join(_REPO_ROOT, "tensorized_minhash")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PKG_DIR)

# Default CSV path resolves to the package data directory regardless of cwd
_DEFAULT_CSV = os.path.join(_PKG_DIR, "data", "wed_data.csv")

from log_setup import setup_logging  # noqa: E402

setup_logging()

# Silence the per-call TTMinHashConfig INFO lines - they're useful in library
# context but clutter the benchmark output display.
logging.getLogger("core.config").setLevel(logging.WARNING)
logging.getLogger("data.builder").setLevel(logging.WARNING)
logging.getLogger("data.loader").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_W = 68  # total output width


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def section(title: str) -> None:
    """Top-level section banner."""
    bar = "=" * _W
    print(f"\n{bar}")
    print(f" {title}")
    print(bar)


def subsection(title: str) -> None:
    """Sub-section divider inside a section."""
    print(f"\n ┌ {title}")
    print(f" └──{'─' * (_W - 4)}")


def kv(label: str, value: str, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{label:<28} {value}")


def rule(char: str = "-", indent: int = 2) -> None:
    print(" " * indent + char * (_W - indent))


def table(headers, rows, col_widths=None) -> None:
    if col_widths is None:
        col_widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def badge(label: str, ok: bool) -> str:
    return f"[{'✓' if ok else 'X'}] {label}"


# --------------------------------------------------------------------------
# Shared: load and filter real data
# --------------------------------------------------------------------------

def load_and_filter(csv_path: str):
    """Load CIC-IDS2017 CSV, filter noise, return (df_filtered, shape)."""
    from tensorized_minhash.data.builder import NetworkTensorBuilder

    df = NetworkTensorBuilder.load_cic_ids2017(csv_path)

    n_src_raw = df["src_ip"].nunique()
    n_dst_raw = df["dst_ip"].nunique()
    n_port_raw = df["port"].nunique()

    port_counts = df["port"].value_counts()
    meaningful_ports = port_counts[port_counts > 10].index
    df_filtered = df[df["port"].isin(meaningful_ports)].copy()

    dst_counts = df_filtered["dst_ip"].value_counts()
    meaningful_dsts = dst_counts[dst_counts > 100].index
    df_filtered = df_filtered[df_filtered["dst_ip"].isin(meaningful_dsts)].copy()

    n_src = df_filtered["src_ip"].nunique()
    n_dst = df_filtered["dst_ip"].nunique()
    n_port = df_filtered["port"].nunique()

    print()
    kv("Total rows (raw):", f"{len(df):>10,}")
    kv(" src IPs / dst IPs / ports:", f"{n_src_raw} / {n_dst_raw} / {n_port_raw}")
    kv("After noise filter:", f"{len(df_filtered):>10,} rows kept")
    kv(" src IPs / dst IPs / ports:", f"{n_src} / {n_dst} / {n_port}")

    return df_filtered, (n_src, n_dst, n_port)


# --------------------------------------------------------------------------
# Deliverable 1: Tensor-Aware Hashing Module
# --------------------------------------------------------------------------

def run_hashing_module_demo(quick: bool = False):
    section("1 • Tensor-Aware Hashing Module")

    from tensorized_minhash.benchmarks import ground_truth_jaccard
    from tensorized_minhash.core import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig
    from tensorized_minhash.data.builder import NetworkTensorBuilder
    import numpy as np

    subsection("Dataset")
    df_filtered, (n_src, n_dst, n_port) = load_and_filter(_DEFAULT_CSV)

    shape = (30, 30, 30) if quick else (n_src, n_dst, n_port)

    subsection("Tensor construction")
    kv("Shape:", str(shape))
    kv("Total cells:", f"{int(np.prod(shape)):,}")

    builder = NetworkTensorBuilder(n_src=shape[0], n_dst=shape[1], n_port=shape[2])
    half = len(df_filtered) // 2
    tensor_a = builder.build_tensor(df_filtered.iloc[:half])
    tensor_b = builder.build_tensor(df_filtered.iloc[half:])

    dens_a = tensor_a.mean() * 100
    dens_b = tensor_b.mean() * 100
    kv(
        "Tensor A (non-zeros / total):",
        f"{int(tensor_a.sum()):,} / {tensor_a.size:,} ({dens_a:.2f}% density)",
    )
    kv(
        "Tensor B (non-zeros / total):",
        f"{int(tensor_b.sum()):,} / {tensor_b.size:,} ({dens_b:.2f}% density)",
    )

    subsection("Hash function memory")
    cfg = TTMinHashConfig(shape=shape, num_hashes=128, seed=42)
    kron = KroneckerMinHash(cfg)
    tt = TTDecomposedMinHash(cfg)
    mem_kron = kron.memory_stats()
    mem_tt = tt.memory_stats()

    table(
        ["Method", "Params", "Memory", "vs Full matrix"],
        [
            [
                "Kronecker MinHash",
                f"{mem_kron['kron_params']:,}",
                f"{mem_kron['kron_bytes'] / 1024:.1f} KB",
                f"{mem_kron['compression_ratio']:,}x smaller",
            ],
            [
                "Tensor Train (TT)",
                f"{mem_tt['tt_params']:,}",
                f"{mem_tt['tt_bytes'] / 1024:.1f} KB",
                f"{mem_tt['compression_ratio']:,}x smaller",
            ],
            [
                "Full matrix (baseline)",
                f"{mem_kron['full_params_theoretical']:,}",
                f"{mem_kron['full_bytes_theoretical'] / 1_000_000:.1f} MB",
                "-",
            ],
        ],
        col_widths=[24, 12, 12, 18],
    )

    subsection("Jaccard similarity (k=128 hash planes)")
    sig_a_kron = kron.hash_tensor(tensor_a)
    sig_b_kron = kron.hash_tensor(tensor_b)
    sig_a_tt = tt.hash_tensor(tensor_a)
    sig_b_tt = tt.hash_tensor(tensor_b)

    exact_j = ground_truth_jaccard(tensor_a, tensor_b)
    kron_j = kron.jaccard_from_signatures(sig_a_kron, sig_b_kron)
    tt_j = tt.jaccard_from_signatures(sig_a_tt, sig_b_tt)

    def _bar(j: float) -> str:
        filled = int(j * 20)
        return "█" * filled + "░" * (20 - filled)

    print(f"\n  {'Method':<28} {'Jaccard':>8} {'Error':>7}  Visual")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 7}  {'-' * 22}")
    print(f"  {'Exact (ground truth)':<28} {exact_j:>8.4f} {'':>7}  {_bar(exact_j)}")
    print(
        f"  {'Kronecker MinHash':<28} {kron_j:>8.4f} {abs(kron_j - exact_j):>7.4f}  {_bar(kron_j)}"
    )
    print(
        f"  {'Tensor Train (TT)':<28} {tt_j:>8.4f} {abs(tt_j - exact_j):>7.4f}  {_bar(tt_j)}"
    )


# --------------------------------------------------------------------------
# Deliverable 2: Computational Resource Profile
# --------------------------------------------------------------------------

def run_resource_profile(quick: bool = False, real_tensors: list = None):
    section("Computational Resource Profile")

    from tensorized_minhash.benchmarks import (
        benchmark_accuracy,
        benchmark_accuracy_from_tensors,
        benchmark_memory,
        benchmark_ram,
        benchmark_speed,
        benchmark_speed_real,
    )

    # Parameter count across tensor sizes
    subsection("2a • Parameter Count - Kronecker vs Full Matrix")
    shapes_3d = [(10, 10, 10), (30, 30, 30), (50, 50, 50), (100, 100, 100)]
    shapes_4d = [(10, 10, 10, 10), (20, 20, 20, 20)]
    all_shapes = shapes_3d + ([] if quick else shapes_4d)

    mem_results = benchmark_memory(all_shapes, num_hashes=128)
    table(
        ["Shape", "Total cells", "Kron params", "Full params", "Compression"],
        [
            [
                str(r["shape"]),
                f"{r['total_cells']:,}",
                f"{r['kron_params']:,}",
                f"{r['full_params_theoretical']:,}",
                f"{r['compression_ratio']:,}x",
            ]
            for r in mem_results
        ],
        col_widths=[22, 14, 14, 18, 14],
    )

    # Accuracy
    subsection("2b • Accuracy - Jaccard Approximation")

    if real_tensors is not None and len(real_tensors) >= 2:
        n_pairs = 30 if quick else 100
        acc = benchmark_accuracy_from_tensors(
            tensors=real_tensors,
            shape=real_tensors[0].shape,
            num_hashes=128,
            n_pairs=n_pairs,
        )
        # benchmark_accuracy_from_tensors returns flat results; wrap for display
        scenario_data = None
    else:
        print("  No real tensors provided - using synthetic data\n")
        n_pairs = 30 if quick else 100
        acc = benchmark_accuracy(n_pairs=n_pairs, shape=(30, 30, 30), num_hashes=128)
        scenario_data = acc.get("scenarios")

    method_map = {
        "tensorized_kron": "Kronecker MinHash",
        "tt_decomposition": "Tensor Train (TT)",
        "datasketch_baseline": "Datasketch (standard)",
    }
    col_widths = [22, 8, 8, 7, 7, 10, 11]
    headers = ["Method", "Exact J", "Est. J", "MAE", "RMSE", "Pearson r", "Spearman ρ"]

    def method_rows(result_dict):
        avg_exact = result_dict.get("avg_exact", 0.0)
        return [
            [
                method_map[k],
                f"{avg_exact:.4f}",
                f"{result_dict[k]['avg_estimated']:.4f}",
                f"{result_dict[k]['mae']:.4f}",
                f"{result_dict[k]['rmse']:.4f}",
                f"{result_dict[k]['pearson_r']:.4f}",
                f"{result_dict[k]['spearman_r']:.4f}",
            ]
            for k in method_map
            if k in result_dict
        ]

    # Per-scenario tables
    if scenario_data:
        scenario_labels = {
            "high": "Scenario 1 - High similarity   (J ≈ 0.85-0.98)",
            "medium": "Scenario 2 - Medium similarity (J ≈ 0.35-0.65)",
            "low": "Scenario 3 - Low similarity    (J ≈ 0.01-0.12)",
        }
        for key, label in scenario_labels.items():
            s = scenario_data[key]
            j_lo, j_hi = s["jaccard_range"]
            subsection(f"{label}  [{j_lo:.2f}–{j_hi:.2f}] n={s['n_pairs']} pairs")
            table(headers, method_rows(s), col_widths=col_widths)
            print()

    # Combined / overall table
    j_lo, j_hi = acc["jaccard_range"]
    subsection(f"Overall (all scenarios)  [Jaccard {j_lo:.2f}–{j_hi:.2f}]")
    table(headers, method_rows(acc), col_widths=col_widths)
    print("  Proposal target: Spearman ρ > 0.85")

    # Speed
    subsection("2c • Hashing Throughput")

    if real_tensors is not None and len(real_tensors) >= 2:
        n_t = 20 if quick else len(real_tensors)
        speed = benchmark_speed_real(
            tensors=real_tensors[:n_t],
            num_hashes=128,
        )
    else:
        n_t = 100 if quick else 500
        speed = benchmark_speed(shape=(30, 30, 30), num_hashes=128, n_tensors=n_t)

    table(
        ["Method", "Tensors/sec", "ms/tensor"],
        [
            [k, f"{v['tensors_per_sec']:.1f}", f"{v['per_tensor_ms']:.2f}"]
            for k, v in speed.items()
        ],
        col_widths=[14, 14, 12],
    )

    # Peak RAM
    ram_shape = real_tensors[0].shape if real_tensors else (50, 50, 50)
    ram = benchmark_ram(shape=ram_shape, num_hashes=128)

    subsection("2d • Memory - vs Standard Random Projection")

    from tensorized_minhash.benchmarks import benchmark_random_projection

    rp = benchmark_random_projection(real_tensors, num_hashes=128)

    if rp["feasible"]:
        table(
            ["Method", "RAM", "Tensors/sec", "ms/tensor"],
            [
                [
                    "Random Projection (standard)",
                    f"{rp['theoretical_mb']:.1f} MB",
                    f"{rp['tensors_per_sec']:.1f}",
                    f"{rp['per_tensor_ms']:.2f}",
                ],
                [
                    "Kronecker (tensor decomp.)",
                    f"{ram['kron_peak_mb']:.2f} MB",
                    f"{speed['Kron']['tensors_per_sec']:.1f}",
                    f"{speed['Kron']['per_tensor_ms']:.2f}",
                ],
            ],
            col_widths=[30, 12, 14, 12],
        )
    else:
        print(f"  Random Projection: {rp['note']}")
        kv(
            "Kronecker:",
            f"{ram['kron_peak_mb']:.2f} MB ({ram['kron_vs_full']:.0f}x smaller than full matrix)",
        )
        table(
            ["Method", "RAM required", "Feasible?", "Params"],
            [
                [
                    "Full Random Projection",
                    f"{rp['theoretical_mb']:.0f} MB",
                    "No - too large",
                    f"{rp['flat_dim'] * 128:,}",
                ],
                [
                    "Tensor Train (TT)",
                    f"{ram['tt_peak_mb']:.2f} MB",
                    "Yes",
                    f"{ram['tt_params']:,}",
                ],
                [
                    "Kronecker decomposition",
                    f"{ram['kron_peak_mb']:.2f} MB",
                    "Yes",
                    f"{ram['kron_params']:,}",
                ],
            ],
            col_widths=[26, 14, 16, 18],
        )
    print(f"\n  Kron is {ram['kron_vs_tt']:.1f}x more compact than TT decomposition")


# --------------------------------------------------------------------------
# Deliverable 3: Local Scalability Prototype (PySpark or multiprocessing)
# --------------------------------------------------------------------------

def run_scalability_prototype(quick: bool = False, use_spark: bool = False):
    section("Local Scalability Prototype")

    from tensorized_minhash.core import KroneckerMinHash, TTMinHashConfig
    from tensorized_minhash.data.builder import NetworkTensorBuilder
    from tensorized_minhash.spark.pipeline import LocalTensorHashPipeline

    n_windows = 20 if quick else 60

    # Load and filter real data
    subsection("Dataset (time-windowed)")
    df_filtered, (n_src, n_dst, n_port) = load_and_filter(_DEFAULT_CSV)

    shape = (30, 30, 30) if quick else (n_src, n_dst, n_port)
    kv("Windows:", f"{n_windows}")
    kv("Window size:", f"{len(df_filtered) // n_windows:,} rows")
    kv("Tensor shape:", str(shape))
    print(f"\n  Building {n_windows} time-window tensors...")

    builder = NetworkTensorBuilder(n_src=shape[0], n_dst=shape[1], n_port=shape[2])
    tensors = builder.build_tensor_batch(df_filtered, window_size=len(df_filtered) // n_windows)
    tensors = tensors[:n_windows]
    ids = [f"w{i:03d}" for i in range(len(tensors))]

    cfg = TTMinHashConfig(shape=shape, num_hashes=128)
    hasher = KroneckerMinHash(cfg)

    if use_spark:
        try:
            import time

            from tensorized_minhash.spark.pipeline import SparkTensorHasher
            from tensorized_minhash.spark.session import _get_or_create_spark

            spark = _get_or_create_spark()
            pipe = SparkTensorHasher(hasher, spark=spark)
            sc = spark.sparkContext
            rdd = SparkTensorHasher.tensors_to_rdd(tensors, sc, ids)
            print("  Running Spark hash map...")
            t0 = time.perf_counter()
            sig_rdd = pipe.hash_rdd(rdd)
            similar_rdd = pipe.find_similar_pairs(sig_rdd, threshold=0.4)
            similar = similar_rdd.collect()
            elapsed = time.perf_counter() - t0
            kv("Spark:", f"{len(tensors)} tensors in {elapsed:.2f}s")
            spark.stop()
        except Exception as e:
            print(f"  Spark unavailable ({e}); falling back to multiprocessing")
            use_spark = False

    if not use_spark:
        import time

        pipe = LocalTensorHashPipeline(hasher)
        t0 = time.perf_counter()
        id_sigs = pipe.hash_all(tensors, ids, parallel=True)
        similar = pipe.find_similar_pairs(id_sigs, threshold=0.45)
        elapsed = time.perf_counter() - t0
        subsection("Multiprocessing hashing")
        kv("Tensors hashed:", f"{len(tensors)}")
        kv("Wall time:", f"{elapsed:.2f} s")
        kv("Throughput:", f"{len(tensors) / elapsed:.0f} tensors/sec")

    subsection("Top similar window pairs (potential repeated patterns)")
    if similar:
        table(
            ["Window A", "Window B", "Jaccard est."],
            [[a, b, f"{j:.4f}"] for a, b, j in similar[:10]],
            col_widths=[12, 12, 14],
        )
    else:
        print("  None above threshold - traffic is diverse")

    mem = hasher.memory_stats()
    subsection("Hash-function memory")
    kv(
        "Kron params:",
        f"{mem['kron_params']:,} bytes ({mem['kron_bytes'] / 1024:.1f} KB). for {cfg.num_hashes} ha",
    )
    kv("Full matrix equiv.:", f"{mem['full_bytes_theoretical'] / 1_000_000:.1f} MB")
    kv("Compression:", f"{mem['compression_ratio']:,}x")

    # Return tensors so D2 can reuse them
    return tensors


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tensorized MinHash pipeline")
    parser.add_argument("--quick", action="store_true", help="Smaller sizes for fast demo")
    parser.add_argument("--spark", action="store_true", help="Enable PySpark distributed mode")
    parser.add_argument("--only", choices=["1", "2", "3"], help="Run only one deliverable")
    args = parser.parse_args()

    import datetime

    mode = "quick" if args.quick else "full"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'=' * _W}")
    print(f" Tensorized MinHash - Kronecker & TT Compressed Hashing")
    print(f" CIC-IDS-2017 Wednesday  |  mode={mode}  |  {ts}")
    print(f"{'=' * _W}\n")

    real_tensors = None

    if args.only in (None, "1"):
        run_hashing_module_demo(quick=args.quick)

    if args.only in (None, "3"):
        real_tensors = run_scalability_prototype(quick=args.quick, use_spark=args.spark)

    if args.only in (None, "2"):
        run_resource_profile(quick=args.quick, real_tensors=real_tensors)

    print(f"\n{'=' * _W}")
    print(" Done. python main.py --help for options.")
    print(f"{'=' * _W}\n")


if __name__ == "__main__":
    main()
