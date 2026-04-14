#!/usr/bin/env python3
"""
main.py — Tensorized MinHash: full pipeline runner

Produces all three deliverables:
  1. Tensor-Aware Hashing Module (runs and validates)
  2. Computational Resource Profile (memory + RAM + speed tables)
  3. Local Scalability Prototype (PySpark demo or multiprocessing fallback)

Usage:
    python main.py              # full run
    python main.py --quick      # reduced sizes, faster
    python main.py --spark      # enable PySpark (requires pyspark installed)
    python main.py --only 2     # run only deliverable 2
"""

import sys
import os
import argparse
import logging
import numpy as np

# Make local packages importable
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    bar = "═" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load and filter real data
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter(csv_path: str):
    """
    Load CIC-IDS2017 CSV, filter noise, return (df_filtered, shape).
    """
    from data.loader import NetworkTensorBuilder

    builder_temp = NetworkTensorBuilder(1, 1, 1)
    df = builder_temp.load_cic_ids2017(csv_path)

    n_src_raw  = df['src_ip_raw'].nunique()
    n_dst_raw  = df['dst_ip_raw'].nunique()
    n_port_raw = df['port'].nunique()
    print(f"\nOriginal Dataset:")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Unique src IPs: {n_src_raw}  |  dst IPs: {n_dst_raw}  |  ports: {n_port_raw}")

    port_counts = df['port'].value_counts()
    meaningful_ports = port_counts[port_counts > 10].index
    df_filtered = df[df['port'].isin(meaningful_ports)].copy()

    dst_counts = df_filtered['dst_ip_raw'].value_counts()
    meaningful_dsts = dst_counts[dst_counts > 100].index
    df_filtered = df_filtered[df_filtered['dst_ip_raw'].isin(meaningful_dsts)].copy()

    n_src  = df_filtered['src_ip_raw'].nunique()
    n_dst  = df_filtered['dst_ip_raw'].nunique()
    n_port = df_filtered['port'].nunique()

    print(f"Filtered rows : {len(df_filtered):,}")
    print(f"Unique src IPs: {n_src}  |  dst IPs: {n_dst}  |  ports: {n_port}")

    return df_filtered, (n_src, n_dst, n_port)


# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 1: Tensor-Aware Hashing Module
# ─────────────────────────────────────────────────────────────────────────────

def run_hashing_module_demo(quick: bool = False):
    section("DELIVERABLE 1 — Tensor-Aware Hashing Module")

    from core.tt_minhash import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig
    from data.loader import NetworkTensorBuilder
    from benchmarks.benchmark import ground_truth_jaccard, datasketch_jaccard

    # Load and filter real data
    df_filtered, (n_src, n_dst, n_port) = load_and_filter("data/wed_data.csv")

    shape = (30, 30, 30) if quick else (n_src, n_dst, n_port)
    print(f"Tensor shape  : {shape}")

    builder = NetworkTensorBuilder(n_src=shape[0], n_dst=shape[1], n_port=shape[2])

    # Full tensor
    tensor = builder.build_tensor(df_filtered)
    print(f"\nFull tensor:")
    print(f"  Non-zero cells : {int(tensor.sum()):,}  /  {tensor.size:,}")
    print(f"  Density        : {tensor.mean():.4f}")

    # Meaningful split: common ports vs unusual ports
    common_ports = [80, 443, 22, 53, 389, 88]
    df_common  = df_filtered[df_filtered['port'].isin(common_ports)]
    df_unusual = df_filtered[~df_filtered['port'].isin(common_ports)]

    # Fall back to chronological split if one side is empty
    if len(df_common) == 0 or len(df_unusual) == 0:
        print("\nPort-based split empty — falling back to chronological split")
        half = len(df_filtered) // 2
        df_common  = df_filtered.iloc[:half]
        df_unusual = df_filtered.iloc[half:]

    tensor_common  = builder.build_tensor(df_common)
    tensor_unusual = builder.build_tensor(df_unusual)

    print(f"\nCommon-port tensor  nonzeros: {int(tensor_common.sum()):,}")
    print(f"Unusual-port tensor nonzeros: {int(tensor_unusual.sum()):,}")

    # Hash both with Kron and TT
    cfg  = TTMinHashConfig(shape=shape, num_hashes=128, seed=42)
    kron = KroneckerMinHash(cfg)
    tt   = TTDecomposedMinHash(cfg)

    sig_kron_c = kron.hash_tensor(tensor_common)
    sig_kron_u = kron.hash_tensor(tensor_unusual)
    sig_tt_c   = tt.hash_tensor(tensor_common)
    sig_tt_u   = tt.hash_tensor(tensor_unusual)

    j_kron  = kron.jaccard_from_signatures(sig_kron_c, sig_kron_u)
    j_tt    = tt.jaccard_from_signatures(sig_tt_c, sig_tt_u)
    j_exact = ground_truth_jaccard(tensor_common, tensor_unusual)
    j_ds    = datasketch_jaccard(tensor_common, tensor_unusual, num_perm=128)

    print("\nJaccard similarity (common ports vs unusual ports):")
    table(
        ["Method", "Jaccard", "Error vs exact"],
        [
            ["Exact (ground truth)", f"{j_exact:.4f}", "—"],
            ["Datasketch baseline",  f"{j_ds:.4f}",    f"{abs(j_ds-j_exact):.4f}"],
            ["Tensorized Kron",      f"{j_kron:.4f}",  f"{abs(j_kron-j_exact):.4f}"],
            ["Tensor Train (TT)",    f"{j_tt:.4f}",    f"{abs(j_tt-j_exact):.4f}"],
        ],
        col_widths=[24, 10, 20],
    )

    # Memory summary
    mem = kron.memory_stats()
    print(f"\nKron parameters : {mem['kron_params']:,}")
    print(f"Full parameters : {mem['full_params_theoretical']:,}")
    print(f"Compression     : {mem['compression_ratio']:,}×")


# ─────────────────────────────────────────────────────────────────────────────
# Computational Resource Profile
# ─────────────────────────────────────────────────────────────────────────────

def run_resource_profile(quick: bool = False, real_tensors: list = None):
    section("Computational Resource Profile")

    from benchmarks.benchmark import (
        benchmark_memory,
        benchmark_accuracy, benchmark_accuracy_from_tensors,
        benchmark_speed, benchmark_speed_real,
        benchmark_ram,
    )

    # Parameter count across tensor sizes
    print("\n Parameter count: Kronecker vs Full matrix \n")
    shapes_3d = [(10,10,10), (30,30,30), (50,50,50), (100,100,100)]
    shapes_4d = [(10,10,10,10), (20,20,20,20)]
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
                f"{r['compression_ratio']:,}×",
            ]
            for r in mem_results
        ],
        col_widths=[22, 14, 14, 18, 14],
    )

    # Accuracy
    print("\n Jaccard approximation accuracy \n")

    if real_tensors is not None and len(real_tensors) >= 2:
        # Use real tensors 
        #print("  Using real time-window tensors from Deliverable 3\n")
        n_pairs = 50 if quick else 200
        acc = benchmark_accuracy_from_tensors(
            tensors=real_tensors,
            shape=real_tensors[0].shape,
            num_hashes=128,
            n_pairs=n_pairs,
        )
        #print(f"  Windows : {acc['n_windows']}  |  "
        #      f"Pairs   : {acc['n_pairs']}  |  "
        #      f"Jaccard range: [{acc['jaccard_range'][0]:.3f}, "
        #      f"{acc['jaccard_range'][1]:.3f}]")
    else:
        # Fall back to synthetic data
        print("  No real tensors provided — using synthetic data\n")
        n_pairs = 50 if quick else 200
        acc = benchmark_accuracy(n_pairs=n_pairs, shape=(30,30,30), num_hashes=128)

    table(
        ["Method", "MAE", "RMSE", "Pearson r"],
        [
            ["Tensorized Kron",
             f"{acc['tensorized_kron']['mae']:.4f}",
             f"{acc['tensorized_kron']['rmse']:.4f}",
             f"{acc['tensorized_kron']['pearson_r']:.4f}"],
            ["Datasketch baseline",
             f"{acc['datasketch_baseline']['mae']:.4f}",
             f"{acc['datasketch_baseline']['rmse']:.4f}",
             f"{acc['datasketch_baseline']['pearson_r']:.4f}"],
        ],
        col_widths=[22, 10, 10, 12],
    )

    # Speed
    print("\n Hashing throughput\n")

    if real_tensors is not None and len(real_tensors) >= 2:
        #print("  Using real time-window tensors from Deliverable 3\n")
        n_t = 20 if quick else len(real_tensors)
        speed = benchmark_speed_real(
            tensors=real_tensors[:n_t],
            num_hashes=128,
        )
    else:
        n_t = 100 if quick else 500
        speed = benchmark_speed(shape=(30,30,30), num_hashes=128, n_tensors=n_t)

    table(
        ["Method", "Tensors/sec", "ms/tensor"],
        [
            [k,
             f"{v['tensors_per_sec']:.1f}",
             f"{v['per_tensor_ms']:.2f}"]
            for k, v in speed.items()
        ],
        col_widths=[14, 14, 12],
    )

    # Peak RAM
    #print("\nPeak RAM allocation\n")
    ram_shape = real_tensors[0].shape if real_tensors else (50, 50, 50)
    ram = benchmark_ram(shape=ram_shape, num_hashes=128)
    

    #table(
    #["Method", "Peak RAM", "Params", "vs Full matrix"],
    #[
    #    ["Full matrix (standard vectoriz.)",
    #     f"{ram['full_theoretical_mb']:.2f} MB",
    #     f"{ram['full_params']:,}",
    #     "1×"],
    #    ["Tensor Train (TT) decomposition",
    #     f"{ram['tt_peak_mb']:.2f} MB",
    #     f"{ram['tt_params']:,}",
    #     f"{ram['tt_vs_full']:.0f}×"],
    #    ["Kronecker decomposition (Kron)",
    #     f"{ram['kron_peak_mb']:.2f} MB",
    #     f"{ram['kron_params']:,}",
    #     f"{ram['kron_vs_full']:.0f}×"],
    #],
    #col_widths=[35, 12, 16, 16],
    #)
    #print(f"\n  Kron is {ram['kron_vs_tt']:.1f}× more compact than TT decomposition")
    #table(
    #    ["", "Value"],
    #    [
    #        ["Kron peak RAM",              f"{ram['kron_peak_mb']:.2f} MB"],
    #        ["Datasketch peak RAM",        f"{ram['datasketch_peak_mb']:.2f} MB"],
    #        ["Full matrix (theoretical)",  f"{ram['full_theoretical_mb']:.2f} MB"],
    #        ["Kron RAM savings (vs full)", f"{ram['ram_compression']:.1f}×"],
    #        ["Kron vs Datasketch",         f"{ram['kron_vs_datasketch']:.1f}×"],
    #    ],
    #    col_widths=[30, 20],
    #)

    print("\nStandard vectorization vs Tensor Decomposition\n")

    from benchmarks.benchmark import benchmark_random_projection
    rp = benchmark_random_projection(real_tensors, num_hashes=128)

    if rp["feasible"]:
        table(
            ["Method", "RAM", "Tensors/sec", "ms/tensor"],
            [
            ["Random Projection (standard)",
             f"{rp['theoretical_mb']:.1f} MB",
             f"{rp['tensors_per_sec']:.1f}",
             f"{rp['per_tensor_ms']:.2f}"],
            ["Kronecker (tensor decomp.)",
             f"{ram['kron_peak_mb']:.2f} MB",
             f"{speed['Kron']['tensors_per_sec']:.1f}",
             f"{speed['Kron']['per_tensor_ms']:.2f}"],
            ],
        col_widths=[30, 12, 14, 12],
            )
    else:
        print(f"  Random Projection: {rp['note']}")
        print(f"  Kronecker:         {ram['kron_peak_mb']:.2f} MB  "
          f"({ram['kron_vs_full']:.0f}× smaller)\n")
        table(
        ["Method", "RAM required", "Feasible?", "Params"],
        [
            ["Full Random Projection",
             f"{rp['theoretical_mb']:.0f} MB",
             "No — too large",
             f"{rp['flat_dim']*128:,}"],
            ["Tensor Train (TT)",
             f"{ram['tt_peak_mb']:.2f} MB",
             "Yes",
             f"{ram['tt_params']:,}"],
            ["Kronecker decomposition",
             f"{ram['kron_peak_mb']:.2f} MB",
             "Yes",
             f"{ram['kron_params']:,}"],
        ],
        col_widths=[26, 14, 16, 18],
            )
    print(f"\n  Kron is {ram['kron_vs_tt']:.1f}× more compact than TT decomposition")
# ─────────────────────────────────────────────────────────────────────────────
# Local Scalability Prototype (PySpark or multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability_prototype(quick: bool = False, use_spark: bool = False):
    section("Local Scalability Prototype")

    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig
    from data.loader import NetworkTensorBuilder
    from spark.distributed_hasher import LocalTensorHashPipeline

    n_windows = 20 if quick else 60

    # Load and filter real data
    df_filtered, (n_src, n_dst, n_port) = load_and_filter("data/wed_data.csv")

    print(f"Window size: {len(df_filtered) // n_windows:,} rows per window")

    shape = (30, 30, 30) if quick else (n_src, n_dst, n_port)
    print(f"Tensor shape: {shape}")
    print(f"Building {n_windows} time-window tensors…")

    builder = NetworkTensorBuilder(n_src=shape[0], n_dst=shape[1], n_port=shape[2])
    tensors = builder.build_tensor_batch(
        df_filtered, window_size=len(df_filtered) // n_windows
    )
    tensors = tensors[:n_windows]
    ids = [f"w{i:03d}" for i in range(len(tensors))]

    cfg    = TTMinHashConfig(shape=shape, num_hashes=128)
    hasher = KroneckerMinHash(cfg)

    if use_spark:
        try:
            from spark.distributed_hasher import SparkTensorHasher, _get_or_create_spark
            import time
            spark = _get_or_create_spark()
            pipe  = SparkTensorHasher(hasher, spark=spark)
            sc    = spark.sparkContext
            rdd   = SparkTensorHasher.tensors_to_rdd(tensors, sc, ids)
            print("Running Spark hash map…")
            t0 = time.perf_counter()
            sig_rdd     = pipe.hash_rdd(rdd)
            similar_rdd = pipe.find_similar_pairs(sig_rdd, threshold=0.4)
            similar     = similar_rdd.collect()
            elapsed     = time.perf_counter() - t0
            print(f"Spark: {len(tensors)} tensors in {elapsed:.2f}s")
        except Exception as e:
            print(f"Spark unavailable ({e}); falling back to multiprocessing")
            use_spark = False

    if not use_spark:
        import time
        pipe    = LocalTensorHashPipeline(hasher)
        t0      = time.perf_counter()
        id_sigs = pipe.hash_all(tensors, ids, parallel=True)
        similar = pipe.find_similar_pairs(id_sigs, threshold=0.45)
        elapsed = time.perf_counter() - t0
        print(f"Multiprocessing: {len(tensors)} tensors in {elapsed:.2f}s  "
              f"({len(tensors)/elapsed:.0f} tensors/sec)")

    print(f"\nTop similar window pairs (potential repeated patterns):")
    if similar:
        table(
            ["Window A", "Window B", "Jaccard est."],
            [(a, b, f"{j:.4f}") for a, b, j in similar[:10]],
            col_widths=[12, 12, 14],
        )
    else:
        print("  None above threshold — traffic is diverse")

    mem = hasher.memory_stats()
    print(f"\nHash-function memory : {mem['kron_bytes']:,} bytes "
          f"({mem['kron_bytes']/1024:.1f} KB) for {cfg.num_hashes} hash planes")
    print(f"Equivalent full matrix: {mem['full_bytes_theoretical']/1_000_000:.1f} MB")
    print(f"Compression           : {mem['compression_ratio']:,}×")

    # Return tensors so D2 can reuse them
    return tensors


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tensorized MinHash pipeline")
    parser.add_argument("--quick", action="store_true", help="Smaller sizes for fast demo")
    parser.add_argument("--spark", action="store_true", help="Enable PySpark distributed mode")
    parser.add_argument("--only",  choices=["1", "2", "3"], help="Run only one deliverable")
    args = parser.parse_args()


    print("  Tensorized MinHash — Kronecker Compressed Hashing")
    print("  Jaccard Similarity on Multi-Dimensional Network Tensors")

    real_tensors = None

    if args.only in (None, "3"):
        real_tensors = run_scalability_prototype(
            quick=args.quick, use_spark=args.spark
        )

    if args.only in (None, "1"):
        run_hashing_module_demo(quick=args.quick)

    if args.only in (None, "2"):
        run_resource_profile(quick=args.quick, real_tensors=real_tensors)


if __name__ == "__main__":
    main()
