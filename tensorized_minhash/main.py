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
# Deliverable 1: Tensor-Aware Hashing Module
# ─────────────────────────────────────────────────────────────────────────────

def run_hashing_module_demo(quick: bool = False):
    section("DELIVERABLE 1 — Tensor-Aware Hashing Module")

    from core.tt_minhash import KroneckerMinHash, TTDecomposedMinHash, TTMinHashConfig
    from data.loader import NetworkLogGenerator, NetworkTensorBuilder

    n = 30 if quick else 50
    shape = (n, n, n)

    # Generate data
    #gen = NetworkLogGenerator(n_src=n, n_dst=n, n_port=n, n_benign=2000, seed=7)
    #df, attack_groups = gen.generate()

    builder = NetworkTensorBuilder(n_src=n, n_dst=n, n_port=n)
    df = builder.load_cic_ids2017("data/friday_data.csv")
    df["label"] = "unknown"   # no label column in your file
    attack_groups = {}         # skip attack similarity matrix
    #builder = NetworkTensorBuilder(n_src=n, n_dst=n, n_port=n)
    tensor = builder.build_tensor(df)

    print(f"\nTensor shape : {tensor.shape}")
    print(f"Non-zero cells: {int(tensor.sum()):,}  /  {tensor.size:,}")
    print(f"Density       : {tensor.mean():.4f}")

    # Build two tensors: one with attack, one benign-only
    #df_benign = df[df["label"] == "benign"]
    #df_attack = df[df["label"] != "benign"]
    
    half = len(df) // 2
    df_benign = df.iloc[:half]
    df_attack = df.iloc[half:]
    tensor_benign = builder.build_tensor(df_benign)
    tensor_attack = builder.build_tensor(df_attack)

    # Hash both
    cfg = TTMinHashConfig(shape=shape, num_hashes=128, seed=42)
    kron = KroneckerMinHash(cfg)
    tt   = TTDecomposedMinHash(cfg)

    sig_kron_b = kron.hash_tensor(tensor_benign)
    sig_kron_a = kron.hash_tensor(tensor_attack)
    sig_tt_b   = tt.hash_tensor(tensor_benign)
    sig_tt_a   = tt.hash_tensor(tensor_attack)

    j_kron = kron.jaccard_from_signatures(sig_kron_b, sig_kron_a)
    j_tt   = tt.jaccard_from_signatures(sig_tt_b, sig_tt_a)

    from benchmarks.benchmark import ground_truth_jaccard, datasketch_jaccard
    j_exact = ground_truth_jaccard(tensor_benign, tensor_attack)
    j_ds    = datasketch_jaccard(tensor_benign, tensor_attack, num_perm=128)

    print("\nJaccard similarity (benign vs attack traffic tensors):")
    table(
        ["Method", "Jaccard", "Error vs exact"],
        [
            ["Exact (ground truth)",  f"{j_exact:.4f}", "—"],
            ["Datasketch baseline",   f"{j_ds:.4f}",    f"{abs(j_ds-j_exact):.4f}"],
            ["Tensorized Kron",       f"{j_kron:.4f}",  f"{abs(j_kron-j_exact):.4f}"],
            ["Tensor Train (TT)",     f"{j_tt:.4f}",    f"{abs(j_tt-j_exact):.4f}"],
        ],
        col_widths=[24, 10, 20],
    )

    # Show attack group detection
    print("\nAttack pattern similarity matrix (Kron MinHash):")
    labels = list(attack_groups.keys())
    attack_tensors = {}
    for lbl in labels:
        at = np.zeros(shape, dtype=np.float32)
        for (s, d, p) in attack_groups[lbl]:
            at[s % n, d % n, p % n] = 1.0
        attack_tensors[lbl] = at

    sigs = {lbl: kron.hash_tensor(at) for lbl, at in attack_tensors.items()}

    header = [""] + labels
    rows = []
    for a in labels:
        row = [a]
        for b in labels:
            if a == b:
                row.append("1.0000")
            else:
                j = kron.jaccard_from_signatures(sigs[a], sigs[b])
                row.append(f"{j:.4f}")
        rows.append(row)
    table(header, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 2: Computational Resource Profile
# ─────────────────────────────────────────────────────────────────────────────

def run_resource_profile(quick: bool = False):
    section("DELIVERABLE 2 — Computational Resource Profile")

    from benchmarks.benchmark import (
        benchmark_memory, benchmark_accuracy, benchmark_speed, benchmark_ram
    )

    # 2a: Memory footprint across tensor orders / sizes
    print("\n── 2a. Parameter count: Kronecker vs full matrix ──\n")
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

    # 2b: Accuracy vs datasketch baseline
    print("\n── 2b. Jaccard approximation accuracy ──\n")
    n = 50 if quick else 200
    shape = (30, 30, 30)
    acc = benchmark_accuracy(n_pairs=n, shape=shape, num_hashes=128)

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

    # 2c: Speed comparison
    print("\n── 2c. Hashing throughput ──\n")
    n_t = 100 if quick else 500
    speed = benchmark_speed(shape=(30, 30, 30), num_hashes=128, n_tensors=n_t)

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

    # 2d: Peak RAM
    print("\n── 2d. Peak RAM allocation ──\n")
    ram = benchmark_ram(shape=(50, 50, 50), num_hashes=128)
    table(
        ["", "Value"],
        [
            ["Kron peak RAM",              f"{ram['kron_peak_mb']:.2f} MB"],
            ["Full matrix (theoretical)",  f"{ram['full_theoretical_mb']:.2f} MB"],
            ["RAM savings",                f"{ram['ram_compression']:.1f}×"],
        ],
        col_widths=[30, 20],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deliverable 3: Local Scalability Prototype (PySpark or multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability_prototype(quick: bool = False, use_spark: bool = False):
    section("DELIVERABLE 3 — Local Scalability Prototype")

    from core.tt_minhash import KroneckerMinHash, TTMinHashConfig
    from data.loader import NetworkLogGenerator, NetworkTensorBuilder
    from spark.distributed_hasher import LocalTensorHashPipeline

    n = 50 if quick else 100
    n_windows = 20 if quick else 60
    shape = (n, n, n)

    print(f"\nGenerating {n_windows} time-window tensors  (shape={shape})…")
    #gen = NetworkLogGenerator(n_src=n, n_dst=n, n_port=n, n_benign=5000, seed=2024)
    #df, _ = gen.generate()
    builder_temp = NetworkTensorBuilder(n_src=n, n_dst=n, n_port=n)
    df = builder_temp.load_cic_ids2017("data/friday_data.csv")
    builder = NetworkTensorBuilder(n_src=n, n_dst=n, n_port=n)
    tensors = builder.build_tensor_batch(df, window_size=len(df) // n_windows)
    tensors = tensors[:n_windows]
    ids = [f"w{i:03d}" for i in range(len(tensors))]

    cfg = TTMinHashConfig(shape=shape, num_hashes=128)
    hasher = KroneckerMinHash(cfg)

    if use_spark:
        try:
            from spark.distributed_hasher import SparkTensorHasher, _get_or_create_spark
            spark = _get_or_create_spark()
            pipe = SparkTensorHasher(hasher, spark=spark)
            sc = spark.sparkContext
            rdd = SparkTensorHasher.tensors_to_rdd(tensors, sc, ids)
            print("Running Spark hash map…")
            import time; t0 = time.perf_counter()
            sig_rdd = pipe.hash_rdd(rdd)
            similar_rdd = pipe.find_similar_pairs(sig_rdd, threshold=0.4)
            similar = similar_rdd.collect()
            elapsed = time.perf_counter() - t0
            print(f"Spark: {len(tensors)} tensors in {elapsed:.2f}s")
        except Exception as e:
            print(f"Spark unavailable ({e}); falling back to multiprocessing")
            use_spark = False

    if not use_spark:
        import time
        pipe = LocalTensorHashPipeline(hasher)
        t0 = time.perf_counter()
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
        print("  None above threshold — traffic is diverse (expected for random data)")

    mem = hasher.memory_stats()
    print(f"\nHash-function memory: {mem['kron_bytes']:,} bytes "
          f"({mem['kron_bytes']/1024:.1f} KB) for {cfg.num_hashes} hash planes")
    print(f"Equivalent full matrix: {mem['full_bytes_theoretical']/1_000_000:.1f} MB")
    print(f"Compression: {mem['compression_ratio']:,}×")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tensorized MinHash pipeline")
    parser.add_argument("--quick",  action="store_true", help="Smaller sizes for fast demo")
    parser.add_argument("--spark",  action="store_true", help="Enable PySpark distributed mode")
    parser.add_argument("--only",   choices=["1", "2", "3"], help="Run only one deliverable")
    args = parser.parse_args()

    print("\n" + "█"*64)
    print("  Tensorized MinHash — TT/Kronecker Compressed Hashing")
    print("  Jaccard Similarity on Multi-Dimensional Network Tensors")
    print("█"*64)

    if args.only != "2" and args.only != "3":
        run_hashing_module_demo(quick=args.quick)

    if args.only != "1" and args.only != "3":
        run_resource_profile(quick=args.quick)

    if args.only != "1" and args.only != "2":
        run_scalability_prototype(quick=args.quick, use_spark=args.spark)

    print("\n✓ All deliverables complete.\n")


if __name__ == "__main__":
    main()
