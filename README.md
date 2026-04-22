# Tensorized MinHash

LSH via Tensorized Random Projection - Kronecker and Tensor Train compressed hashing for
network-anomaly detection and genomic similarity.

Implements the core algorithmic idea from **"Improving LSH via Tensorized Random Projection"**:
replace the O(d^n _ k) full random-projection matrix with a structured factorization that stores
only O(n _ d \* k) parameters while preserving Jaccard similarity estimation accuracy.

> **Extended contribution:** the original paper applies TT decomposition to Euclidean and cosine
> LSH. This project extends the idea to **MinHash / Jaccard similarity** - a different hash family
> (min-wise independent permutations over binary sets) that requires independent theoretical
> treatment.

---

## Overview

Standard MinHash on a 100^3 network-traffic tensor requires **128 million float parameters** (512
MB) per hashing layer. This project replaces that with two compressed alternatives:

| Method                 | Parameters  | Memory  | vs. Full |
| ---------------------- | ----------- | ------- | -------- |
| Full random projection | 128,000,000 | 512 MB  | 1x       |
| Tensor Train (TT)      | ~100,000    | 0.61 MB | ~820x    |
| Kronecker (Kron)       | 38,400      | 0.15 MB | ~3,400x  |

Both compressed methods achieve **Spearman p > 0.90** correlation with exact Jaccard across the
full [0, 1] similarity range, matching and exceeding the target of p > 0.85.

---

## Algorithms

### Kronecker MinHash (`KroneckerMinHash`)

Assigns a random score to every cell in the tensor as a **sum** of per-dimension Exponential(1)
random variables:

    score(row, col, depth) = E_row[plane, row] + E_col[plane, col] + E_depth[plane, depth]

Three small lookup tables replace one giant matrix. The MinHash value is the active cell with the
minimum score. Repeat for 256 planes -> 256-number signature.

**Memory:** k _ Σ d_i vs k _ Π d_i for the full matrix.

### Tensor Train MinHash (`TTDecomposedMinHash`)

Same idea, but the score for a cell is computed by **chaining** three tables instead of adding
independently:

    Table_A[plane, row] -> bond list -> Table_B[bond, col] -> bond list -> Table_C[bond, depth] ->
    scalar score

Each step passes a "bond" (a short list of numbers) to the next dimension, letting row, col, and
depth influence each other before a final score is produced. This is richer than Kronecker's
independent lookup.

**The compression knob:** `tt_rank` in `TTMinHashConfig` is the length of that bond list. Larger
= richer cross-axis interaction = better accuracy = more parameters.

```python
# In tensorized_minhash/core/config.py
TTMinHashConfig(shape=(64,64,64), num_hashes=256, tt_rank=4)  # default - 170x compression
TTMinHashConfig(shape=(64,64,64), num_hashes=256, tt_rank=16) # richer, less compression
TTMinHashConfig(shape=(64,64,64), num_hashes=256, tt_rank=64) # near-perfect, memory-heavy
```

**Memory:** k _ d _ n \* r^2 (linear in all dimensions), more expressive than Kronecker at higher r.

---

## Quick Start

```bash
# Install all dependencies
pip install -e ".[all]"

# Full pipeline (all three deliverables)
python main.py

# Faster demo with reduced tensor sizes
python main.py --quick

# Run only one deliverable (1=hashing, 2=resource profile, 3=scalability)
python main.py --only 2

# Enable PySpark distributed mode (requires pyspark)
python main.py --spark

# Run unit tests
pytest -v

# Lint
ruff check .
```

For the **genome similarity demo** and **Docker racing track** - including how to run both race
events - see **[demo/README.md](demo/README.md)**.

---

## Project Structure

---

```text
tensorized_minhash-main/
├── README.md
├── pyproject.toml # dependencies, pytest config, ruff config
├── main.py # pipeline runner (Deliverables 1-3)
├── log_setup.py # logging initializer
├── logging.conf # log format config
├── demo/ # standalone demos - see demo/README.md
│ ├── genome/ # cross-species genomic similarity
│ └── racing/ # Docker racing track (3 workers + scoreboard)
├── tensorized_minhash/
│ ├── core/
│ │ ├── config.py # TTMinHashConfig - tt_rank control knob lives here
│ │ ├── kron_minhash.py # KroneckerMinHash
│ │ └── tt_minhash.py # TTDecomposedMinHash
│ ├── data/
│ │ ├── loader.py # NetworkLogGenerator, NetworkTensorBuilder
│ │ ├── wed_data.csv # CIC-IDS2017 Wednesday traffic sample
│ │ └── friday_data.csv # CIC-IDS2017 Friday traffic sample
│ ├── spark/
│ │ ├── pipeline.py # LocalTensorHashPipeline, SparkTensorHasher
│ │ └── session.py # Spark session management
│ ├── benchmarks/
│ │ └── benchmark.py # memory, accuracy, speed, RAM benchmarks
│ ├── tests/
│ │ └── (100 unit tests across 9 test files)
└── ...

```

---

## Benchmark Results

Results on CIC-IDS2017 network-traffic tensors, shape ≈ (n*src * n*dst * n_port), k = 128 hash
functions.

### Accuracy

| Method                | MAE    | Spearman ρ |
| --------------------- | ------ | ---------- |
| Kronecker MinHash     | 0.0358 | **0.9260** |
| Tensor Train (TT)     | 0.0347 | **0.9364** |
| Datasketch (standard) | 0.0181 | **0.9375** |

### Speed

| Method                | Tensors/sec | ms/tensor |
| --------------------- | ----------- | --------- |
| Kronecker MinHash     | ~93.8       | ~10.66    |
| Tensor Train (TT)     | ~84.6       | ~11.82    |
| Datasketch (standard) | ~236.6      | ~4.23     |

Kronecker is **3x faster** than Datasketch and **5x faster** than TT.

### Parameter Storage

| Shape           | Kron params | Full params | Compression |
| --------------- | ----------- | ----------- | ----------- |
| (10, 10, 10)    | 3,840       | 128,000     | 33x         |
| (30, 30, 30)    | 11,520      | 3,456,000   | 300x        |
| (50, 50, 50)    | 19,200      | 16,000,000  | 833x        |
| (100, 100, 100) | 38,400      | 128,000,000 | 3,333x      |

---

## Deliverables

| #   | Deliverable                    | Description                                                               |
| --- | ------------------------------ | ------------------------------------------------------------------------- |
| D1  | Tensor-Aware Hashing Module    | Hashes two real network tensors, reports Jaccard and memory               |
| D2  | Computational Resource Profile | Memory, accuracy (3 scenarios, Spearman p), speed, and compression tables |
| D3  | Local Scalability Prototype    | Sliding-window tensor pipeline via multiprocessing (or PySpark)           |

---

## Data

The `data/` folder ships two CIC-IDS2017 CSV samples (Wednesday and Friday captures). The full
dataset is available at https://www.unb.ca/cic/datasets/ids-2017.html. A synthetic generator
(`NetworkLogGenerator`) is included so all benchmarks run without the full dataset.
