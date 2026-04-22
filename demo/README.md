# TensorizedMinHash — Demos

Two self-contained demos built on top of the core engine in `tensorized_minhash/`.
Each demo exercises the same `KroneckerMinHash` and `TTDecomposedMinHash` classes
used in the main benchmarks, so running them is the best way to verify the engine
end-to-end before deployment.

---

## 1. Genome Demo — Human vs Animals

Compares human mitochondrial DNA against six other species using all three
algorithms (Kronecker, TT, Datasketch) and exact Jaccard ground truth.

Demonstrates that Kronecker and TT MinHash produce accurate similarity estimates
on real biological sequence data — and that the similarity gradient matches
evolutionary distance (great apes closest, yeast furthest).

**No Docker required.**

### Setup

All FASTA files are downloaded automatically from NCBI on the first run.
To pre-download manually:

```bash
cd demo/genome/data/

curl -o human.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_012920.1&rettype=fasta&retmode=text"
curl -o chimp.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_011120.1&rettype=fasta&retmode=text"
curl -o gorilla.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_011120.1&rettype=fasta&retmode=text"
curl -o orangutan.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_002083.1&rettype=fasta&retmode=text"
curl -o mouse.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_005089.1&rettype=fasta&retmode=text"
curl -o chicken.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001323.1&rettype=fasta&retmode=text"
curl -o zebrafish.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_002333.2&rettype=fasta&retmode=text"
```

### Run

```bash
# From repo root
python demo/genome/similarity_report.py
```

### Expected output (similarity decreases down the table)

---

```
Pair                Exact     Kron       TT  Datasketch
Human <-> Chimp     0.256    0.309    0.254       0.250
Human <-> Gorilla   0.204    0.203    0.207       0.195
Human <-> Orangutan 0.137    0.172    0.141       0.152
Human <-> Mouse     0.059    0.059    0.059       0.059
Human <-> Chicken   0.047    0.066    0.020       0.031
Human <-> Zebrafish 0.046    0.055    0.039       0.035

Estimation accuracy (vs exact Jaccard)
Kronecker MinHash   MAE 0.019  Max error 0.052
Tensor Train (TT)   MAE 0.007  Max error 0.027
Datasketch          MAE 0.010  Max error 0.016
```

---

## 2. Racing Track — Docker Compose

Three algorithm workers race to estimate all pairwise similarities from a shared
input. A Streamlit scoreboard reads live CPU, RAM, and progress from Docker in
real time — no API ports needed inside containers.

### Prerequisites

- Docker Desktop running (Linux containers)
- Genome FASTA files present in `demo/genome/data/` (step 1 above, or auto-downloaded)

### Event 1 — Three-way race: Kron vs TT vs Datasketch

```bash
cd demo/racing

# Build images (first time, or after any code change)
docker compose -f docker-compose.event1.yml build

# Start the race (default load level 7)
LOAD_LEVEL=7 docker compose -f docker-compose.event1.yml up -d

# Open the live scoreboard
open http://localhost:8501      # Mac/Linux
start http://localhost:8501     # Windows

# Tear down when done (scores/ folder on host is preserved)
docker compose -f docker-compose.event1.yml down
```

### Event 2 — Head-to-head: Kron vs TT at high load (512 hashes)

```bash
LOAD_LEVEL=9 docker compose -f docker-compose.event2.yml up -d
open http://localhost:8501
docker compose -f docker-compose.event2.yml down
```

### LOAD_LEVEL reference

| Level | Tensor shape | Pairs | Approx Datasketch time |
| ----- | ------------ | ----- | ---------------------- |
| 1     | 15³          | 30    | ~2 s                   |
| 5     | 50³          | 150   | ~28 s                  |
| 7     | 70³          | 250   | ~65 s                  |
| 9     | 90³          | 400   | ~150 s                 |
| 10    | 100³         | 500   | ~240 s                 |

### Smoke-testing workers without Docker

```bash
# From repo root — set the env vars Docker would normally inject
$env:SHARED_INPUT_DIR = "demo/racing/shared_input"
$env:SCORES_DIR       = "demo/racing/scores"
$env:NUM_HASHES       = "64"
$env:SEED             = "42"

python demo/racing/worker_kron/worker.py
python demo/racing/worker_tt/worker.py
python demo/racing/worker_datasketch/worker.py
```

### Volumes

| Volume         | Purpose                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| `shared_input` | Tensor pairs generated once before the race starts                                   |
| `scores`       | JSON metrics written by each worker; also persisted to `demo/racing/scores/` on host |
