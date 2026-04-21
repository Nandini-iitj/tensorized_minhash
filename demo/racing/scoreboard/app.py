"""
app.py - Racing Scoreboard (Streamlit + Docker SDK)

Reads everything from outside the containers:
 • CPU% / RAM  -> container.stats(stream=False)
 • Progress    -> container.logs(tail=20) - workers print "PROGRESS: x/total"
 • Start time  -> container.attrs['State']['StartedAt']
 • Metrics     -> reads JSON files from /scores/ volume

Run:
    streamlit run app.py --server.port 8501
"""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import time

import docker
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/scores"))

ALGO_COLORS = {
    "kron": " #4e79a7",   # blue
    "tt": " #f28e2b",   # orange
    "minhash": " #e15759",   # red
}

ALGO_LABELS = {
    "kron": "Kronecker",
    "tt": "Tensor Train",
    "minhash": "Minhash (dense)",
}

CONTAINER_NAMES = {
    "event1": ["racing_worker_kron", "racing_worker_tt", "racing_worker_minhash"],
    "event2": ["racing_worker_kron", "racing_worker_tt"]
}

# -----------------------------------------------------------------------------
# Docker helpers
# -----------------------------------------------------------------------------


@st.cache_resource
def get_docker_client():
    return docker.from_env()


def get_worker_containers(
        client: docker.DockerClient, event: str
        ) -> dict[str, docker.models.containers.Container]:
    """Return {algo: container} for the given compose event profile."""
    name_map = {
        "event1": ["racing_worker_kron", "racing_worker_tt", "racing_worker_datasketch"],
        "event2": ["racing_worker_kron", "racing_worker_tt"],
    }
    result = {}
    for cname in name_map.get(event, []):
        try:
            c = client.containers.get(cname)
            algo = cname.replace("racing_worker_", "")
            result[algo] = c
        except docker.errors.NotFound:
            pass
    return result


def container_stats(container) -> dict:
    """Return CPU % and RAM MB for a running container."""
    try:
        raw = container.stats(stream=False)
        cpu = _cpu_percent(raw)
        mem = raw["memory_stats"].get("usage", 0) / (1024 * 1024)
        return {"cpu_pct": round(cpu, 1), "ram_mb": round(mem, 1)}
    except Exception:
        return {"cpu_pct": 0.0, "ram_mb": 0.0}


def _cpu_percent(stats: dict) -> float:
    try:
        cd = stats["cpu_stats"]
        pd_ = stats["precpu_stats"]
        cpu_d = cd["cpu_usage"]["total_usage"] - pd_["cpu_usage"]["total_usage"]
        sys_d = cd["system_cpu_usage"] - pd_["system_cpu_usage"]
        n = cd.get("online_cpus") or len(cd["cpu_usage"].get("percpu_usage", [1]))
        return (cpu_d / sys_d) * n * 100.0 if sys_d > 0 else 0.0
    except (KeyError, ZeroDivisionError):
        return 0.0


def container_progress(container) -> str | None:
    """Extract last PROGRESS line from container logs."""
    try:
        logs = container.logs(tail=20).decode("utf-8", errors="replace")
        for line in reversed(logs.splitlines()):
            if "PROGRESS:" in line or "DONE:" in line:
                return line.strip()
    except Exception:
        pass
    return None


def container_elapsed(container) -> float:
    """Return elapsed seconds since container start."""
    try:
        started = container.attrs["State"]["StartedAt"]
        dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        return (datetime.now(UTC) - dt).total_seconds()
    except Exception:
        return 0.0


def container_status(container) -> str:
    try:
        container.reload()
        return container.status  # running | exited | created | ...
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Scores volume reader
# -----------------------------------------------------------------------------


def load_score_files() -> list[dict]:
    """Read all *_metrics.json files from the scores volume."""
    records = []
    if not SCORES_DIR.exists():
        return records
    for f in sorted(SCORES_DIR.glob("*_metrics.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
                d["_file"] = f.name
                records.append(d)
        except Exception:
            pass
    return records


# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------


def _init_history():
    if "history" not in st.session_state:
        st.session_state.history = {}  # algo -> list of {t, cpu, ram}


def _push_history(algo: str, cpu: float, ram: float):
    h = st.session_state.history.setdefault(algo, [])
    h.append({"t": time.time(), "cpu": cpu, "ram": ram})
    if len(h) > MAX_HISTORY:
        h.pop(0)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="MinHash Racing Track",
        page_icon="🏎️",
        layout="wide",
    )
    _init_history()

    client = get_docker_client()

    # --- Sidebar controls
    with st.sidebar:
        st.title("🏎️ Racing Controls")

        event = st.selectbox(
            "Event",
            ["event1", "event2"],
            format_func=lambda e: {
                "event1": "Event 1 - Kron vs TT vs Datasketch",
                "event2": "Event 2 - Kron vs TT (higher load)",
            }[e],
        )

        load_level = st.slider(
            "LOAD_LEVEL",
            min_value=1,
            max_value=10,
            value=7,
            help="Higher = bigger tensor shape + more pairs",
        )

        level_info = {
            1: "shape 15³ • 30 pairs",
            2: "shape 20³ • 50 pairs",
            3: "shape 28³ • 70 pairs",
            4: "shape 35³ • 100 pairs",
            5: "shape 50³ • 150 pairs",
            6: "shape 60³ • 200 pairs",
            7: "shape 70³ • 250 pairs",
            8: "shape 80³ • 320 pairs",
            9: "shape 90³ • 400 pairs",
            10: "shape 100³ • 500 pairs",
        }
        st.caption(f"ℹ️ {level_info[load_level]}")

        st.divider()
        st.markdown("**Input generation**")
        if st.button("🪄 Generate Input", use_container_width=True):
            with st.spinner("Generating tensor pairs..."):
                import subprocess
                import sys

                gen = Path(__file__).parent.parent / "shared_input" / "generate_input.py"
                subprocess.run(
                    [
                        sys.executable,
                        str(gen),
                        "--load-level",
                        str(load_level),
                        "--out-dir",
                        str(Path(__file__).parent.parent / "shared_input"),
                    ],
                    check=True,
                )
            st.success("Input ready - restart containers to race!")

        st.divider()
        refresh = st.toggle("Auto-refresh", value=True)

    # Auto-refresh
    if refresh:
        st.html(f"""<meta http-equiv="refresh" content="{REFRESH_MS // 1000}">""")

    # --- Header
    st.title("🏁 MinHash Algorithm Racing Track")
    st.caption(
        f"Event: **{event}** •  Load level: **{load_level}** •  "
        f"Refreshing every {REFRESH_MS // 1000}s"
    )

    # --- Live container table
    st.subheader("Live Race Status")
    workers = get_worker_containers(client, event)

    if not workers:
        st.warning(
            "No containers found. Start them with: "
            f"`docker compose -f docker-compose.{event}.yml up -d`"
        )
    else:
        rows = []
        for algo, container in workers.items():
            status = container_status(container)
            elapsed = container_elapsed(container)
            stats = (
                container_stats(container)
                if status == "running"
                else {"cpu_pct": 0.0, "ram_mb": 0.0}
            )
            progress = container_progress(container) or "-"

            _push_history(algo, stats["cpu_pct"], stats["ram_mb"])

            # Parse progress fraction
            frac = 0.0
            m = re.search(r"PROGRESS:\s*(\d+)/(\d+)", progress)
            if m:
                frac = int(m.group(1)) / max(int(m.group(2)), 1)
            if "DONE:" in progress:
                frac = 1.0

            rows.append(
                {
                    "Algorithm": algo.upper(),
                    "Status": status,
                    "CPU %": f"{stats['cpu_pct']:.1f}",
                    "RAM (MB)": f"{stats['ram_mb']:.0f}",
                    "Progress": f"{frac * 100:.0f}%",
                    "Elapsed (s)": f"{elapsed:.1f}",
                    "Last log": progress[:80],
                }
            )

        df_live = pd.DataFrame(rows)
        st.dataframe(df_live, use_container_width=True, hide_index=True)

        # Sparklines (CPU)
        st.subheader("CPU % over time")
        spark_data = {}
        for algo, hist in st.session_state.history.items():
            if hist:
                spark_data[algo.upper()] = [h["cpu"] for h in hist]
        if spark_data:
            max_len = max(len(v) for v in spark_data.values())
            df_spark = pd.DataFrame(
                {k: v + [None] * (max_len - len(v)) for k, v in spark_data.items()}
            )
            st.line_chart(df_spark)

    # --- Historical scores
    st.subheader("Historical Run Results (from scores volume)")
    score_records = load_score_files()
    if score_records:
        df_scores = pd.DataFrame(
            [
                {
                    "Run time": datetime.fromtimestamp(r.get("timestamp", 0)).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Algorithm": r.get("algo", "?").upper(),
                    "Level": r.get("load_level", "?"),
                    "Shape": "x".join(str(d) for d in r.get("shape", [])),
                    "Pairs": r.get("n_pairs", "?"),
                    "Elapsed (s)": r.get("elapsed_s", "?"),
                    "MAE": r.get("mae", "?"),
                    "Max err": r.get("max_error", "?"),
                    "Mem (KB)": r.get("memory_kb", "?"),
                }
                for r in score_records
            ]
        )
        st.dataframe(df_scores, use_container_width=True, hide_index=True)

        # Time-series: elapsed vs load_level per algo
        st.subheader("Speed vs Load Level")
        pivot = df_scores.pivot_table(
            index="Level", columns="Algorithm", values="Elapsed (s)", aggfunc="min"
        ).reset_index()
        if not pivot.empty:
            st.line_chart(pivot.set_index("Level"))

        st.subheader("Accuracy (MAE) vs Load Level")
        pivot_mae = df_scores.pivot_table(
            index="Level", columns="Algorithm", values="MAE", aggfunc="min"
        ).reset_index()
        if not pivot_mae.empty:
            st.line_chart(pivot_mae.set_index("Level"))
    else:
        st.info("No score files found yet - run a race first!")


if __name__ == "__main__":
    main()