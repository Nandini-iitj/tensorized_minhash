"""
app.py - MinHash Racing Scoreboard (Streamlit + Docker SDK)

Architecture:
  • Uses st.rerun() for live polling - keeps the same Streamlit session so
    st.session_state.history accumulates properly (meta-refresh creates a new
    session every cycle and wipes history).
  • Two tabs:
      🏁 Live Race   - per-algo cards, progress bars, real-time CPU+RAM charts
      📊 Historical  - past races replayed from telemetry JSON + results table
  • When all containers finish, a telemetry JSON is saved to SCORES_DIR so the
    historical tab can show the same CPU/RAM time-series later.

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

# ── Config ─────────────────────────────────────────────────────────────────

SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/scores"))

ALGO_COLORS = {
    "kron":    "#4e79a7",  # blue
    "tt":      "#f28e2b",  # orange
    "minhash": "#e15759",  # red
}

ALGO_LABELS = {
    "kron":    "Kronecker",
    "tt":      "Tensor Train",
    "minhash": "MinHash (dense)",
}

CONTAINER_NAMES = {
    "event1": ["racing_worker_kron", "racing_worker_tt", "racing_worker_minhash"],
    "event2": ["racing_worker_kron", "racing_worker_tt"],
}

LEVEL_INFO = {
    1: "15³ · 30 pairs",   2: "20³ · 50 pairs",
    3: "28³ · 70 pairs",   4: "35³ · 100 pairs",
    5: "50³ · 150 pairs",  6: "60³ · 200 pairs",
    7: "70³ · 250 pairs",  8: "80³ · 320 pairs",
    9: "90³ · 400 pairs",  10: "100³ · 500 pairs",
}

POLL_S   = 1.5   # seconds between live refreshes
MAX_PTS  = 300   # max time-series points retained per algo
SHARED_INPUT_DIR = Path(os.environ.get("SHARED_INPUT_DIR", "/shared_input"))

# ── Docker helpers ──────────────────────────────────────────────────────────


@st.cache_resource
def get_docker_client():
    return docker.from_env()


def get_workers(client, event: str) -> dict:
    result = {}
    for cname in CONTAINER_NAMES.get(event, []):
        try:
            c = client.containers.get(cname)
            result[cname.replace("racing_worker_", "")] = c
        except docker.errors.NotFound:
            pass
    return result


def c_status(c) -> str:
    try:
        c.reload()
        return c.status
    except Exception:
        return "unknown"


def c_stats(c) -> dict:
    try:
        raw = c.stats(stream=False)
        cpu = _cpu_pct(raw)
        ram = raw["memory_stats"].get("usage", 0) / (1024 * 1024)
        return {"cpu": round(cpu, 1), "ram": round(ram, 1)}
    except Exception:
        return {"cpu": 0.0, "ram": 0.0}


def _cpu_pct(stats: dict) -> float:
    try:
        cd  = stats["cpu_stats"]
        pd_ = stats["precpu_stats"]
        cpu_d = cd["cpu_usage"]["total_usage"] - pd_["cpu_usage"]["total_usage"]
        sys_d = cd["system_cpu_usage"] - pd_["system_cpu_usage"]
        n = cd.get("online_cpus") or len(cd["cpu_usage"].get("percpu_usage", [1]))
        return (cpu_d / sys_d) * n * 100.0 if sys_d > 0 else 0.0
    except (KeyError, ZeroDivisionError):
        return 0.0


def c_elapsed(c) -> float:
    try:
        started = c.attrs["State"]["StartedAt"]
        dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        return (datetime.now(UTC) - dt).total_seconds()
    except Exception:
        return 0.0


def c_start_time(c) -> str | None:
    try:
        return c.attrs["State"]["StartedAt"]
    except Exception:
        return None


def c_progress_line(c) -> str | None:
    try:
        logs = c.logs(tail=30).decode("utf-8", errors="replace")
        for line in reversed(logs.splitlines()):
            if any(kw in line for kw in ("PROGRESS:", "DONE:", "OOM:")):
                return line.strip()
    except Exception:
        pass
    return None


# ── Scores / telemetry I/O ─────────────────────────────────────────────────


def load_manifest() -> dict | None:
    """Read the actual input manifest.json written by generate_input"""
    p = SHARED_INPUT_DIR / "input_manifest.json"
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def load_score_files() -> list[dict]:
    if not SCORES_DIR.exists():
        return []
    out = []
    for f in sorted(SCORES_DIR.glob("*_metrics.json"), reverse=True):
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["_file"] = f.name
            out.append(d)
        except Exception:
            pass
    return out


def load_telemetry_files() -> list[dict]:
    if not SCORES_DIR.exists():
        return []
    out = []
    for f in sorted(SCORES_DIR.glob("telemetry_*.json"), reverse=True):
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["_file"] = f.name
            out.append(d)
        except Exception:
            pass
    return out


def save_telemetry(race_id: str, history: dict, algos_done: dict):
    ts = int(time.time())
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORES_DIR / f"telemetry_{ts}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "race_id":   race_id,
                "timestamp": ts,
                "algos":     list(history.keys()),
                "telemetry": history,
                "final":     algos_done,
            },
            f,
            indent=2,
        )


# ── Session state ───────────────────────────────────────────────────────────


def init_state():
    defaults = {
        "history":        {},    # algo -> [{t, cpu, ram}]
        "race_id":        None,  # earliest container start-time string
        "race_done":      False, # telemetry saved for current race
        "race_start_abs": None,  # float: time.time() when race first detected
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def push_history(algo: str, t_rel: float, cpu: float, ram: float):
    h = st.session_state.history.setdefault(algo, [])
    h.append({"t": round(t_rel, 1), "cpu": cpu, "ram": ram})
    if len(h) > MAX_PTS:
        h.pop(0)


def detect_race_id(workers: dict) -> str | None:
    times = [c_start_time(c) for c in workers.values()]
    times = [t for t in times if t]
    return min(times) if times else None


# ── Chart helpers ───────────────────────────────────────────────────────────


def timeseries_df(history: dict, metric: str) -> "pd.DataFrame | None":
    if not history:
        return None
    frames = {}
    for algo, pts in history.items():
        if pts:
            label = ALGO_LABELS.get(algo, algo.upper())
            frames[label] = pd.Series(
                [p[metric] for p in pts],
                index=[p["t"] for p in pts],
            )
    if not frames:
        return None
    df = pd.DataFrame(frames)
    df.index.name = "seconds"
    return df


def draw_ts(history: dict, metric: str, height: int = 220):
    df = timeseries_df(history, metric)
    if df is not None and not df.empty:
        st.line_chart(df, height=height, use_container_width=True)
    else:
        st.caption("Waiting for data...")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="MinHash Racing Track",
        page_icon="🏁",
        layout="wide",
    )
    init_state()
    client = get_docker_client()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🏁 Racing Controls")

        event = st.selectbox(
            "Event",
            ["event1", "event2"],
            format_func=lambda e: {
                "event1": "Event 1 - Kron vs TT vs MinHash",
                "event2": "Event 2 - Kron vs TT (higher load)",
            }[e],
        )

        # active race info read from Manifest, not hard coded
        manifest = load_manifest()
        if manifest:
            m_level = manifest.get("load_level", "?")
            m_shape = manifest.get("shape", [])
            m_pairs = manifest.get("n_pairs", "?")
            shape_str = "\u00d7".join(str(d) for d in m_shape)
            st.markdown("**Active Race Input**")
            st.markdown(
                f"""
                | | |
                |---|---|
                | Load level | `{m_level}` |
                | Tensor Shape | `{shape_str}` |
                | Pairs | `{m_pairs}` |
                """
            )
        else:
            st.info("No input Manifest Found Yet")

        st.divider()
        auto_refresh = st.toggle("Auto-refresh (live)", value=True)

        st.divider()
        st.markdown("**Generate New Input**")
        st.caption("Canging Level here only takes effect when you click Generate, then restart containers")
        load_level = st.slider(
            "New LOAD_LEVEL", 1, 10, manifest.get("loadl_level", 8) if manifest else 8,
            help = "Higher = Larger Tensor shape + more pairs",
        )
        st.caption(LEVEL_INFO.get(load_level, "?"))
        if st.button("\u26a1 Generate Input", use_container_width=True):
            import subprocess
            import sys as _sys
            # generate input.py is copied to /app/shared_input/ inside the container
            # (same directory as app.py's parent), output goes to the shared volume
            gen = Path(__file__).parent / "shared_input" / "generate_input.py"
            out_dir = str(SHARED_INPUT_DIR)
            try:
                with st.spinner("Generating Tensor Pairs..."):
                    subprocess.run(
                        [_sys.executable, str(gen),
                         "--load-level", str(load_level),
                         "--out-dir", out_dir],
                         check=True,
                         capture_output=True,
                         text=True,
                    )
                st.success("input ready \u2014 restart containers to race!")
            except subprocess.CalledProcessError as e:
                st.error(
                    f"generate_input.py failed (exit {e.returncode}).\n\n"
                    f"**stderr:** {e.stderr[-600:] if e.stderr else '(none)'}\n\n"
                    "In Docker, set the load level via the environment variable instead: \n"
                    "```\n"
                    f"LOAD_LEVEL={load_level} docker compose -g ddocker-compose.event1.yml up -d\n"
                    "```"
                )

    # ── Tab layout ─────────────────────────────────────────────────────────
    tab_live, tab_history = st.tabs(["🏁  Live Race", "📊  Historical Runs"])

    # ======================================================================
    #  Collect live data - runs regardless of which tab is visible
    # ======================================================================
    workers = get_workers(client, event)
    any_running = False
    algo_info: dict = {}
    all_done = True
    algos_done: dict = {}

    if workers:
        race_id = detect_race_id(workers)
        if race_id != st.session_state.race_id:
            # New race detected - reset accumulated history
            st.session_state.history = {}
            st.session_state.race_id = race_id
            st.session_state.race_done = False
            st.session_state.race_start_abs = time.time()

        race_start = st.session_state.race_start_abs or time.time()
        t_rel = time.time() - race_start

        for algo, c in workers.items():
            status  = c_status(c)
            elapsed = c_elapsed(c)
            is_running = status == "running"
            if is_running:
                all_done = False
                any_running = True

            stats = c_stats(c) if is_running else {"cpu": 0.0, "ram": 0.0}
            log   = c_progress_line(c) or "-"

            frac = 0.0
            m = re.search(r"PROGRESS:\s*(\d+)/(\d+)", log)
            if m:
                frac = int(m.group(1)) / max(int(m.group(2)), 1)

            done_m = re.search(r"DONE:.*elapsed=([\d.]+)s.*mae=([\d.]+)", log)
            oom    = "OOM:" in log

            if done_m or status == "exited":
                frac = 1.0
                algos_done[algo] = {
                    "elapsed_s": float(done_m.group(1)) if done_m else elapsed,
                    "mae":       float(done_m.group(2)) if done_m else None,
                }

            push_history(algo, t_rel, stats["cpu"], stats["ram"])

            algo_info[algo] = {
                "status":  status,
                "running": is_running,
                "elapsed": elapsed,
                "cpu":     stats["cpu"],
                "ram":     stats["ram"],
                "frac":    frac,
                "log":     log,
                "oom":     oom,
            }

        # Save telemetry once when all containers finish
        if all_done and not st.session_state.race_done and algos_done:
            save_telemetry(race_id, st.session_state.history, algos_done)
            st.session_state.race_done = True

    # ======================================================================
    #  TAB 1 - LIVE RACE
    # ======================================================================
    with tab_live:
        if not workers:
            st.warning(
                f"No containers found. Start them with:\n\n"
                f"```\ndocker compose -f docker-compose.{event}.yml up -d\n```"
            )
        else:
            # ── Header blinker ──────────────────────────────────────────
            if all_done:
                st.success("🏆  Race complete! All workers have finished.")
            else:
                blink = "🔴" if int(time.time() * 2) % 2 == 0 else "🟠"
                elapsed_total = time.time() - (st.session_state.race_start_abs or time.time())
                st.markdown(
                    f"<h2 style='margin:0'>{blink}&nbsp;&nbsp;Race in progress...</h2>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Race clock: **{elapsed_total:.0f}s**  ·  "
                    f"auto-refreshing every {POLL_S:.0f}s"
                )

            st.divider()

            # ── Per-algo cards ──────────────────────────────────────────
            finish_order = sorted(
                [a for a, i in algo_info.items() if i["frac"] == 1.0],
                key=lambda a: algo_info[a]["elapsed"],
            )
            medals = dict(zip(finish_order, ["🥇", "🥈", "🥉"], strict=False))

            cols = st.columns(len(algo_info))
            for col, (algo, info) in zip(cols, algo_info.items(), strict=False):
                color = ALGO_COLORS.get(algo, "#888")
                label = ALGO_LABELS.get(algo, algo.upper())
                medal = medals.get(algo, "")

                if info["oom"]:
                    icon, badge_txt = "💥", "OOM"
                elif info["frac"] == 1.0:
                    icon = medal or "🏁"
                    badge_txt = f"{info['elapsed']:.1f}s"
                elif info["running"]:
                    icon, badge_txt = "🏃", f"{info['elapsed']:.1f}s"
                else:
                    icon, badge_txt = "⏳", info["status"]

                with col:
                    st.markdown(
                        f"""
                        <div style="border-left:5px solid {color};
                                    padding:10px 14px;border-radius:6px;
                                    background:rgba(255,255,255,0.04);
                                    margin-bottom:8px;">
                          <div style="font-size:0.85em;color:#aaa;margin-bottom:2px">
                            {icon} {label}
                          </div>
                          <div style="font-size:2em;font-weight:bold;
                                      color:{color};line-height:1.1">
                            {badge_txt}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(info["frac"], text=f"{info['frac'] * 100:.0f}%")
                    st.caption(f"CPU {info['cpu']:.1f}%  ·  RAM {info['ram']:.0f} MB")
                    st.caption(info["log"][:72])

            st.divider()

            # ── Real-time CPU + RAM charts ──────────────────────────────
            hist = st.session_state.history
            if hist and any(hist.values()):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**CPU % - real-time**")
                    draw_ts(hist, "cpu")
                with c2:
                    st.markdown("**RAM (MB) - real-time**")
                    draw_ts(hist, "ram")
            else:
                st.info("Charts will appear once polling begins...")

            # ── Results table (once race is done) ───────────────────────
            if all_done:
                st.divider()
                st.subheader("Race Results")
                now = time.time()
                recent = [r for r in load_score_files()
                          if abs(r.get("timestamp", 0) - now) < 300]
                if recent:
                    rows = []
                    for r in sorted(recent, key=lambda r: r.get("elapsed_s", 999)):
                        rows.append({
                            "Rank":        ["🥇", "🥈", "🥉", "4th"][min(len(rows), 3)],
                            "Algorithm":   ALGO_LABELS.get(r.get("algo", ""), r.get("algo", "?")),
                            "Elapsed (s)": r.get("elapsed_s", "?"),
                            "MAE":         r.get("mae", "?"),
                            "Max err":     r.get("max_error", "?"),
                            "Pairs":       r.get("n_pairs", "?"),
                            "Mem (KB)":    r.get("memory_kb", "?"),
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ======================================================================
    #  TAB 2 - HISTORICAL RUNS
    # ======================================================================
    with tab_history:
        tel_files   = load_telemetry_files()
        score_files = load_score_files()

        if not tel_files and not score_files:
            st.info("No historical data yet - complete a race first!")
        elif tel_files:
            st.subheader(f"{len(tel_files)} past race(s) on record")
            for i, tel in enumerate(tel_files):
                ts       = tel.get("timestamp", 0)
                run_lbl  = datetime.fromtimestamp(ts).strftime("%Y-%m-%d  %H:%M:%S")
                tel_hist = tel.get("telemetry", {})
                algos_in = tel.get("algos", [])

                all_t    = [p["t"] for pts in tel_hist.values() for p in pts]
                duration = f"{max(all_t):.0f}s" if all_t else "?"

                header = (
                    f"🏁  {run_lbl}  -  duration {duration}  -  "
                    + ", ".join(ALGO_LABELS.get(a, a.upper()) for a in algos_in)
                )
                with st.expander(header, expanded=(i == 0)):
                    if tel_hist:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**CPU % during race**")
                            draw_ts(tel_hist, "cpu")
                        with c2:
                            st.markdown("**RAM (MB) during race**")
                            draw_ts(tel_hist, "ram")
                    else:
                        st.caption("No telemetry data stored for this run.")

                    run_scores = [
                        r for r in score_files
                        if abs(r.get("timestamp", 0) - ts) < 180
                    ]
                    if run_scores:
                        st.markdown("**Results**")
                        rows = []
                        for r in sorted(run_scores, key=lambda r: r.get("elapsed_s", 999)):
                            rows.append({
                                "Algorithm":   ALGO_LABELS.get(r.get("algo", ""), r.get("algo", "?")),
                                "Shape":       "×".join(str(d) for d in r.get("shape", [])),
                                "Pairs":       r.get("n_pairs", "?"),
                                "Elapsed (s)": r.get("elapsed_s", "?"),
                                "MAE":         r.get("mae", "?"),
                                "Max err":     r.get("max_error", "?"),
                                "Mem (KB)":    r.get("memory_kb", "?"),
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption("No matching metric files for this run.")
        else:
            # Fallback: telemetry files not yet present, show raw summary table
            st.subheader("Run Summary (no telemetry available yet)")
            rows = []
            for r in score_files:
                rows.append({
                    "Run time":    datetime.fromtimestamp(r.get("timestamp", 0)).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Algorithm":   ALGO_LABELS.get(r.get("algo", ""), r.get("algo", "?")),
                    "Level":       r.get("load_level", "?"),
                    "Shape":       "×".join(str(d) for d in r.get("shape", [])),
                    "Pairs":       r.get("n_pairs", "?"),
                    "Elapsed (s)": r.get("elapsed_s", "?"),
                    "MAE":         r.get("mae", "?"),
                    "Max err":     r.get("max_error", "?"),
                    "Mem (KB)":    r.get("memory_kb", "?"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Live auto-refresh - st.rerun() keeps the same session alive ─────────
    if auto_refresh and any_running:
        time.sleep(POLL_S)
        st.rerun()


if __name__ == "__main__":
    main()
