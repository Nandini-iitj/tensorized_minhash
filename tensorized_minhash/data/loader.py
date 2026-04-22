"""
Data pipeline: load CIC-IDS2017 (or KDD Cup) network logs and reshape to
3D tensors of shape (src_ip_idx, dst_ip_idx, port_idx).

Each cell T[i, j, k] = 1 if source IP i connected to destination IP j on
port k, else 0. This encodes the presence/absence of attack patterns.

We include a synthetic data generator that mimics the statistical properties
of CIC-IDS2017 (Botnet, DDoS, PortScan attack patterns) so the code runs
without downloading the full 50 GB dataset.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["NetworkLogGenerator", "NetworkTensorBuilder"]

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Synthetic data generator (mirrors CIC-IDS2017 structure)
# -----------------------------------------------------------------------------

class NetworkLogGenerator:
    """
    Generates synthetic network connection logs resembling CIC-IDS2017.

    Injects three attack patterns:
    - PortScan: one source IP hits many ports on one destination
    - DDoS: many sources flood one destination on one port
    - Botnet: a cluster of IPs communicate on non-standard ports
    """

    def __init__(
        self,
        n_src: int = 100,
        n_dst: int = 100,
        n_port: int = 100,
        n_benign: int = 5_000,
        n_attack: int = 500,
        seed: int = 42,
    ):
        self.n_src = n_src
        self.n_dst = n_dst
        self.n_port = n_port
        self.n_benign = n_benign
        self.n_attack = n_attack
        self.rng = np.random.default_rng(seed)

    def generate(self) -> tuple[pd.DataFrame, dict[str, list]]:
        """
        Returns (DataFrame with columns [src_ip, dst_ip, port, label],
                 attack_groups: dict mapping label -> list of (src, dst, port))
        """
        rows = []

        # Benign traffic: random sparse connections
        src_ips = self.rng.integers(0, self.n_src, self.n_benign)
        dst_ips = self.rng.integers(0, self.n_dst, self.n_benign)
        ports = self.rng.integers(0, self.n_port, self.n_benign)
        for s, d, p in zip(src_ips, dst_ips, ports, strict=False):
            rows.append({"src_ip": s, "dst_ip": d, "port": p, "label": "benign"})

        attack_groups = {}

        # Attack 1: PortScan - one source hits all ports on one destination
        ps_src = self.rng.integers(0, self.n_src)
        ps_dst = self.rng.integers(0, self.n_dst)
        ps_ports = self.rng.choice(self.n_port, size=min(50, self.n_port), replace=False)
        for p in ps_ports:
            rows.append({"src_ip": ps_src, "dst_ip": ps_dst, "port": p, "label": "portscan"})
        attack_groups["portscan"] = [(ps_src, ps_dst, p) for p in ps_ports]

        # Attack 2: DDoS - many sources hit one destination on one port
        ddos_dst = self.rng.integers(0, self.n_dst)
        ddos_port = self.rng.integers(0, self.n_port)
        ddos_srcs = self.rng.choice(self.n_src, size=min(40, self.n_src), replace=False)
        for s in ddos_srcs:
            rows.append({"src_ip": s, "dst_ip": ddos_dst, "port": ddos_port, "label": "ddos"})
        attack_groups["ddos"] = [(s, ddos_dst, ddos_port) for s in ddos_srcs]

        # Attack 3: Botnet - cluster of IPs talking on unusual port range
        bot_srcs = self.rng.choice(self.n_src, size=10, replace=False)
        bot_dsts = self.rng.choice(self.n_dst, size=10, replace=False)
        bot_ports = self.rng.integers(self.n_port - 10, self.n_port, size=5)
        for s in bot_srcs:
            for d in bot_dsts:
                for p in bot_ports:
                    rows.append({"src_ip": s, "dst_ip": d, "port": int(p), "label": "botnet"})
        attack_groups["botnet"] = [
            (int(s), int(d), int(p)) for s in bot_srcs for d in bot_dsts for p in bot_ports
        ]

        df = pd.DataFrame(rows).drop_duplicates(subset=["src_ip", "dst_ip", "port"])
        logger.info(f"Generated {len(df):,} unique connections ({len(rows):,} raw)")
        return df, attack_groups


# -----------------------------------------------------------------------------
# Tensor builder
# -----------------------------------------------------------------------------

class NetworkTensorBuilder:
    """
    Converts a network-log DataFrame into a 3D binary tensor.

    Shape: (n_src_buckets, n_dst_buckets, n_port_buckets)
    Value: 1.0 if any connection was made, else 0.0

    For real CIC-IDS2017 data (millions of IPs), we hash IPs into buckets
    of configurable size to control tensor dimensions.
    """

    def __init__(
        self,
        n_src: int = 100,
        n_dst: int = 100,
        n_port: int = 100,
    ):
        self.n_src = n_src
        self.n_dst = n_dst
        self.n_port = n_port

    def build_tensor(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert log DataFrame to 3D binary tensor
        Columns expected: src_ip, dst_ip, port (integer indices or hashable)

        Note: uses modulo bucketing, so distinct IPs/ports that differ only by
        a multiple of the bucket size map to the same index. This is a known
        trade-off: it slightly inflates Jaccard estimates for sparse data but
        keeps tensor dimensions fixed regardless of IP cardinality
        """
        tensor = np.zeros((self.n_src, self.n_dst, self.n_port), dtype=np.float32)

        src_idx = (df["src_ip"].astype(int) % self.n_src).values
        dst_idx = (df["dst_ip"].astype(int) % self.n_dst).values
        port_idx = (df["port"].astype(int) % self.n_port).values

        tensor[src_idx, dst_idx, port_idx] = 1.0
        return tensor

    def build_tensor_batch(
        self,
        df: pd.DataFrame,
        time_col: str | None = None,
        window_size: int = 1000,
    ) -> list[np.ndarray]:
        """
        Build a list of tensors from rolling time windows
        Useful for detecting temporal attack patterns
        """
        if time_col is None or time_col not in df.columns:
            # Fall back to index-based windows
            chunks = [df.iloc[i : i + window_size] for i in range(0, len(df), window_size)]
        else:
            df = df.sort_values(time_col)
            chunks = [df.iloc[i : i + window_size] for i in range(0, len(df), window_size)]

        tensors = [self.build_tensor(chunk) for chunk in chunks if len(chunk) > 0]
        logger.info(f"Built {len(tensors)} tensor windows from {len(df):,} rows")
        return tensors

    @staticmethod
    def load_cic_ids2017(csv_path: str) -> pd.DataFrame:
        """
        Load and normalise a CIC-IDS2017 CSV file.
        Expected columns (subset): 'Source IP', 'Destination IP', 'Destination Port'
        The dataset uses awkward leading-space column names
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # Map raw columns to standard names
        rename = {
            "Source IP": "src_ip_raw",
            "Destination IP": "dst_ip_raw",
            "Destination Port": "port",
            "Label": "label",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "label" not in df.columns:
            df["label"] = "unknown"

        # Vectorised IP-string -> integer conversion (avoids slow row-by-row .apply)
        df["src_ip"] = NetworkTensorBuilder._ip_series_to_int(df["src_ip_raw"])
        df["dst_ip"] = NetworkTensorBuilder._ip_series_to_int(df["dst_ip_raw"])
        df["port"] = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)

        logger.info(f"Loaded CIC-IDS2017: {len(df):,} rows")
        return df

    @staticmethod
    def _ip_series_to_int(ip_series: pd.Series) -> pd.Series:
        """Vectorised dotted-decimal IP to integer (avoids slow .apply loop)."""
        parts = (
            ip_series.astype(str)
            .str.split(".", expand=True)
            .reindex(columns=[0, 1, 2, 3], fill_value="0")
        )
        result = pd.Series(0, index=ip_series.index, dtype=np.int64)
        for i in range(4):
            result += pd.to_numeric(parts[i], errors="coerce").fillna(0).astype(np.int64) * (
                256 ** (3 - i)
            )

        return result