"""
NetworkTensorBuilder - converts network-log DataFrames into 3D binary tensors.

Shape: (n_src_buckets, n_dst_buckets, n_port_buckets)
Value: 1.0 if any connection was made in that (src, dst, port) bucket, else 0.0

For real CIC-IDS2017 data (millions of IPs), IPs are hashed into buckets of
configurable size to keep tensor dimensions fixed regardless of IP cardinality.
Note: modulo bucketing slightly inflates Jaccard estimates for sparse data but
is a known trade-off for tractable tensor dimensions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["NetworkTensorBuilder"]

logger = logging.getLogger(__name__)


class NetworkTensorBuilder:
    """
    Converts a network-log DataFrame into a 3D binary tensor.

    Shape: (n_src_buckets, n_dst_buckets, n_port_buckets)
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
        Convert log DataFrame to 3D binary tensor.
        Columns expected: src_ip, dst_ip, port (integer indices or hashable)
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
        Expected columns (subset): ' Source IP', ' Destination IP', ' Destination Port'
        The dataset uses awkward leading-space column names.
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