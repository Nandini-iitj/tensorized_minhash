# NetworkLogGenerator - synthetic CIC-IDS2017-style network-log generator.

"""
Generates DataFrames mimicking the statistical structure of the CIC-IDS2017
dataset (Botnet, DDoS, PortScan attack patterns), so benchmarks and tests run
without downloading the full 50 GB dataset.
"""

import logging

import numpy as np
import pandas as pd

__all__ = ["NetworkLogGenerator"]

logger = logging.getLogger(__name__)


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
        n_benign: int = 500,
        n_attack: int = 200,
        seed: int = 42,
    ):
        self.n_src = n_src
        self.n_dst = n_dst
        self.n_port = n_port
        self.n_benign = n_benign
        self.n_attack = n_attack
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> tuple[pd.DataFrame, dict[str, list]]:
        """
        Returns (DataFrame with columns [src_ip, dst_ip, port, label],
        attack_groups: dict mapping label → list of (src, dst, port))
        """
        rows = []

        # Benign traffic: random sparse connections
        src_ips = self.rng.integers(0, self.n_src, self.n_benign)
        dst_ips = self.rng.integers(0, self.n_dst, self.n_benign)
        ports = self.rng.integers(0, self.n_port, self.n_benign)
        for s, d, p in zip(src_ips, dst_ips, ports, strict=False):
            rows.append({"src_ip": s, "dst_ip": d, "port": p, "label": "benign"})

        attack_groups = {}

        # Attack 1: PortScan — one source hits all ports on one destination
        ps_src = self.rng.integers(0, self.n_src)
        ps_dst = self.rng.integers(0, self.n_dst)
        ps_ports = self.rng.choice(self.n_port, size=min(50, self.n_port), replace=False)
        for p in ps_ports:
            rows.append({"src_ip": ps_src, "dst_ip": ps_dst, "port": p, "label": "portscan"})
        attack_groups["portscan"] = [(ps_src, ps_dst, p) for p in ps_ports]

        # Attack 2: DDoS — many sources hit one destination on one port
        ddos_dst = self.rng.integers(0, self.n_dst)
        ddos_port = self.rng.integers(0, self.n_port)
        ddos_srcs = self.rng.choice(self.n_src, size=min(40, self.n_src), replace=False)
        for s in ddos_srcs:
            rows.append({"src_ip": s, "dst_ip": ddos_dst, "port": ddos_port, "label": "ddos"})
        attack_groups["ddos"] = [(s, ddos_dst, ddos_port) for s in ddos_srcs]

        # Attack 3: Botnet — cluster of IPs talking on unusual port range
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
        logger.info(f"Generated {len(df)} unique connections ({len(rows)} raw)")
        return df, attack_groups