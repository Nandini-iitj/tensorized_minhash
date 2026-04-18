"""
demo.genome.loader — NCBI FASTA loader for mitochondrial DNA.

Downloads and parses sequence data for various species to build
the binary k-mer tensors used in the similarity benchmarks.
"""

import os
import subprocess
import numpy as np

__all__ = ["load_genome_tensor"]

# Species mapping from NCBI Accession IDs
SPECIES_IDS = {
    "human": "NC_012920.1",
    "chimp": "NC_001643.1",
    "gorilla": "NC_011120.1",
    "orangutan": "NC_002083.1",
    "mouse": "NC_005089.1",
    "chicken": "NC_001323.1",
    "zebrafish": "NC_002333.2"
}

def load_genome_tensor(species: str, shape: tuple[int, ...] = (64, 64, 64)) -> np.ndarray:
    """
    Loads a genome sequence, computes k-mers, and maps them into a 3D tensor.
    """
    sequence = _get_sequence(species)
    return _sequence_to_tensor(sequence, shape)

def _get_sequence(species: str) -> str:
    """Fetches FASTA from local disk, downloading from NCBI if missing."""
    data_dir = os.path.join("demo", "genome", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{species}.fasta")
    
    if not os.path.exists(file_path):
        ncbi_id = SPECIES_IDS.get(species.lower())
        if not ncbi_id:
            raise ValueError(f"Unknown species: {species}")
            
        print(f"Downloading {species} mitochondrial DNA ({ncbi_id})...")
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={ncbi_id}&rettype=fasta&retmode=text"
        subprocess.run(["curl", "-o", file_path, url], check=True)

    with open(file_path, "r") as f:
        # Skip the header line and join the sequence
        lines = f.readlines()
        return "".join(line.strip() for line in lines if not line.startswith(">"))

def _sequence_to_tensor(sequence: str, shape: tuple[int, ...]) -> np.ndarray:
    """
    Maps DNA k-mers (3-mers) into a 3rd-order binary tensor.
    Each dimension represents a base (A, C, T, G) mapped via modulo.
    """
    tensor = np.zeros(shape, dtype=np.float32)
    
    # k-mer length matching tensor order (3)
    k = len(shape)
    
    # Map bases to numerical values for indexing
    mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        try:
            # Map each base in the k-mer to a coordinate in the tensor
            # Using modulo to fit into the specified shape
            idx = tuple(mapping[base] % shape[j] for j, base in enumerate(kmer))
            tensor[idx] = 1.0
        except KeyError:
            # Skip non-ACTG characters (e.g., 'N')
            continue
            
    return tensor