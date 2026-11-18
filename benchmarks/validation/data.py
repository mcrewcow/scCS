"""Versioned public-data access and explicit barcode alignment for benchmarks."""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

SOURCES = {
    "larry_velocity": (
        "https://ndownloader.figshare.com/files/37028569",
        "larry_invitro_adata_sub_raw.h5ad",
        "md5",
        "d618305c0ee0e41570b1147e63df332a",
    ),
    "aml_loom": (
        "https://zenodo.org/records/6587903/files/Pollyea_merged.loom?download=1",
        "Pollyea_merged.loom",
        "md5",
        "00a4646536502f76e7c6f4ad399bb896",
    ),
    "aml_metadata": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116481/suppl/"
        "GSE116481_cell_metadata_full.txt.gz",
        "aml_metadata.tsv.gz",
        "sha256",
        "f13f7e17a3cc3c2290048f75cef371a6005427fec0a56bae9c6ffc7b40d17523",
    ),
    "larry": (
        "https://kleintools.hms.harvard.edu/tools/downloads/cospar/LARRY_adata_preprocessed.h5ad",
        "LARRY_adata_preprocessed.h5ad",
        "sha256",
        "16607046cd2550dbcdb33c24c58e5b8852c5558f7b52a437e0ae20b0137ef881",
    ),
}


def fetch(key, directory):
    url, name, algorithm, expected = SOURCES[key]
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            with (
                urllib.request.urlopen(url, timeout=90) as response,
                temporary.open("wb") as f,
            ):
                while block := response.read(1024 * 1024):
                    f.write(block)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if expected:
        hasher = hashlib.new(algorithm)
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch for {name}; remove the file and retry.")
    return path


def _strings(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load_aml(directory, n_genes=2000):
    """Read only mapped cells and coverage-selected genes from the original loom.

    Source annotations mix blasts with immature erythroid cells. No malignant
    genotype is inferred here. The join uses (sample, barcode), never barcode alone.
    """
    metadata = pd.read_csv(fetch("aml_metadata", directory), sep="\t", index_col=0)
    sample = metadata["sample"].astype(str)
    metadata["join_key"] = sample + ":" + metadata.index.str.split("-").str[0]
    if metadata.join_key.duplicated().any():
        raise ValueError("Ambiguous sample/barcode keys in AML metadata.")
    with h5py.File(fetch("aml_loom", directory)) as f:
        ids = _strings(f["col_attrs/CellID"][:])
        keys = np.array([x.split("_")[1] + ":" + x.split(":")[1].split("-")[0] for x in ids])
        if pd.Index(keys).duplicated().any():
            raise ValueError("Ambiguous sample/barcode keys in AML loom.")
        lookup = metadata.set_index("join_key")
        cell_idx = np.flatnonzero(pd.Index(keys).isin(lookup.index))
        obs = lookup.loc[keys[cell_idx]].copy()
        obs.index = ids[cell_idx]
        assert len(obs) == len(metadata), "Some annotated AML cells could not be joined."
        genes = _strings(f["row_attrs/Gene"][:])
        abundance = np.zeros(len(genes))
        variance = np.zeros(len(genes))
        unspliced = np.zeros(len(genes))
        for start in range(0, len(genes), 256):
            end = min(start + 256, len(genes))
            s = f["layers/spliced"][start:end, :][:, cell_idx].astype(float)
            u = f["layers/unspliced"][start:end, :][:, cell_idx].astype(float)
            abundance[start:end] = s.sum(1)
            unspliced[start:end] = u.sum(1)
            variance[start:end] = s.var(1) / (s.mean(1) + 1)
        eligible = (abundance >= 20) & (unspliced >= 10)
        # Keep marker genes for inspection as well as variable genes for velocity.
        markers = np.isin(
            genes,
            [
                "CD34",
                "KIT",
                "GATA1",
                "KLF1",
                "HBB",
                "HBA1",
                "LYZ",
                "S100A8",
                "CTSS",
                "MPO",
                "GATA2",
            ],
        )
        ranked = np.argsort(np.where(eligible, variance, -1))[-n_genes:]
        gene_idx = np.flatnonzero(np.isin(np.arange(len(genes)), ranked) | markers)
        spliced = sparse.csr_matrix(
            f["layers/spliced"][gene_idx, :][:, cell_idx].T, dtype=np.float32
        )
        unspliced = sparse.csr_matrix(
            f["layers/unspliced"][gene_idx, :][:, cell_idx].T, dtype=np.float32
        )
    a = ad.AnnData(X=spliced.copy(), obs=obs, var=pd.DataFrame(index=genes[gene_idx]))
    a.var_names_make_unique()
    a.layers["spliced"], a.layers["unspliced"] = spliced, unspliced
    a.obs["patient"] = "patient_1"
    return a
