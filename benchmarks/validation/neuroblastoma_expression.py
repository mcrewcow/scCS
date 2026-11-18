"""Stream human-reference expression matrices from the GSE288099 xenografts."""

import gzip
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
import shutil

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

IDENTITY = ["PHOX2B", "UCHL1", "CHGA", "CHGB", "NCAM1", "ELAVL4"]
NEURON = ["TUBB3", "NEFM", "DCX", "GAP43", "TH", "MAP2"]
CYCLING = ["MKI67", "TOP2A", "UBE2C", "CENPF", "BIRC5", "CDK1"]
HELD_OUT = ["NEFL", "STMN2", "MCM2", "PCNA"]


def geo_input(directory, gsm, library, suffix):
    filename = f"{gsm}_{library}_{suffix}.gz"
    path = Path(directory) / filename
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{filename}"
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with urlopen(url, timeout=180) as source, temp.open("wb") as dest:
            shutil.copyfileobj(source, dest, length=1 << 20)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def expression_chunks(path):
    """Yield complete barcodes from the column-sorted GEO coordinate matrices."""
    with gzip.open(path, "rt") as stream:
        assert stream.readline().strip() == "%%MatrixMarket matrix coordinate integer general"
        dimensions = next(line for line in stream if not line.startswith("%"))
        n_genes, n_barcodes, expected = map(int, dimensions.split())
        pending = np.empty((0, 3), dtype=np.int32)
        total = 0
        previous = 0
        for frame in pd.read_csv(
            stream, sep=r"\s+", header=None, dtype=np.int32, chunksize=1_000_000
        ):
            rows = np.concatenate([pending, frame.to_numpy()])
            total += len(frame)
            cols = rows[:, 1]
            if cols[0] <= previous or np.any(cols[1:] < cols[:-1]):
                raise ValueError("Expected entries ordered by barcode index")
            stop = np.searchsorted(cols, cols[-1])
            if stop:
                previous = cols[stop - 1]
                yield rows[:stop], n_genes, n_barcodes
            pending = rows[stop:].copy()
        if len(pending):
            yield pending, n_genes, n_barcodes
        if total != expected:
            raise ValueError(f"Incomplete matrix: {total} entries, expected {expected}")


def read_tumor(directory, gsm, library, condition, cap=800):
    feature_file = geo_input(directory, gsm, library, "features.tsv")
    barcode_file = geo_input(directory, gsm, library, "barcodes.tsv")
    matrix_file = geo_input(directory, gsm, library, "matrix.mtx")
    features = pd.read_csv(feature_file, sep="\t", header=None, names=["id", "symbol", "type"])
    barcodes = pd.read_csv(barcode_file, sep="\t", header=None)[0].to_numpy()
    assert np.all(features.type == "Gene Expression") and features.id.is_unique
    indices = {
        g: np.flatnonzero(features.symbol.eq(g)) for g in IDENTITY + NEURON + CYCLING + HELD_OUT
    }
    assert all(len(v) == 1 for v in indices.values())
    mito = features.symbol.str.startswith("MT-").to_numpy()
    obs_parts, matrices = [], []
    n_qc = 0
    for rows, n_genes, n_barcodes in expression_chunks(matrix_file):
        assert (n_genes, n_barcodes) == (len(features), len(barcodes))
        low, high = rows[0, 1], rows[-1, 1]
        x = sparse.csr_matrix(
            (rows[:, 2], (rows[:, 1] - low, rows[:, 0] - 1)), shape=(high - low + 1, n_genes)
        )
        depth = np.asarray(x.sum(axis=1)).ravel()
        hits = np.diff(x.indptr)
        qc = (
            (depth >= 1500)
            & (hits >= 700)
            & (np.asarray(x[:, mito].sum(axis=1)).ravel() <= 0.2 * depth)
        )
        n_qc += int(qc.sum())
        pos = np.flatnonzero(qc)
        if not len(pos):
            continue
        markers = pd.DataFrame({g: x[pos][:, idx].toarray().ravel() for g, idx in indices.items()})
        keep = np.flatnonzero(markers[IDENTITY].gt(0).sum(axis=1).to_numpy() >= 2)
        if not len(keep):
            continue
        pos = pos[keep]
        marker_log = np.log1p(markers.iloc[keep].to_numpy() * (1e4 / depth[pos, None]))
        obs = pd.DataFrame(
            marker_log,
            columns=markers.columns,
            index=[gsm + "|" + b for b in barcodes[pos + low - 1]],
        )
        obs["depth"] = depth[pos]
        obs["neuronal_score"] = obs[NEURON].mean(axis=1)
        obs["cycling_score"] = obs[CYCLING].mean(axis=1)
        obs["balance"] = obs.neuronal_score - obs.cycling_score
        obs["tumor"], obs["condition"] = library, condition
        obs_parts.append(obs)
        matrices.append(x[pos])
    obs = pd.concat(obs_parts)
    assert obs.index.is_unique
    selected = np.sort(
        np.random.default_rng(int(gsm[3:])).choice(len(obs), min(cap, len(obs)), replace=False)
    )
    sample = ad.AnnData(
        sparse.vstack(matrices, format="csr")[selected].astype("float32"),
        obs=obs.iloc[selected].copy(),
        var=features.set_index("id"),
    )
    audit = dict(
        tumor=library,
        condition=condition,
        barcodes=len(barcodes),
        pass_qc=n_qc,
        neural_profiles=len(obs),
        modeled=len(selected),
    )
    return sample, obs, audit
