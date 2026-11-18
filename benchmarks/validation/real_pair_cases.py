"""Load published velocity layers and construct the two real-data PairScorer cases."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

HASHES = json.loads(Path(__file__).with_name("pair_case_checksums.json").read_text())
SEED = 19
IDENTITY = ["Cd79a", "Cd79b", "Ms4a1", "Cd19", "Cd74"]
ACTIVATION = ["Cd69", "Cd83", "Cd86", "Myc", "Nfkbia"]
SECRETORY = ["Jchain", "Mzb1", "Sdc1", "Tnfrsf17", "Derl3"]
REGULATORS = ["Irf4", "Prdm1", "Xbp1"]
HELD_OUT = ["Ell2", "Sec11c", "Hspa5", "Aicda", "Bcl6"]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def geo_file(directory, name):
    """Download atomically and validate the uncompressed GEO file, including cached copies."""
    path = Path(directory) / name
    if path.exists():
        if digest(path) != HASHES[name]:
            raise ValueError(f"Checksum mismatch for {path}")
        return path
    accession = name.split("_")[0]
    kind = "samples" if accession.startswith("GSM") else "series"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/{kind}/{accession[:-3]}nnn/{accession}/suppl/{name}.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, gzip.GzipFile(fileobj=response) as src, temp.open("wb") as dest:
            shutil.copyfileobj(src, dest)
        if digest(temp) != HASHES[name]:
            raise ValueError(f"Checksum mismatch for {url}")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def chromaffin_metadata(directory):
    path = geo_file(directory, "GSE204700_adata_ChC.h5ad")
    with h5py.File(path) as h:
        obs = h["obs"]
        table = pd.DataFrame(index=obs["_index"].asstr()[:])
        for key in ["annot", "location", "devtime", "CytoTRACE"]:
            if key in obs["__categories"]:
                table[key] = obs["__categories"][key].asstr()[:][obs[key][:]]
            else:
                table[key] = obs[key][:]
    assert table.index.is_unique
    return table


def b_cell_metadata(h, names, ids):
    """Call disjoint transcript-based proxies before sampling or reading condition."""
    spliced = h["layers/spliced"]
    depth = np.zeros(len(ids), dtype=np.int64)
    detected = depth.copy()
    mito = depth.copy()
    symbols = IDENTITY + ACTIVATION + SECRETORY + REGULATORS + ["Cd40"] + HELD_OUT
    expr = pd.DataFrame(0, index=ids, columns=symbols, dtype=np.int64)
    if set(symbols) - set(names):
        raise ValueError("Missing markers: " + str(set(symbols) - set(names)))
    for start in range(0, len(names), 512):
        genes = names[start:start + 512]
        chunk = spliced[start:start + 512]
        depth += chunk.sum(axis=0, dtype=np.int64)
        detected += np.count_nonzero(chunk, axis=0)
        mito += chunk[np.char.startswith(genes.astype(str), "mt-")].sum(axis=0, dtype=np.int64)
        for symbol in set(genes).intersection(symbols):
            expr[symbol] += chunk[genes == symbol].sum(axis=0, dtype=np.int64)
    positive = expr.gt(0)
    b = positive[IDENTITY].sum(axis=1)
    secretory = positive[SECRETORY].sum(axis=1)
    plasma = secretory.ge(2) & positive[REGULATORS].any(axis=1)
    pre = positive.Irf4 & positive.Prdm1 & ~positive[["Cd40","Sdc1","Tnfrsf17"]].any(axis=1) & secretory.le(1)
    activated = b.ge(2) & positive[ACTIVATION].sum(axis=1).ge(2) & ~positive.Prdm1 & secretory.le(1)
    qc = (detected >= 300) & (mito <= .2 * depth)
    states = np.select([plasma, pre, activated], ["Plasma-like", "Pre-plasma-like", "Activated B"], default="Other B")
    obs = pd.DataFrame({"state": states, "keep": qc & (b.ge(2) | plasma | pre),
                        "spliced_depth": depth, "detected_genes": detected}, index=ids)
    obs["strict_plasma"] = plasma & positive.Prdm1 & positive[["Jchain", "Sdc1"]].any(axis=1)
    for key in HELD_OUT:
        obs[key] = positive[key]
    return obs


def count_layers(h, positions):
    layers = {}
    for name in ("spliced", "unspliced"):
        dataset = h["layers/" + name]
        blocks = [sparse.csr_matrix(dataset[i:i+512][:, positions], dtype=np.float32)
                  for i in range(0, dataset.shape[0], 512)]
        layers[name] = sparse.vstack(blocks).T.tocsr()
    return layers


def load_case(case, directory):
    """Return raw subset, full input inventory, and pre-sampling state counts."""
    if case not in {"chromaffin", "bcell"}:
        raise ValueError(case)
    adrenal = case == "chromaffin"
    meta = chromaffin_metadata(directory) if adrenal else None
    prefix = "GSM618" if adrenal else "GSM724"
    parts, rows, input_rows, observed_ids = [], [], [], []
    rng = np.random.default_rng(SEED)
    reference_genes = None
    for file in sorted(k for k in HASHES if k.startswith(prefix)):
        accession = file.split("_")[0]
        with h5py.File(geo_file(directory, file)) as h:
            genes = h["row_attrs/Gene"].asstr()[:]
            stable = h["row_attrs/Accession"].asstr()[:]
            if reference_genes is None:
                reference_genes = (stable, genes)
            else:
                for a, b in zip(reference_genes, (stable, genes)):
                    np.testing.assert_array_equal(a, b)
            ids = pd.Index(h["col_attrs/CellID"].asstr()[:])
            if adrenal:
                ids = ids.str.replace("_R1_unique.bam", "", regex=False)
                match = ids.intersection(meta.index)
                observed_ids.extend(match)
                obs = meta.reindex(ids).copy()
                obs["state"] = obs.annot
                obs["condition"] = np.where(obs.devtime.eq("E16.5"), "E16.5", "E12.5-E14.5")
                obs["keep"] = obs.location.eq("Adrenal") & obs.devtime.isin(["E12.5","E13.5","E14.5","E16.5"])
            else:
                obs = b_cell_metadata(h, genes, ids)
                obs["condition"] = "CAG-sHBs" if int(accession[3:]) <= 7245823 else "WT"
            obs["library"] = accession
            eligible = obs.loc[obs.keep].copy()
            rows.append(eligible)
            selected = eligible.index
            if not adrenal:
                sampled = []
                for state, group in eligible.groupby("state", sort=True):
                    cap = {"Activated B": 150, "Pre-plasma-like": 120, "Plasma-like": 120, "Other B": 250}[state]
                    sampled.extend(rng.choice(group.index, min(cap, len(group)), replace=False))
                selected = pd.Index(sampled)
            positions = np.flatnonzero(ids.isin(selected))
            input_rows.append((accession,len(ids),len(eligible),len(positions)))
            if len(positions):
                layers = count_layers(h, positions)
                part = ad.AnnData(X=layers["spliced"].copy(), layers=layers, obs=obs.iloc[positions].copy(),
                                  var=pd.DataFrame({"symbol": genes}, index=stable))
                parts.append(part)
    if adrenal:
        assert len(observed_ids) == len(set(observed_ids)) == len(meta)
    obj = ad.concat(parts, merge="same")
    assert obj.obs_names.is_unique
    obj.var_names = obj.var.symbol.astype(str)
    obj.var_names_make_unique()
    obj.var.index.name = "feature"
    inventory = pd.DataFrame(input_rows, columns=["library","loom_cells","eligible","modeled"])
    return obj, inventory, pd.concat(rows)


def model_genes(obj, n_genes=1500):
    """Rank measured, shared genes on spliced overdispersion, excluding Ig, ribosomal and mitochondrial genes."""
    s, u = obj.layers["spliced"], obj.layers["unspliced"]
    mean = np.asarray(s.mean(axis=0)).ravel()
    variance = np.asarray(s.power(2).mean(axis=0)).ravel() - mean**2
    coverage = (np.asarray((s > 0).sum(axis=0)).ravel() >= 10) & (np.asarray((u > 0).sum(axis=0)).ravel() >= 10)
    exclude = obj.var.symbol.str.match(r"^(mt-|Rp[ls][0-9]|Ig[hkl][vdjcmgaek])", case=False).to_numpy()
    candidates = np.flatnonzero(coverage & ~exclude)
    top = candidates[np.argsort(-(variance[candidates] / (mean[candidates] + 1)), kind="stable")[:n_genes]]
    return obj[:, np.sort(top)].copy()
