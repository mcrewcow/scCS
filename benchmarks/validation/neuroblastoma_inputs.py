"""Audit GSE288099 inputs and run velocyto only on validated tagged BAM files.

A gene-expression MTX or a Seurat export cannot recreate spliced/unspliced
counts. Original PIP-seq reads need chemistry-aware barcode/UMI processing and
alignment. This script never treats expression counts as velocity counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import pandas as pd


def public_manifest():
    rows = []
    conditions = ["CTRL", "CTRL", "RA", "RA", "NEN", "NEN", "RA+NEN", "RA+NEN"]
    libraries = ["C1", "C2", "R1", "R2", "N1", "N2", "RN1", "RN2"]
    for i, (condition, library) in enumerate(zip(conditions, libraries)):
        gsm = f"GSM{8758977 + i}"
        rows.append(
            {
                "sample": gsm,
                "library": library,
                "condition": condition,
                "biological_replicate": i % 2 + 1,
                "catalog_audited": "2026-11-17",
                "public_processed_format": "MTX/TSV gene-expression counts",
                "verified_public_velocity_layers": False,
                "status": "NEEDS_TAGGED_BAM_OR_AUTHOR_VELOCITY_COUNTS",
                "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288099",
                "sample_source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + gsm,
            }
        )
    return pd.DataFrame(rows)


def audit_bam(path, barcodes, gtf, n_reads=10000):
    if Path(path).suffix.lower() not in {".bam", ".sam"}:
        raise ValueError("Provide aligned SAM/BAM, not an expression matrix or h5ad.")
    for p in [path, barcodes, gtf]:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    try:
        import pysam
    except ImportError as exc:
        raise ImportError("BAM inspection requires pysam: pip install pysam") from exc
    mapped = 0
    tagged = 0
    with pysam.AlignmentFile(path, "rb" if str(path).endswith(".bam") else "r") as bam:
        references = bam.references
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            mapped += 1
            tagged += int(read.has_tag("CB") and read.has_tag("UB"))
            if mapped >= n_reads:
                break
    if not mapped or tagged / mapped < 0.5:
        raise ValueError(
            "The BAM lacks adequate CB/UB-tagged reads. Process original PIP-seq "
            "barcodes/UMIs first; do not assume the 10x read layout."
        )
    return {
        "mapped_reads_checked": mapped,
        "CB_UB_tagged_fraction": tagged / mapped,
        "reference_count": len(references),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tutorial_outputs/neuroblastoma_inputs.csv")
    parser.add_argument("--bam")
    parser.add_argument("--barcodes")
    parser.add_argument("--gtf")
    parser.add_argument("--sample", default="neuroblastoma")
    parser.add_argument("--output", default="tutorial_outputs/neuroblastoma_velocity")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = public_manifest()
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(manifest[["sample", "condition", "status"]].to_string(index=False))
    if args.bam:
        if not args.barcodes or not args.gtf:
            parser.error("--bam also requires --barcodes and --gtf")
        print(audit_bam(args.bam, args.barcodes, args.gtf))
        command = [
            "velocyto",
            "run",
            "-b",
            args.barcodes,
            "-o",
            args.output,
            "--sampleid",
            args.sample,
            args.bam,
            args.gtf,
        ]
        print("Command arguments:", command)
        if args.execute:
            if shutil.which("velocyto") is None:
                raise RuntimeError("Install the velocyto CLI before requesting --execute.")
            subprocess.run(command, check=True)
    elif args.execute:
        parser.error("--execute requires validated aligned inputs; no BAM was supplied")


if __name__ == "__main__":
    main()
