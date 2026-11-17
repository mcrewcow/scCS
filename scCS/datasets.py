"""Small public-data adapters; no RegVelo modeling installation is needed."""

from __future__ import annotations

from pathlib import Path
import hashlib

SCHWANN_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1ZpG1gz0jjPNuXc_bsz3XInVtHooHU02M&export=download&confirm=t"
)
SCHWANN_SHA256 = "a1915c47515b40f65609e42196c1d70a0f55495cbd637aa01e2c2b9b8a66e5eb"


def schwann(
    file_path="data/schwann/adata_schwann_preprocessed.h5ad", *, verify_checksum=True
):
    """Load the public mouse Schwann dataset used by RegVelo.

    The file and checksum match the dataset referenced by RegVelo 1.0.0's
    official dataset loader. This is a data adapter, not the RegVelo model.
    Existing files are verified by default so annotation changes caused by
    a different input file do not masquerade as a software result.
    """
    from scanpy import read

    path = Path(file_path)
    adata = read(path, backup_url=SCHWANN_URL, sparse=True, cache=False)
    if verify_checksum:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != SCHWANN_SHA256:
            raise ValueError(
                "Schwann dataset checksum does not match the tutorial reference. "
                "Use a fresh download, or pass verify_checksum=False deliberately "
                "when analyzing a different data version."
            )
    return adata
