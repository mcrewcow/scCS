# Install the scCS 0.8.1rc2 tutorial environment

Unzip this source archive and open a terminal in the extracted `scCS-0.8.1rc2` folder (the folder containing `pyproject.toml`). Use a fresh conda environment; conda already provides the isolated environment, so a nested Python venv is unnecessary.

```bash
conda create -n sccs-081rc2 -c conda-forge python=3.12 pip -y
conda activate sccs-081rc2
python -m pip install --upgrade pip "setuptools>=77.0.3" wheel
python -m pip install -c constraints-tutorials.txt -e ".[tutorials]"
python -m ipykernel install --user --name sccs-081rc2 --display-name "scCS 0.8.1rc2"
python -c "import scCS; print(scCS.__version__); print(scCS.__file__)"
python -m pip check
jupyter lab tutorials/
```

Check that the printed version is `0.8.1rc2`, its path points into this extracted source folder, and each notebook uses the **scCS 0.8.1rc2** kernel. The editable install makes later local source edits available after restarting the kernel. This archive is the release candidate; a plain `pip install scCS-py` may install a different published version.

The equivalent environment-file route, from this same source folder, is:

```bash
conda env create -f environment-tutorials.yml
conda activate sccs-081rc2
python -m ipykernel install --user --name sccs-081rc2 --display-name "scCS 0.8.1rc2"
jupyter lab tutorials/
```

Run 01, 02, then 03–06, 07–09, 10–11. Run one biological notebook kernel at a time to avoid retaining several full datasets in memory. Notebook 12 defaults to complete problems up to one million cells; `SCCS_SCALABILITY_PROFILE=extended` enables the full size ladder. The exact 200-million-cell profile requires a separate high-memory machine. Do not enable that profile on a normal workstation.

All notebooks were updated, synchronized with their documentation copies, and include reviewed executed outputs. Public data downloads need internet access. The Schwann dataset is loaded by `scCS.datasets.schwann`; installing the RegVelo modeling stack is unnecessary for these scVelo tutorials.

The biological notebooks share versioned preprocessing caches under `data/scCS_cache/` relative to the notebook working directory. The first dynamical fit can take tens of minutes; later notebooks reuse the fit when the package, parameters and scientific dependency versions match. Pseudo-condition perturbations are applied after loading the unperturbed cache. Keep running the notebooks from the same folder to share those caches. Set `SCCS_RECOMPUTE_VELOCITY=1` before launching Jupyter to force a fresh fit. `SCCS_N_JOBS` controls scVelo parallel jobs and defaults to 1.

The compatibility bound `pandas<3` is intentional: the tested scVelo 0.3.4 velocity path fails with pandas 3.1.1. The constraints above reproduce the reviewed Python 3.12 scientific stack. Other Python/platform combinations remain unverified.

After the tutorials, retain the output folders, printed scCS version and environment:

```bash
python -m pip freeze > sccs-tutorial-environment.txt
```

For optional developer checks:

```bash
python -m pip install -c constraints-tutorials.txt -e ".[dev,tutorials]"
python -m pytest -q
```

The notebooks retain the complete velocity fit in the shared preprocessing cache,
then release unused kinetic-fit working layers for downstream analysis. This reduces
AnnData copy costs while retaining expression, velocities, graphs and orderings.
