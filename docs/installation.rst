Installation
===============

Package installation
-----------------------

.. code-block:: bash

   pip install scCS-py

RNA-velocity utilities require the optional scVelo dependency:

.. code-block:: bash

   pip install "scCS-py[velocity]"

For all optional analysis features:

.. code-block:: bash

   pip install "scCS-py[all]"

Tutorial environment
-----------------------

Clone or download the source repository, then open a terminal in the directory
containing ``pyproject.toml``. The supplied environment reproduces the tested
Python 3.12 scientific stack:

.. code-block:: bash

   conda env create -f environment-tutorials.yml
   conda activate sccs-100
   python -m ipykernel install --user --name sccs-100 --display-name "scCS 1.0.0"
   jupyter lab tutorials/

Alternatively, create an environment and install the tutorial dependencies with
the supplied constraints:

.. code-block:: bash

   conda create -n sccs-100 -c conda-forge python=3.12 pip -y
   conda activate sccs-100
   python -m pip install -c constraints-tutorials.txt -e ".[tutorials]"
   python -m ipykernel install --user --name sccs-100 --display-name "scCS 1.0.0"
   jupyter lab tutorials/

Select the **scCS 1.0.0** kernel. Check the imported package before starting:

.. code-block:: bash

   python -c "import scCS; print(scCS.__version__); print(scCS.__file__)"

The version should be ``1.0.0`` and the path should point into your source
checkout. The editable install uses that checkout when you restart the kernel.
Conda already isolates this environment; a nested virtual environment is unnecessary.
The tested scVelo 0.3.4 path requires ``pandas<3``. The Schwann data adapter
does not require installing the RegVelo modeling package.

Data downloads require internet access. Run one biological notebook kernel at
a time to avoid retaining multiple full datasets in memory. The first dynamical
velocity fit can take tens of minutes; later notebooks reuse matching fits from
``data/scCS_cache/`` relative to the notebook working directory. Keep running
notebooks from the same directory to share those caches.

Set ``SCCS_RECOMPUTE_VELOCITY=1`` before launching Jupyter to force a fresh fit.
``SCCS_N_JOBS`` controls parallel workers and defaults to one. The notebooks
retain complete fits in the preprocessing cache and release unused kinetic
working layers during downstream analysis.

See :doc:`tutorials` for the recommended run order and :doc:`scalability` for
memory requirements. The standard scalability profile stops at one million
cells; the exact 200-million-cell profile requires a separate high-memory host.

Development and documentation
--------------------------------

.. code-block:: bash

   pip install -e ".[dev]"

The future-fate solver uses the transition matrix already present in the
AnnData object or a transition matrix supplied directly. scVelo is needed only
when scCS is asked to obtain or calculate RNA-velocity quantities itself.

Validation and benchmark notebooks
-------------------------------------

The validation notebooks use the real installed package and fixed random seeds.
The real-data notebooks download versioned public data outside the source tree.
To include the optional CellRank comparisons and BAM input audit:

.. code-block:: bash

   python -m pip install -e ".[tutorials,benchmarks]"
   python -m ipykernel install --user --name sccs-validation --display-name "scCS validation"

Select the **scCS validation** kernel. See :doc:`tutorials_pair` for real-count
workflows and :doc:`tutorials_validation`
for the non-trivial questions, assumptions and interpretation guides. The LARRY
count deposit is approximately 1.53 GB and is downloaded once. Its genuine
spliced/unspliced layers are exported to a verified analysis-subset loom locally;
neither the count deposit nor generated data files are bundled into scCS.
A velocyto CLI and correctly aligned inputs are needed separately to generate
new neuroblastoma loom files; the benchmark extra does not create those inputs.
