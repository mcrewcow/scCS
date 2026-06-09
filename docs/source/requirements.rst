Requirements
============

Python
------

scCS requires **Python ≥ 3.9**.

Installation
------------

From PyPI (recommended)::

    pip install scCS-py

From GitHub (latest development version)::

    pip install git+https://github.com/mcrewcow/scCS.git

Development install (editable)::

    git clone https://github.com/mcrewcow/scCS.git
    cd scCS
    pip install -e ".[dev]"

Core Dependencies
-----------------

These are installed automatically with scCS:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Package
     - Version
     - Purpose
   * - numpy
     - ≥ 1.24
     - Array math, velocity projections
   * - scipy
     - ≥ 1.10
     - Statistical tests, sparse matrix support
   * - pandas
     - ≥ 1.5
     - Results tables, DataFrames
   * - matplotlib
     - ≥ 3.7
     - All visualizations
   * - seaborn
     - ≥ 0.12
     - Plot styling
   * - anndata
     - ≥ 0.9
     - AnnData object support
   * - scanpy
     - ≥ 1.9
     - Single-cell preprocessing, neighbors
   * - scikit-learn
     - ≥ 1.2
     - NN entropy computation, GMM fate detection
   * - statsmodels
     - ≥ 0.14
     - LOWESS smoothing in expression trends

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Package
     - Version
     - Purpose
   * - scvelo
     - ≥ 0.2.5
     - RNA velocity computation
   * - gseapy
     - ≥ 1.0
     - Pathway enrichment (KEGG, GO BP, Reactome)
   * - cellrank
     - ≥ 2.0
     - CellRank-based fate detection

Install optional dependencies::

    pip install scCS-py[velocity]     # scvelo
    pip install scCS-py[enrichment]   # gseapy
    pip install scCS-py[all]          # everything

Docs Dependencies
-----------------

To build the documentation locally::

    pip install -e ".[docs]"

Includes: ``sphinx``, ``sphinx-autoapi``, ``furo``, ``nbsphinx``, ``ipykernel``.

Tested Environments
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - OS
     - Python
     - Notes
   * - Ubuntu 22.04
     - 3.9, 3.11
     - Primary development environment
   * - macOS 14
     - 3.10, 3.11
     - Tested
   * - Windows 11
     - 3.10
     - Via Anaconda
