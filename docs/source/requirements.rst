Requirements
============

Python
------

scCS requires **Python ≥ 3.9**.

Installation
------------

From GitHub (latest)::

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
   * - scvelo
     - ≥ 0.3
     - RNA velocity computation
   * - statsmodels
     - ≥ 0.14
     - LOWESS smoothing in expression trends
   * - scikit-learn
     - ≥ 1.2
     - GMM fate detection, clustering

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Package
     - Version
     - Purpose
   * - gseapy
     - ≥ 1.0
     - Pathway enrichment (KEGG, GO BP, Reactome)
   * - cellrank
     - ≥ 2.0
     - CellRank-based fate detection
   * - leidenalg
     - any
     - Leiden clustering for fate detection

Install optional dependencies::

    pip install gseapy          # pathway enrichment
    pip install cellrank        # CellRank fate detection

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
