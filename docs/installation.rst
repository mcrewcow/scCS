Installation
============

Source candidate tutorials
--------------------------

The 0.8.1rc2 source archive contains ``INSTALL_CONDA.md``,
``constraints-tutorials.txt`` and ``environment-tutorials.yml``. From its root:

.. code-block:: bash

   conda create -n sccs-081rc2 -c conda-forge python=3.12 pip -y
   conda activate sccs-081rc2
   python -m pip install -c constraints-tutorials.txt -e ".[tutorials]"
   python -m ipykernel install --user --name sccs-081rc2 --display-name "scCS 0.8.1rc2"
   jupyter lab tutorials/

Use the named kernel. The tested scVelo 0.3.4 path requires pandas<3.
The Schwann data adapter does not require the RegVelo modeling package.
The first model fit is saved for reuse across notebooks; setting
``SCCS_RECOMPUTE_VELOCITY=1`` forces a fresh fit.

Base installation
-----------------

.. code-block:: bash

   pip install scCS-py

RNA velocity
------------

.. code-block:: bash

   pip install "scCS-py[velocity]"

All optional analysis features
------------------------------

.. code-block:: bash

   pip install "scCS-py[all]"

Development and documentation
-----------------------------

.. code-block:: bash

   pip install -e ".[dev]"

The future-fate solver uses the transition matrix already present in the
AnnData object or a transition matrix supplied directly. scVelo is needed only
when scCS is asked to obtain or calculate RNA-velocity quantities itself.
