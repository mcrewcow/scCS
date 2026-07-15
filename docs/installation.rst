Installation
============

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
