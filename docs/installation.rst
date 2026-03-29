Installation
============

From source (recommended)
-------------------------

.. code-block:: bash

   git clone https://github.com/zerotonin/satyre.git
   cd satyre
   pip install -e ".[dev,docs]"

Conda environment
-----------------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate satyre
   pip install -e .

Dependencies
------------

SATYRE requires Python ≥ 3.10 and the following packages:

- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7
- Pandas ≥ 2.0
- Seaborn ≥ 0.12
- Shapely ≥ 2.0
