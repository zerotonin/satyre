Quickstart
==========

Preprocessing raw data
----------------------

Convert experimental prototypical-movement (PM) sequences
and velocity tables into the Markov transition matrices
required by the simulator:

.. code-block:: bash

   satyre-preprocess --idx data/ORL_IDX.txt \
                     --velo data/ORL_C.txt  \
                     --out  data/preprocessed

This creates three ``.npy`` files in ``data/preprocessed/``.

Running a Markov-driven simulation
----------------------------------

.. code-block:: python

   import numpy as np
   from satyre import MarkovWalker

   trans = np.load("data/preprocessed/cumsum_transition_matrix.npy")
   pm_idx = np.load("data/preprocessed/pm_index_matrix.npy")
   velos  = np.load("data/preprocessed/velocity_array.npy")

   walker = MarkovWalker(
       trans, pm_idx, velos,
       max_steps=150_000,
       n_trials=100,
       food_mode="random",
   )
   results = walker.simulate_multiple()
   print(f"Mean food found: {results['food_found'].mean():.1f}")

Running a Lévy-flight simulation
---------------------------------

.. code-block:: python

   from satyre import LevyWalker

   walker = LevyWalker(
       cauchy_alpha=1.5,
       mode="cauchy",
       max_steps=150_000,
       n_trials=100,
   )
   results = walker.simulate_multiple()

Analysing results
-----------------

.. code-block:: python

   from satyre.analysis import (
       summarise_food_collected,
       step_size_distribution,
   )

   print(summarise_food_collected(results))
   print(step_size_distribution(results["step_sizes"]))
