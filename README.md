# SATYRE — Sensory locomotion str**AT**eg**Y** f**R**am**E**work

[![Tests](https://github.com/zerotonin/satyre/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/satyre/actions/workflows/tests.yml)
[![Documentation](https://github.com/zerotonin/satyre/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/satyre)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/578341364.svg)](https://doi.org/10.5281/zenodo.19326027)

SATYRE simulates and analyses the exploratory behaviour of *Drosophila
melanogaster* under visual deprivation.  It provides Markov-chain replay of
empirical locomotion data, Lévy-flight and random-walk reference agents, a
toroidal arena with food-collection mechanics, and post-simulation analysis
tools for exploration rate, drift angles, step-size distributions, and
foraging efficiency.

The code accompanies the paper:

> **Tactile space expansion in vision-deprived flies**
> Kristina Corthals, Irene M. Aji, Miriam Berger, Philip Hehlert,
> Jonas Albers, Christian Dullin, Heribert Gras, Martin C. Göpfert,
> Naoyuki Fuse & Bart R.H. Geurten
> *Proceedings of the Royal Society B* (submitted)

---

## Highlights

- **Markov walker** — replays empirical prototypical-movement (PM) sequences
  via a first-order Hidden Markov Model fitted to high-speed tracking data
  (500 fps) of dark-fly and OregonR strains.
- **Lévy / random-walk agents** — provides baseline foraging models with
  heavy-tailed (Cauchy) or Gaussian step-size distributions.
- **Toroidal arena** — the `HyperspaceSolver` wraps trajectories across
  arena boundaries, preserving direction and remaining step length.
- **Tactile & visual detection** — food items are collected by body-overlap
  (dark condition) or within a configurable visual field (light condition).
- **Analysis suite** — orientation-dependent vs. disc-based exploration
  rates, movement-type classification, drift-angle (ψ) statistics, and
  bootstrap confidence intervals on foraging efficiency.

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/zerotonin/satyre.git
cd satyre
pip install -e ".[dev,docs]"
```

### Conda

```bash
conda env create -f environment.yml
conda activate satyre
pip install -e .
```

---

## Quick start

### 1. Preprocess raw data

Convert experimental PM index sequences and velocity tables into Markov
transition matrices:

```bash
satyre-preprocess --idx data/ORL_IDX.txt \
                  --velo data/ORL_C.txt  \
                  --out  data/preprocessed
```

### 2. Run a Markov-driven simulation

```python
import numpy as np
from satyre import MarkovWalker

trans  = np.load("data/preprocessed/cumsum_transition_matrix.npy")
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
```

### 3. Run a Lévy-flight baseline

```python
from satyre import LevyWalker

walker = LevyWalker(
    cauchy_alpha=1.5,
    mode="cauchy",
    max_steps=150_000,
    n_trials=100,
)
results = walker.simulate_multiple()
```

### 4. Analyse results

```python
from satyre.analysis import summarise_food_collected, step_size_distribution

print(summarise_food_collected(results))
print(step_size_distribution(results["step_sizes"]))
```

---

## Project structure

```
satyre/
├── satyre/
│   ├── __init__.py                 # public API
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── markov_walker.py        # HMM-driven agent
│   │   └── levy_walker.py          # Lévy / random-walk agent
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── area_covered.py         # exploration-rate metrics
│   │   ├── food_found.py           # foraging-efficiency stats
│   │   ├── step_size.py            # step-length distributions
│   │   └── movement_types.py       # movement classification
│   └── utils/
│       ├── __init__.py
│       ├── hyperspace_solver.py    # toroidal boundary wrap
│       └── preprocessing.py        # raw data → transition matrices
├── tests/
│   └── test_core.py
├── docs/                           # Sphinx documentation
├── examples/
│   └── five_strategy_comparison.py
├── .github/workflows/
│   ├── tests.yml                   # CI: pytest across Python versions
│   └── docs.yml                    # CD: Sphinx → GitHub Pages
├── pyproject.toml
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Running the tests

```bash
python -m pytest -v
```

---

## Documentation

Full API documentation is built with Sphinx and deployed automatically to
GitHub Pages on every push to `main`:

📖 **https://zerotonin.github.io/satyre**

To build locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

---

## Authors

| Name | Affiliation |
|------|-------------|
| **Irene M. Aji** | Georg-August-University Göttingen, Dept. of Cellular Neuroscience |
| **Bart R.H. Geurten** | University of Otago, Dept. of Zoology, Dunedin, New Zealand |

---

## Citing SATYRE

If you use this software, please cite:

```bibtex
@software{satyre,
  author    = {Aji, Irene M. and Geurten, Bart R.H.},
  title     = {{SATYRE} -- Sensory locomotion strATegY fRamEwork},
  year      = {2026},
  url       = {https://github.com/zerotonin/satyre},
  license   = {MIT},
}
```

and the accompanying paper:

```bibtex
@article{corthals2026tactile,
  author  = {Corthals, Kristina and Aji, Irene M. and Berger, Miriam
             and Hehlert, Philip and Albers, Jonas and Dullin, Christian
             and Gras, Heribert and G\"{o}pfert, Martin C. and Fuse, Naoyuki
             and Geurten, Bart R.H.},
  title   = {Tactile space expansion in vision-deprived flies},
  journal = {Proceedings of the Royal Society B},
  year    = {2026},
}
```

---

## License

[MIT](LICENSE) © 2018–2026 Irene M. Aji & Bart R.H. Geurten
