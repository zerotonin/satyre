Below is a comprehensive README template for your GitHub repository. You can customize it further as needed.

---

# Satyre: A Simulation Framework for Fly Movement Analysis

Satyre (*S*ensory locomotion str*AT*eg*Y* f*R*am*E*work) is a simulation and analysis framework designed to model and investigate the exploratory behavior of flies. It incorporates experimental data to generate probabilistic models of prototypical movements and simulates fly trajectories under various conditions, including different light environments and movement strategies. The framework also provides a suite of analysis tools to evaluate key metrics such as food collection efficiency, area covered by the mechanosensory field, step–size distributions, and detailed movement statistics.

## Overview

The Satyre module integrates several components:

- **Data Preprocessing:**  
  The `rawData_preprocessing.py` script processes raw experimental data (prototypical movement indices and velocity recordings) to generate matrices of transition probabilities. These matrices are then used as input for the simulation models.

- **Simulation Models:**  
  Satyre offers multiple simulation approaches:
  - **Dark Fly Simulations:**  
    `walkSimOO_updated_darkFly_lightCondition_multiTrials_v2.py` and  
    `walkSimOO_updated_darkFly_multiTrials_v2.py` implement simulations based on experimental dark–fly and Oregon R (ORL) movement data.
  - **Levy Flight Simulations:**  
    `walkSimOO_updated_LevyRandom_v3.py` simulates a Levy flight search strategy based on heavy-tailed step distributions.

- **Analysis Tools:**  
  A variety of scripts are provided to analyze simulation outputs:
  - **Area Covered:** `analysis_areaCovered.py`
  - **Food Collection Efficiency:** `analysis_foodFound.py` and `analysis_PMbased_FoodFound.py`
  - **Movement Types:** `analysis_movementTypes.py`
  - **Step Sizes and Velocities/Angles:** `analysis_PMbased_StepsVeloAngles.py`, `analysis_stepSize.py`, and `analysis_velocitiesAngles.py`

- **Utilities and Comparison Scripts:**  
  - **Boundary Handling:**  
    `hyperSpaceSolver_updated.py` implements a hyperspace solver that re-maps trajectories that exit the simulation arena so that the fly re-enters from the opposite side.
  - **Examples and Comparisons:**  
    `PMexamples.py` provides quick examples of prototypical movement velocities, while `comparisons_v2.py` and `comparisons_v3.py` offer comprehensive comparisons between different movement models.

## Features

- **Flexible Simulation Options:**  
  Simulate fly trajectories using different strategies (e.g., random walk, ORL-based Levy, dark–fly-based Levy) under various conditions (light vs. dark).
  
- **Robust Data Analysis:**  
  Evaluate metrics such as the area covered by the fly’s sensory field, food collection efficiency, and detailed step–size/velocity/angle distributions.
  
- **Accurate Boundary Handling:**  
  The hyperspace solver ensures realistic handling of trajectories that cross the virtual arena's boundaries.
  
- **Statistical Comparisons and Visualizations:**  
  Built-in scripts generate publication–quality plots (using Matplotlib, Pandas, and Seaborn) and perform statistical tests to compare simulation results.

## Requirements

- Python 3.x
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Shapely](https://pypi.org/project/Shapely/)

Install dependencies via pip:

```bash
pip install numpy scipy matplotlib pandas seaborn shapely
```

## Installation

Clone the repository from GitHub:

```bash
git clone https://github.com/YourUsername/satyre.git
cd satyre
```

## Directory Structure

```
satyre/
├── README.md
├── rawData_preprocessing.py
├── walkSimOO_updated_darkFly_lightCondition_multiTrials_v2.py
├── walkSimOO_updated_darkFly_multiTrials_v2.py
├── walkSimOO_updated_LevyRandom_v3.py
├── analysis/
│   ├── analysis_areaCovered.py
│   ├── analysis_foodFound.py
│   ├── analysis_movementTypes.py
│   ├── analysis_PMbased_FoodFound.py
│   ├── analysis_PMbased_StepsVeloAngles.py
│   ├── analysis_stepSize.py
│   └── analysis_velocitiesAngles.py
├── utilities/
│   ├── hyperSpaceSolver_updated.py
│   ├── PMexamples.py
│   ├── comparisons_v2.py
│   └── comparisons_v3.py
```

- **Data Preprocessing:**  
  `rawData_preprocessing.py` converts raw text files of movement data into matrices that are used by the simulation modules.

- **Simulation Scripts:**  
  The `walkSimOO_updated_*` scripts simulate fly trajectories using either dark–fly or ORL movement profiles and incorporate boundary handling via the hyperspace solver.

- **Analysis Scripts:**  
  Scripts in the `analysis/` directory compute and visualize metrics such as the area covered, food found, movement types, step sizes, and velocity/angle distributions.

- **Utilities:**  
  The `utilities/` folder contains:
  - `hyperSpaceSolver_updated.py`: Handles re–entry of trajectories that exit the simulation arena.
  - `PMexamples.py`: Provides examples for prototypical movement (PM) velocity data.
  - `comparisons_v2.py` and `comparisons_v3.py`: Compare performance across different simulation models.

## Usage

### Running Simulations

1. **Preprocess Raw Data:**  
   Run the preprocessing script to generate the necessary probability matrices and velocity arrays:
   ```bash
   python rawData_preprocessing.py
   ```

2. **Run a Simulation:**  
   For example, to simulate fly movement under light conditions using dark–fly data:
   ```bash
   python walkSimOO_updated_darkFly_lightCondition_multiTrials_v2.py
   ```
   Make sure the required preprocessed data is available in the expected paths.

### Data Analysis

Run any of the analysis scripts to generate plots and statistics. For example:
```bash
python analysis/analysis_areaCovered.py
```
This script will compute the area covered by the fly’s mechanosensory field and generate comparative boxplots.

### Comparisons

To compare different simulation models, use:
```bash
python utilities/comparisons_v2.py
# or
python utilities/comparisons_v3.py
```
These scripts aggregate and visualize data from multiple simulation runs.

### Example: Visualizing Prototypical Movements

Run the example script to see a bar chart of PM velocities:
```bash
python utilities/PMexamples.py
```

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## Authors

- **Irene Aji**
- **Dr. Bart Geurten**

## License

[Specify your license here, e.g., MIT License]

