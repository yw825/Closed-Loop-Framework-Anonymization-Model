# Closed-Loop Framework Anonymization Model

## Overview
This repository implements a closed-loop optimization framework 
to balance privacy protection and machine learning utility. This framework directly uses machine learning performance (classification loss) to iteratively refine the strategies of anonymization.

## Files
- `model_train.py` – Machine learning model training procedures
- `particle_swarm.py` – PSO optimization implementation
- `utils.py` – Helper functions
- `constants.py` – Datasets and Global parameters
- `closed_loop.ipynb` – Experiment notebook to implement the closed-loop framework

## Usage
Before running any experiment, you must first configure `constants.py`.

This file defines:

- The datasets used in the framework
- The hyperparameters associated with each dataset
- The experiment output directories

You **must configure `DATASET_CONFIGS` and `EXPERIMENT_SOURCES` before running the framework**.

---

### 1️⃣ Configure `DATASET_CONFIGS`

`DATASET_CONFIGS` is a dictionary that specifies all datasets used in the closed-loop framework.

You may include **one or multiple datasets**. Each dataset is defined as a dictionary containing the following fields:

### Required Fields

- **`path`**  
  Path to the dataset file.

- **`NQIs`**  
  List of numeric quasi-identifiers (QIs).

- **`CQIs`**  
  List of categorical quasi-identifiers (QIs).

- **`SAs`**  
  List of sensitive attributes (SAs).

- **`n_clusters`**  
  Number of clusters used in the closed-loop framework.  
  This is a hyperparameter and may contain **multiple values** (e.g., `[2, 3, 4]`).

  Since the optimal number of clusters depends strongly on dataset size and structure, the framework iterates over these values during experiments. It is therefore recommended to define cluster values separately for each dataset.

- **`l`**  
  Parameter for *l*-diversity.

  - If the sensitive attribute has **more than two distinct values**, set:
    ```
    l = 2
    ```
  - If the sensitive attribute has only **two distinct values**, set:
    ```
    1 < l < 2
    ```

### 2️⃣ Configure `EXPERIMENT_SOURCES`

`EXPERIMENT_SOURCES` defines where experiment results will be saved.

You must:
  - Define a base directory for storing results.
  - Define different experiment modes if needed.

The framework supports multiple experiment types, such as:
  - Closed-loop framework with repairing process
  - Closed-loop framework without repairing process

You can configure multiple experiment sources accordingly.

Now, you can use `closed_loop.ipynb` to run the closed-loop framework.

This framework integrates:

- Data anonymization (privacy protection)
- Machine learning evaluation (utility preservation)

The optimization process iteratively updates anonymization strategies based on machine learning performance, forming a closed-loop mechanism.

---

### 3️⃣ Configure `ML_models`
After importing the dataset you want to work with, you must specify the machine learning model(s) to be evaluated within the framework.

The repository provides a list called `ML_models`, where you can:

- Select one or multiple machine learning models
- Configure model-specific hyperparameters

Each model in `ML_models` should include:

- The model name
- The corresponding hyperparameters
- Any additional configuration required for training

This allows you to compare how different models perform under various anonymization strategies.

### 4️⃣ Configure `PSO_PARAMETERS`
`PSO_PARAMETERS` defines the hyperparameters for the [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (PSO) algorithm used in the closed-loop framework.

PSO searches for the optimal anonymization strategy by iteratively updating candidate solutions.

- **`n_population`**  
  Number of particles in the swarm.  

  Each particle represents a candidate solution.  
  A larger population increases exploration ability but also increases computational cost.

- **`maxIter`**  
  Maximum number of iterations for the PSO algorithm.  

  This determines how long the optimization process runs.

- **`n_bootstrap`**  
  Number of repeated runs for the machine learning model evaluation.

  Machine learning results can vary due to randomness (e.g., data splitting, initialization).  
  To obtain stable and robust performance estimates, the framework runs the selected machine learning model multiple times and aggregates the results.


The PSO algorithm in this framework operates in **three phases**:

|------------------|----------------------------|------------------|
      Phase 1                 Phase 2                  Phase 3
    Warmup (20%)           Adaptive (60%)        Exploitation (20%)

The total number of iterations is divided according to:

- `warmup_ratio`
- `adaptive_ratio`
- Remaining iterations are assigned to the exploitation phase.

---

### 🟦 Phase 1 — Warmup

- Focus: Global exploration
- Uses full particle population
- Encourages diversity of candidate solutions

Controlled by:
- `warmup_ratio`

---

### 🟨 Phase 2 — Adaptive

- Focus: Balanced exploration and refinement
- Gradually reduces particles
- Stops early if improvement stagnates

Controlled by:
- `keep_ratio`
- `elite_ratio`
- `patience_phase2`
- `epsilon_phase2`
- `ratio_threshold`

---

### 🟥 Phase 3 — Exploitation

- Focus: Fine-tuning near best solution
- Uses elite particles only
- Stricter convergence criteria

Controlled by:
- `patience_phase3`
- `epsilon_phase3`

---

### ⏱ Runtime Constraint

- `time_budget` limits total optimization runtime (in seconds).

## Contact
For questions or collaborations, please contact [yw825@drexel.edu].