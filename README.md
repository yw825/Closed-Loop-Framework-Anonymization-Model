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
  Parameter for *l-diversity*.

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


## Contact
For questions or collaborations, please contact [yw825@drexel.edu].