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

## 1️⃣ Configure `DATASET_CONFIGS`

`DATASET_CONFIGS` is a dictionary that specifies all datasets used in the closed-loop framework.

You may include **one or multiple datasets**. Each dataset is defined as a dictionary containing the following fields:

### Required Fields

- **`path`**  
  Path to the dataset file.

- **`numeric_qis`**  
  List of numeric quasi-identifiers (QIs).

- **`categorical_qis`**  
  List of categorical quasi-identifiers (QIs).

- **`sensitive_attributes`**  
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

---

### Example

```python
DATASET_CONFIGS = {
    "dataset_name": {
        "path": "data/dataset.csv",
        "numeric_qis": ["age", "income"],
        "categorical_qis": ["gender", "zipcode"],
        "sensitive_attributes": ["disease"],
        "n_clusters": [2, 3, 4],
        "l": 2
    }
}

In EXPERIMENT_SOURCES, you have to define the base path for the folder you want to save all the results. Since we designed different types of experiments to run framework (closed-loop with repairing process, closed-loop without repairing process), you can add multiple experiments in it.


## Contact
For questions or collaborations, please contact [yw825@drexel.edu].