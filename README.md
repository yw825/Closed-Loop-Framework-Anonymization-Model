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


### 1️⃣ Configure `DATASET_CONFIGS`

`DATASET_CONFIGS` is a dictionary that specifies all datasets used in the closed-loop framework.

You may include **one or multiple datasets**. Each dataset is defined as a dictionary containing the following fields:

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

| Phase | Name          | Iteration Ratio | Purpose |
|-------|---------------|-----------------|----------|
| 1     | Warmup        | 20%             | Global exploration |
| 2     | Adaptive      | 60%             | Particle reduction + adaptive refinement |
| 3     | Exploitation  | 20%             | Fine-tuning near best solution |

The total number of iterations is divided according to:

- `warmup_ratio`
- `adaptive_ratio`
- Remaining iterations are assigned to the exploitation phase.


#### 🌍 Phase 1 (Warmup Phase) Parameters

During the warmup phase, the algorithm emphasizes global exploration using the full particle population.  
This phase encourages diversity in candidate solutions before any adaptive reduction is applied.

- **`warmup_ratio`**  
  Proportion of total iterations allocated to the warmup phase.  
  Determines how long the algorithm focuses on broad exploration before transitioning to adaptive refinement.


#### ⚙️ Phase 2 (Adaptive Phase) Parameters

During the adaptive phase, the algorithm gradually reduces the swarm size and monitors convergence.

- **`keep_ratio`**  
  Proportion of particles retained after reduction.  
  Controls how aggressively the swarm size decreases.

- **`elite_ratio`**  
  Proportion of top-performing particles preserved as elite solutions.  
  Ensures high-quality candidates are retained for further refinement.

- **`patience_phase2`**  
  Number of consecutive iterations allowed without significant improvement before triggering adaptation or early stopping.

- **`epsilon_phase2`**  
  Minimum improvement threshold required to consider progress meaningful.  
  If improvement is smaller than this value, it is treated as stagnation.

- **`ratio_threshold`**  
  Threshold used to determine whether population reduction should occur.  
  Helps balance exploration and convergence speed.


#### 🎯 Phase 3 (Exploitation Phase) Parameters

During the exploitation phase, the algorithm focuses on fine-tuning the best-performing solutions.  
The swarm size is reduced, and stricter convergence criteria are applied to refine the search near the current optimum.

- **`patience_phase3`**  
  Number of consecutive iterations allowed without significant improvement before stopping the optimization in this phase.

- **`epsilon_phase3`**  
  Minimum improvement threshold required to consider progress meaningful.  
  Smaller values enforce stricter convergence toward the final solution.

#### ⏱️ Runtime Constraint

- **`time_budget`** 
  Limits total optimization runtime (in seconds).

## Contact
For questions or collaborations, please contact [yw825@drexel.edu].