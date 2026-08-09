# Scripts

This directory contains the preprocessing, training, evaluation, benchmarking, visualization, and robustness-analysis scripts for the project.

The main workflow is executed in the following order:

```text
Preprocessing
     │
     ▼
Training
     │
     ▼
Benchmarking / Evaluation
     │
     ▼
Training Plots
     │
     ▼
Trajectory Plots
     │
     ▼
Perturbation Analysis
```

## Main Pipeline Scripts

### 1. `preprocessing.py`

Processes the nuScenes dataset and generates the preprocessed dataset used by the training pipeline.

The generated dataset is saved according to:

```python
PREPROCESSED_SAVE_PATH
```

in `config.py`.

Run this first when setting up the project or when regenerating the preprocessed dataset.

---

### 2. `train.py`

Trains the trajectory-prediction models using the preprocessed dataset.

The available models are defined through the model configuration and include:

* Ego-only
* Attention
* Top-K
* Gated
* Gated + Top-K

The script saves:

* Best model checkpoints
* Training metrics

to the locations specified in `config.py`.

Individual models can be commented out in the training script when only a specific model needs to be trained.

---

### 3. `benchmarking.py`

Evaluates and benchmarks the trained models.

This script does **not require model training** and can therefore be run independently after trained model checkpoints are available.

If the models have already been trained, this script can be used instead of `train.py` as the next stage of the workflow.

---

### 4. `create_training_plots.py`

Generates visualizations from the training metrics produced by `train.py`.

The script generates:

* Individual model training plots
* Combined model comparison plots

These include metrics such as:

* Training and validation loss
* ADE
* FDE
* Gradient norm
* Trajectory smoothness
* Collision rate
* Per-horizon prediction error

The output locations are controlled by the paths defined in `config.py`.

---

### 5. `create_trajectory_plots.py`

Generates trajectory-prediction visualizations for the trained models.

The generated results include:

* Ego history
* Ground-truth future trajectory
* Model predictions
* ADE
* FDE

Trajectory plots and their corresponding evaluation CSV files are saved according to:

```python
TRAJECTORY_PLOT_PATH
TRAJECTORY_CSV_PATH
```

defined in `config.py`.

---

### 6. `pertuberation_analysis.py`

Performs perturbation-based robustness analysis of the trained trajectory-prediction models.

The script evaluates model behavior under controlled perturbations and generates the corresponding robustness plots and metrics.

Output locations are controlled by:

```python
ROBUSTNESS_PATH
ROBUSTNESS_METRICS_PATH
```

in `config.py`.

---

# Supporting Files

The remaining files provide functionality used by the main pipeline scripts and are generally not intended to be executed independently.

### `config.py`

Contains project-wide configuration values, including:

* Dataset locations
* Preprocessed dataset location
* Model locations
* Training metric locations
* Plot locations
* Robustness-analysis locations
* Model configuration
* Dataset parameters
* Training/model parameters

Paths are constructed relative to the project root using `ROOT_PATH`.

---

### `load_nuscenes.py`

Contains the dataset-loading functionality used to load the preprocessed nuScenes data for training and evaluation.

---

### `training_models.py`

Contains the trajectory-prediction model architectures used by the project.

The implementations correspond to the models defined in the project configuration.

---

### `utils.py`

Contains reusable utility functions used throughout the pipeline, including functionality for:

* Trajectory loss calculation
* Training metric storage/loading
* ADE/FDE-related calculations
* Collision-rate calculation
* Per-horizon error calculation
* Trajectory smoothness calculation
* Other shared processing utilities

---

# Recommended Execution Order

For a fresh setup, run:

```text
1. preprocessing.py
        │
        ▼
2. train.py
        │
        ▼
3. benchmarking.py
        │
        ▼
4. create_training_plots.py
        │
        ▼
5. create_trajectory_plots.py
        │
        ▼
6. pertuberation_analysis.py
```

Training plots require the training metrics generated during training, so the corresponding metric files must already exist.

Trajectory plots, benchmarking, and perturbation analysis require the trained model checkpoints.

---

# Configuration

The scripts share project-wide configuration through:

```text
config.py
```

The project root is automatically determined from the location of `config.py`, allowing the repository to be moved to another machine without modifying hard-coded absolute paths.

Important configuration categories include:

```text
Dataset
├── DATAROOT
└── VERSION

Preprocessing
└── PREPROCESSED_SAVE_PATH

Models
└── MODELS_PATH

Training Metrics
└── TRAINING_METRICS_PATH

Training Plots
└── TRAINING_PLOT_PATH

Trajectory Evaluation
├── TRAJECTORY_PLOT_PATH
└── TRAJECTORY_CSV_PATH

Robustness Analysis
├── ROBUSTNESS_PATH
└── ROBUSTNESS_METRICS_PATH
```

Refer to `config.py` before running the pipeline if the dataset location or output locations need to be changed.

---

# Notes

The scripts in this directory represent the final project pipeline. Older experimental and deprecated implementations are not included in the repository.

Generated datasets, trained model checkpoints, plots, and evaluation outputs are excluded from version control and should be regenerated using the scripts described above or obtained from the corresponding research artifacts.
