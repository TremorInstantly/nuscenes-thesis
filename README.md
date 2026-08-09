# Physics-Guided Interaction Sparsification for Efficient Ego Trajectory Prediction

This repository contains the implementation and experimental pipeline for **Physics-Guided Interaction Sparsification for Efficient Ego Trajectory Prediction**.

The project investigates interaction-aware ego-trajectory prediction using a combination of learned interaction modeling and physics-informed features, with the goal of improving prediction efficiency while maintaining trajectory accuracy and safety.

The detailed methodology, experimental design, and project organization are described in the published research paper included in the [`docs/`](docs/) directory.

## Publication

The published paper is available in the [`docs/`](docs/) directory.

For the complete methodology, model formulation, experiments, and results, please refer to the publication.

---

# Dataset

This project uses the **nuScenes** dataset.

The dataset is **not included in this repository**. It must be downloaded separately from the official nuScenes website.

Dataset setup instructions are provided in:

[`data/nuscenes_mini/README.md`](data/nuscenes_mini/README.md)

The project supports both:

* `v1.0-mini` for development and lightweight experiments
* `v1.0-trainval` for larger-scale experiments

The nuScenes Map Expansion is also required for the preprocessing pipeline.

---

# Requirements

The project is implemented in Python and uses PyTorch for model training.

Install the required dependencies using the project's dependency file:

```bash
pip install -r requirements.txt
```

The exact Python environment and dependency versions used for the experiments should be maintained where possible for reproducibility.

---

# Repository Structure

The repository is organized around the dataset, implementation scripts, documentation, and generated outputs.

```text
.
├── data/
│   ├── nuscenes_mini/
│   │   └── README.md
│   ├── embeddings/
│   └── metrics/
│
├── docs/
│   └── published paper
│
├── output/
│   ├── models/
│   ├── plots/
│   └── metrics/
│
├── sanity_check/
│   └── README.md
│
├── scripts/
│   ├── preprocessing.py
│   ├── train.py
│   ├── benchmarking.py
│   ├── create_training_plots.py
│   ├── create_trajectory_plots.py
│   ├── pertuberation_analysis.py
│   ├── training_models.py
│   ├── load_nuscenes.py
│   ├── utils.py
│   └── config.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets, trained model checkpoints, plots, and evaluation outputs are generally excluded from version control and can be regenerated using the provided scripts.

For a detailed description of the project's methodology and overall organization, refer to the published paper in [`docs/`](docs/).

---

# Pipeline

The primary experimental workflow is:

```text
nuScenes Dataset
       │
       ▼
preprocessing.py
       │
       ▼
Preprocessed Dataset
       │
       ▼
train.py
       │
       ▼
Trained Models + Training Metrics
       │
       ├──────────────► benchmarking.py
       │
       ├──────────────► create_training_plots.py
       │
       ├──────────────► create_trajectory_plots.py
       │
       └──────────────► pertuberation_analysis.py
```

## 1. Preprocessing

Run:

```bash
python scripts/preprocessing.py
```

This processes the downloaded nuScenes dataset and generates the preprocessed dataset used by the training and evaluation scripts.

The output location is controlled by `PREPROCESSED_SAVE_PATH` in `scripts/config.py`.

A small sanity-check utility is provided separately to verify the generated data before training.

---

## 2. Training

Run:

```bash
python scripts/train.py
```

The training pipeline supports the five model variants used in the experiments:

1. Ego-only
2. Attention
3. Top-K
4. Gated
5. Gated + Top-K

Model definitions and filenames are centrally managed through `MODELS_CONFIG` in `config.py`.

The script saves the best model checkpoints and training metrics to the configured output locations.

Individual models can be enabled or disabled in `train.py` depending on the experiment being performed.

---

## 3. Benchmarking

Run:

```bash
python scripts/benchmarking.py
```

Benchmarking evaluates the trained models and does not require retraining.

Therefore, if valid model checkpoints already exist, `benchmarking.py` can be run directly after the required data has been prepared.

---

## 4. Training Plots

Run:

```bash
python scripts/create_training_plots.py
```

This generates training visualizations for the available models, including:

* Training and validation loss
* ADE
* FDE
* Gradient norm
* Trajectory smoothness
* Collision rate
* Per-horizon prediction error

Both individual-model plots and combined model-comparison plots are generated.

---

## 5. Trajectory Plots

Run:

```bash
python scripts/create_trajectory_plots.py
```

This generates qualitative trajectory comparisons for the trained models and records the corresponding trajectory metrics.

The outputs include:

* Ego history
* Ground-truth trajectory
* Predicted trajectories
* ADE
* FDE

Results are organized by model.

---

## 6. Perturbation Analysis

Run:

```bash
python scripts/pertuberation_analysis.py
```

This performs perturbation-based robustness analysis of the trained models.

The resulting robustness plots and metrics are saved to the configured robustness output directories.

---

# Configuration

Project-wide paths and experiment parameters are maintained in:

```text
scripts/config.py
```

The configuration uses `ROOT_PATH` to construct paths relative to the repository root. This allows the repository to be moved between machines without changing absolute paths throughout the codebase.

Important configuration values include:

```python
DATAROOT
VERSION

PREPROCESSED_SAVE_PATH

MODELS_PATH
TRAINING_METRICS_PATH

TRAINING_PLOT_PATH

TRAJECTORY_PLOT_PATH
TRAJECTORY_CSV_PATH

ROBUSTNESS_PATH
ROBUSTNESS_METRICS_PATH
```

The model definitions are maintained through:

```python
MODELS_CONFIG
```

This provides a single source of truth for model names, display names, checkpoint filenames, and training-metric filenames.

---

# Reproducing the Experiments

A typical fresh setup is:

### 1. Obtain the dataset

Follow:

[`data/nuscenes_mini/README.md`](data/nuscenes_mini/README.md)

### 2. Configure the dataset

Update `scripts/config.py` if using a different nuScenes release or dataset location.

### 3. Generate the preprocessed data

```bash
python scripts/preprocessing.py
```

### 4. Perform the preprocessing sanity check

Use the scripts in:

```text
sanity_check/
```

to verify the generated data.

### 5. Train the models

```bash
python scripts/train.py
```

### 6. Benchmark the trained models

```bash
python scripts/benchmarking.py
```

### 7. Generate training plots

```bash
python scripts/create_training_plots.py
```

### 8. Generate trajectory visualizations and metrics

```bash
python scripts/create_trajectory_plots.py
```

### 9. Run robustness analysis

```bash
python scripts/pertuberation_analysis.py
```

---

# Reproducibility Notes

The repository intentionally does not include the original nuScenes dataset or generated large binary artifacts such as the preprocessed `.npz` dataset.

Likewise, trained model checkpoints and generated experimental outputs are excluded from version control unless explicitly provided as research artifacts.

This keeps the repository lightweight while allowing the complete processing and experimental pipeline to be reproduced from the source code and the required dataset.

For the scientific methodology, model descriptions, experimental setup, and reported results, refer to the published paper in [`docs/`](docs/).

---

# Citation

If you use this repository or the associated research in your work, please cite the published paper provided in [`docs/`](docs/).

See the publication for the complete citation information.
