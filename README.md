# Physics-Guided Interaction Sparsification for Efficient Ego Trajectory Prediction

This repository contains the implementation and experimental pipeline for **Physics-Guided Interaction Sparsification for Efficient Ego Trajectory Prediction**.

The project investigates interaction-aware ego-trajectory prediction using a combination of learned interaction modeling and physics-informed features, with the goal of improving prediction efficiency while maintaining trajectory accuracy and safety.

The detailed methodology, experimental design, and project organization are described in the published research paper included in the [`docs/`](docs/) directory.

## Publication

The published paper is available in the (docs/) directory.

For the complete methodology, model formulation, experiments, and results, please refer to the publication.

---

# Dataset

This project uses the nuScenes dataset.

The dataset is not included in this repository. It must be downloaded separately from the official nuScenes website.

Dataset setup instructions are provided in:

data/nuscenes_mini/README.md

The project supports both:

* v1.0-mini for development and lightweight experiments
* v1.0-trainval for larger-scale experiments

The nuScenes Map Expansion is also required for the preprocessing pipeline.

---

# Requirements

The project is implemented in Python and uses PyTorch for model training.

Install the required dependencies using the project's dependency file:

pip install -r requirements.txt

The exact Python environment and dependency versions used for the experiments should be maintained where possible for reproducibility.

---

# Repository Structure

The repository is organized around the dataset, implementation scripts, documentation, and generated outputs.

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

Generated datasets, trained model checkpoints, plots, and evaluation outputs are generally excluded from version control and can be regenerated using the provided scripts.

For a detailed description of the project's methodology and overall organization, refer to the published paper in (docs/).

---

# Pipeline

The primary experimental workflow is:

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

# Reproducing the Experiments

A typical fresh setup is:

### 1. Obtain the dataset

Follow data/nuscenes_mini/README.md

### 2. Configure the dataset

Update 'scripts/config.py' if using a different nuScenes release or dataset location.

### 3. Generate the preprocessed data

python scripts/preprocessing.py

### 4. Perform the preprocessing sanity check

Use the scripts in sanity_check to verify the generated data.

### 5. Train the models

python scripts/train.py

### 6. Benchmark the trained models

python scripts/benchmarking.py

### 7. Generate training plots

python scripts/create_training_plots.py

### 8. Generate trajectory visualizations and metrics

python scripts/create_trajectory_plots.py

### 9. Run robustness analysis

python scripts/pertuberation_analysis.py

---

# Reproducibility Notes

The repository intentionally does not include the original nuScenes dataset or generated large binary artifacts such as the preprocessed `.npz` dataset. Likewise, trained model checkpoints and generated experimental outputs are excluded from version control unless explicitly provided as research artifacts.

This keeps the repository lightweight while allowing the complete processing and experimental pipeline to be reproduced from the source code and the required dataset. For the scientific methodology, model descriptions, experimental setup, and reported results, refer to the published paper in (docs/).

---

# Citation

If you use this repository or the associated research in your work, please cite the published paper provided in (docs/).

See the publication for the complete citation information.
