# Preprocessing Sanity Check

This directory contains scripts used to verify the output of the preprocessing pipeline.

The sanity-check script performs a small-scale inspection of the generated preprocessed dataset. It extracts only a limited number of samples and prints or checks their values to confirm that the preprocessing steps have produced valid data.

## Purpose

The sanity check is intended to verify:

* The preprocessed `.npz` file can be loaded successfully.
* The expected data fields are present.
* Sample shapes and dimensions are correct.
* Ego trajectories contain valid values.
* Neighbor trajectories contain valid values.
* Lane data contains valid values.
* Ground-truth trajectories contain valid values.
* The preprocessing output is suitable for loading by the training pipeline.

Only a small subset of the data is inspected to make the check quick and practical during development.

## Usage

Run the sanity-check script after generating the preprocessed dataset.

The script uses the preprocessing output specified by:

```python
PREPROCESSED_SAVE_PATH
```

in `scripts/config.py`.

For example:

```text
data/
└── embeddings/
    └── nuscenes_preprocessed.npz
```

The sanity check does **not** generate a new training dataset or modify the existing preprocessed file.

## Relationship to the Main Pipeline

The sanity-check code is a validation/debugging utility and is separate from the main preprocessing pipeline.

The intended workflow is:

```text
Raw nuScenes Dataset
        │
        ▼
Preprocessing
        │
        ▼
nuscenes_preprocessed.npz
        │
        ▼
Preprocessing Sanity Check
        │
        ▼
Verify sample values and shapes
        │
        ▼
Training / Evaluation
```

If the sanity check produces unexpected shapes, missing values, or invalid trajectory/lane data, the preprocessing pipeline should be investigated before proceeding to model training.
