import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from config import MODELS_CONFIG, MODELS_PATH, PREPROCESSED_SAVE_PATH, TRAJECTORY_PLOT_PATH, TRAJECTORY_CSV_PATH, PLOT_RANGE, HIDDEN_DIM
from load_nuscenes import NuScenesDataset
from training_models import Model1_EgoOnly, Model2_Attention, Model3_TopK, Model4_Gated, Model5_GatedTopK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CLASSES = {
    "Model1_EgoOnly": Model1_EgoOnly,
    "Model2_Attention": Model2_Attention,
    "Model3_TopK": Model3_TopK,
    "Model4_Gated": Model4_Gated,
    "Model5_GatedTopK": Model5_GatedTopK,
}

MODEL_COLORS = {
    "Model1_EgoOnly": "y",
    "Model2_Attention": "b",
    "Model3_TopK": "m",
    "Model4_Gated": "c",
    "Model5_GatedTopK": "r",
}

def compute_ADE(pred, gt):
    return np.mean(
        np.linalg.norm(
            pred - gt,
            axis=1
        )
    )

def compute_FDE(pred, gt):
    return np.linalg.norm(
        pred[-1] - gt[-1]
    )

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    TRAJECTORY_PLOT_PATH,
    exist_ok=True
)

os.makedirs(
    TRAJECTORY_CSV_PATH,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

dataset = NuScenesDataset(
    PREPROCESSED_SAVE_PATH,
    allow_pickle=True
)

num_samples = min(
    PLOT_RANGE,
    len(dataset)
)

# ============================================================
# LOAD MODELS
# ============================================================

models = {}

for model_name, model_config in MODELS_CONFIG.items():

    # --------------------------------------------------------
    # Get model class
    # --------------------------------------------------------

    model_class = MODEL_CLASSES[
        model_name
    ]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = model_class(
        hidden_dim=HIDDEN_DIM
    )

    # --------------------------------------------------------
    # Get model path from MODELS_CONFIG
    # --------------------------------------------------------

    model_path = os.path.join(
        MODELS_PATH,
        model_config["model_file"]
    )

    display_name = model_config[
        "display_name"
    ]

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        print(
            f"\nWARNING: Model file not found:"
            f"\n{model_path}"
        )

        print(
            f"Skipping {display_name}."
        )

        continue

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    models[
        model_name
    ] = model

    print(
        f"Loaded: {display_name}"
    )


# ============================================================
# EVALUATE EACH MODEL
# ============================================================

for model_name, model in models.items():

    model_config = MODELS_CONFIG[
        model_name
    ]

    display_name = model_config[
        "display_name"
    ]

    # --------------------------------------------------------
    # Create model-specific output directories
    # --------------------------------------------------------

    model_plot_path = os.path.join(
        TRAJECTORY_PLOT_PATH,
        model_name
    )

    os.makedirs(
        model_plot_path,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Model-specific CSV
    # --------------------------------------------------------

    csv_filename = (
        f"{model_name}_trajectory_metrics.csv"
    )

    csv_path = os.path.join(
        TRAJECTORY_CSV_PATH,
        csv_filename
    )

    # --------------------------------------------------------
    # Store this model's metrics
    # --------------------------------------------------------

    rows = []

    print("\n======================================")
    print(
        f"Evaluating: {display_name}"
    )
    print("======================================")

    # ========================================================
    # LOOP THROUGH DATASET
    # ========================================================

    for idx in range(
        num_samples
    ):

        print(
            f"\rProcessing sample "
            f"{idx + 1}/{num_samples}",
            end=""
        )

        # ----------------------------------------------------
        # Load sample
        # ----------------------------------------------------

        ego, nbr, lane, gt = dataset[
            idx
        ]

        ego_input = ego.unsqueeze(
            0
        ).to(DEVICE)

        nbr_input = nbr.unsqueeze(
            0
        ).to(DEVICE)

        lane_input = lane.unsqueeze(
            0
        ).to(DEVICE)

        gt = gt.numpy()

        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                ego_input,
                nbr_input,
                lane_input
            )

            # Models return:
            # pred, gates, lane_weights, topk_idx

            if isinstance(
                output,
                tuple
            ):

                pred = output[0]

            else:

                pred = output

            pred = pred[
                0
            ].cpu().numpy()

        # ----------------------------------------------------
        # Calculate ADE / FDE
        # ----------------------------------------------------

        ade = compute_ADE(
            pred,
            gt
        )

        fde = compute_FDE(
            pred,
            gt
        )

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        rows.append({
            "idx": idx,
            "ADE": ade,
            "FDE": fde
        })

        # ----------------------------------------------------
        # Ego history
        # ----------------------------------------------------

        ego_hist = ego.numpy()

        ego_x = ego_hist[
            :,
            0
        ]

        ego_y = ego_hist[
            :,
            1
        ]

        # ----------------------------------------------------
        # Create trajectory plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 8)
        )

        # Ego history

        plt.plot(
            ego_x,
            ego_y,
            "ko-",
            label="Ego History"
        )

        # Ground truth

        plt.plot(
            gt[:, 0],
            gt[:, 1],
            "g-",
            linewidth=2,
            label="Ground Truth"
        )

        # Model prediction

        plt.plot(
            pred[:, 0],
            pred[:, 1],
            linestyle="--",
            color=MODEL_COLORS[
                model_name
            ],
            label=display_name
        )

        plt.title(
            f"{display_name}: "
            f"Trajectory Prediction "
            f"(idx={idx})"
        )

        plt.xlabel(
            "X Position (meters)"
        )

        plt.ylabel(
            "Y Position (meters)"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.axis(
            "equal"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Save trajectory
        # ----------------------------------------------------

        save_path = os.path.join(
            model_plot_path,
            f"traj_{idx:03d}.png"
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    print()

    # ========================================================
    # SAVE MODEL CSV
    # ========================================================

    with open(
        csv_path,
        mode="w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "idx",
                "ADE",
                "FDE"
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    print(
        f"Plots saved to:"
        f"\n{model_plot_path}"
    )

    print(
        f"CSV saved to:"
        f"\n{csv_path}"
    )

    print(
        f"Mean ADE: "
        f"{np.mean([row['ADE'] for row in rows]):.4f}"
    )

    print(
        f"Mean FDE: "
        f"{np.mean([row['FDE'] for row in rows]):.4f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n======================================")
print("TRAJECTORY EVALUATION COMPLETE")
print("======================================")

print(
    f"Samples per model : {num_samples}"
)

print(
    f"Trajectory plots  : {TRAJECTORY_PLOT_PATH}"
)

print(
    f"Trajectory CSVs   : {TRAJECTORY_CSV_PATH}"
)

print(
    "======================================")
