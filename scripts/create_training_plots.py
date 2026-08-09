import numpy as np
import matplotlib.pyplot as plt
import os

from config import (
    MODEL_CONFIG,
    TRAINING_METRICS_PATH,
    TRAINING_PLOTS_PATH,
)

from utils import load_training_metrics

# ============================================================
# STANDALONE PLOTS
# ============================================================

def plot_losses(
    metrics,
    display_name,
    save_dir,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["train_losses"],
        label="Train"
    )

    plt.plot(
        metrics["val_losses"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.title(
        f"{display_name}: Training vs Validation Loss"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "loss_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_ade(
    metrics,
    display_name,
    save_dir,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["train_ades"],
        label="Train ADE"
    )

    plt.plot(
        metrics["val_ades"],
        label="Validation ADE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("ADE")

    plt.legend()

    plt.title(
        f"{display_name}: ADE vs Epoch"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "ade_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_fde(
    metrics,
    display_name,
    save_dir,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["train_fdes"],
        label="Train FDE"
    )

    plt.plot(
        metrics["val_fdes"],
        label="Validation FDE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("FDE")

    plt.legend()

    plt.title(
        f"{display_name}: FDE vs Epoch"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "fde_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_gradient_norms(
    metrics,
    display_name,
    save_dir,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["grad_norms"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")

    plt.title(
        f"{display_name}: Gradient Norm vs Epoch"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "gradient_norms.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_smoothness(
    metrics,
    display_name,
    save_dir,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["smoothness_scores"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Trajectory Smoothness")

    plt.title(
        f"{display_name}: Prediction Smoothness vs Epoch"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "smoothness.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_collision_rates(
    metrics,
    display_name,
    save_dir,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        metrics["collision_rates"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Collision Rate")

    plt.title(
        f"{display_name}: Collision Rate vs Epoch"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "collision_rates.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_horizon_errors(
    metrics,
    display_name,
    save_dir,
):

    horizon_errors = np.array(
        metrics["horizon_errors"]
    )

    mean_err = horizon_errors.mean(
        axis=0
    )

    horizons = np.arange(
        1,
        len(mean_err) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        horizons,
        mean_err,
        marker="o"
    )

    plt.xlabel("Prediction Horizon")
    plt.ylabel("Average Error")

    plt.title(
        f"{display_name}: Per-Horizon Prediction Error"
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "horizon_errors.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# COMBINED PLOTS
# ============================================================

def plot_combined_metric(
    all_metrics,
    metric_key,
    ylabel,
    title,
    save_name,
    validation=False,
):

    plt.figure(figsize=(8, 5))

    for model_name, model_data in all_metrics.items():

        display_name = MODEL_CONFIG[
            model_name
        ]["display_name"]

        if validation:

            values = model_data[
                f"val_{metric_key}"
            ]

        else:

            values = model_data[
                metric_key
            ]

        plt.plot(
            values,
            label=display_name
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRAINING_PLOTS_PATH,
            "combined",
            save_name
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_combined_horizon_errors(
    all_metrics,
):

    plt.figure(figsize=(8, 5))

    for model_name, model_data in all_metrics.items():

        display_name = MODEL_CONFIG[
            model_name
        ]["display_name"]

        horizon_errors = np.array(
            model_data["horizon_errors"]
        )

        mean_err = horizon_errors.mean(
            axis=0
        )

        horizons = np.arange(
            1,
            len(mean_err) + 1
        )

        plt.plot(
            horizons,
            mean_err,
            marker="o",
            label=display_name
        )

    plt.xlabel("Prediction Horizon")
    plt.ylabel("Average Error")

    plt.title(
        "Per-Horizon Prediction Error Comparison"
    )

    plt.legend()

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRAINING_PLOTS_PATH,
            "combined",
            "combined_horizon_errors.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CREATE ROOT DIRECTORIES
    # ========================================================

    os.makedirs(
        TRAINING_PLOTS_PATH,
        exist_ok=True
    )

    combined_dir = os.path.join(
        TRAINING_PLOTS_PATH,
        "combined"
    )

    os.makedirs(
        combined_dir,
        exist_ok=True
    )

    # ========================================================
    # LOAD ALL MODEL METRICS
    # ========================================================

    all_metrics = {}

    for model_name, model_config in MODEL_CONFIG.items():

        metrics_file = model_config[
            "metrics_file"
        ]

        metrics_path = os.path.join(
            TRAINING_METRICS_PATH,
            metrics_file
        )

        display_name = model_config[
            "display_name"
        ]

        # ----------------------------------------------------
        # Check metrics file
        # ----------------------------------------------------

        if not os.path.exists(
            metrics_path
        ):

            print(
                f"\nWARNING: Metrics file not found:"
                f"\n{metrics_path}"
            )

            print(
                f"Skipping {display_name}."
            )

            continue

        # ----------------------------------------------------
        # Load metrics
        # ----------------------------------------------------

        all_metrics[
            model_name
        ] = load_training_metrics(
            metrics_path
        )

        print(
            f"\nLoaded metrics: "
            f"{display_name}"
        )

    # ========================================================
    # STANDALONE MODEL PLOTS
    # ========================================================

    print("\n======================================")
    print("GENERATING STANDALONE MODEL PLOTS")
    print("======================================")

    for model_name, metrics in all_metrics.items():

        display_name = MODEL_CONFIG[
            model_name
        ]["display_name"]

        # ----------------------------------------------------
        # Model-specific directory
        # ----------------------------------------------------

        save_dir = os.path.join(
            TRAINING_PLOTS_PATH,
            model_name
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        print(
            f"\nGenerating plots: "
            f"{display_name}"
        )

        # ----------------------------------------------------
        # Generate standalone plots
        # ----------------------------------------------------

        plot_losses(
            metrics,
            display_name,
            save_dir
        )

        plot_ade(
            metrics,
            display_name,
            save_dir
        )

        plot_fde(
            metrics,
            display_name,
            save_dir
        )

        plot_gradient_norms(
            metrics,
            display_name,
            save_dir
        )

        plot_smoothness(
            metrics,
            display_name,
            save_dir
        )

        plot_collision_rates(
            metrics,
            display_name,
            save_dir
        )

        plot_horizon_errors(
            metrics,
            display_name,
            save_dir
        )

        print(
            f"Saved to: {save_dir}"
        )

    # ========================================================
    # COMBINED MODEL PLOTS
    # ========================================================

    print("\n======================================")
    print("GENERATING COMBINED MODEL PLOTS")
    print("======================================")

    # --------------------------------------------------------
    # Validation Loss
    # --------------------------------------------------------

    plot_combined_metric(
        all_metrics,
        metric_key="losses",
        ylabel="Validation Loss",
        title="Validation Loss Comparison",
        save_name="combined_val_loss.png",
        validation=True
    )

    # --------------------------------------------------------
    # Validation ADE
    # --------------------------------------------------------

    plot_combined_metric(
        all_metrics,
        metric_key="ades",
        ylabel="Validation ADE",
        title="Validation ADE Comparison",
        save_name="combined_val_ade.png",
        validation=True
    )

    # --------------------------------------------------------
    # Validation FDE
    # --------------------------------------------------------

    plot_combined_metric(
        all_metrics,
        metric_key="fdes",
        ylabel="Validation FDE",
        title="Validation FDE Comparison",
        save_name="combined_val_fde.png",
        validation=True
    )

    # --------------------------------------------------------
    # Gradient Norm
    # --------------------------------------------------------

    plot_combined_metric(
        all_metrics,
        metric_key="grad_norms",
        ylabel="Gradient Norm",
        title="Gradient Norm Comparison",
        save_name="combined_grad_norms.png"
    )

    # --------------------------------------------------------
    # Smoothness
    # --------------------------------------------------------

    plot_combined_metric(
        all_metrics,
        metric_key="smoothness_scores",
        ylabel="Trajectory Smoothness",
        title="Trajectory Smoothness Comparison",
        save_name="combined_smoothness.png"
    )

    # --------------------------------------------------------
    # Collision Rate
    # --------------------------------------------------------

    plot_combined_metric(
        all_metrics,
        metric_key="collision_rates",
        ylabel="Collision Rate",
        title="Collision Rate Comparison",
        save_name="combined_collision_rates.png"
    )

    # --------------------------------------------------------
    # Horizon Errors
    # --------------------------------------------------------

    plot_combined_horizon_errors(
        all_metrics
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n======================================")
    print("ALL TRAINING PLOTS GENERATED")
    print("======================================")

    print(
        f"Standalone plots: "
        f"{TRAINING_PLOTS_PATH}"
    )

    print(
        f"Combined plots: "
        f"{combined_dir}"
    )

