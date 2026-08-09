from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

DATAROOT = ROOT_PATH/"data/nuscenes_mini"
VERSION = "v1.0-mini"

PREPROCESSED_SAVE_PATH = ROOT_PATH/"data/embeddings/nuscenes_preprocessed.npz"
TRAINING_PLOT_PATH = ROOT_PATH/"output/plots/training"
TRAJECTORY_PLOT_PATH = ROOT_PATH/"output/plots/trajectories"
TRAJECTORY_CSV_PATH = ROOT_PATH/"output/metrics/trajectories"
MODELS_PATH = ROOT_PATH/"output/models"
TRAINING_METRICS_PATH = ROOT_PATH/"data/metrics/training_metrics"
ROBUSTNESS_PATH = ROOT_PATH/"output/plots/robustness"
ROBUSTNESS_METRICS_PATH = ROOT_PATH/"output/metrics/robustness"

MODELS_CONFIG = {
    "Model1_EgoOnly": {
        "display_name": "Ego-only",
        "model_file": "Model1_EgoOnly.pth",
        "metrics_file": "Model1_EgoOnly_training_metrics.npz",
    },

    "Model2_Attention": {
        "display_name": "Attention",
        "model_file": "Model2_Attention.pth",
        "metrics_file": "Model2_Attention_training_metrics.npz",
    },

    "Model3_TopK": {
        "display_name": "Top-K",
        "model_file": "Model3_TopK.pth",
        "metrics_file": "Model3_TopK_training_metrics.npz",
    },

    "Model4_Gated": {
        "display_name": "Gated",
        "model_file": "Model4_Gated.pth",
        "metrics_file": "Model4_Gated_training_metrics.npz",
    },

    "Model5_GatedTopK": {
        "display_name": "Gated + Top-K",
        "model_file": "Model5_GatedTopK.pth",
        "metrics_file": "Model5_GatedTopK_training_metrics.npz",
    },
}

T = 6
H = 12
K = 5

N_LANES = 6
LANE_POINTS = 20
RADIUS = 30

PLOT_RANGE = 225
HIDDEN_DIM = 128


NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0]