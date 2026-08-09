import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from config import (
    PREPROCESSED_SAVE_PATH,
    MODELS_PATH,
    MODELS_CONFIG,
    HIDDEN_DIM,
    ROBUSTNESS_PATH,
    ROBUSTNESS_METRICS_PATH,
    NOISE_LEVELS,
)

from load_nuscenes import NuScenesDataset

from training_models import (
    Model1_EgoOnly,
    Model2_Attention,
    Model3_TopK,
    Model4_Gated,
    Model5_GatedTopK,
)

from utils import (
    compute_smoothness,
    collision_rate,
)


# ================= CONFIGURATION =================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32

MODEL_CLASSES = {

    "Model1_EgoOnly":
        Model1_EgoOnly,

    "Model2_Attention":
        Model2_Attention,

    "Model3_TopK":
        Model3_TopK,

    "Model4_Gated":
        Model4_Gated,

    "Model5_GatedTopK":
        Model5_GatedTopK,
}

os.makedirs(
    ROBUSTNESS_PATH,
    exist_ok=True
)

os.makedirs(
    ROBUSTNESS_METRICS_PATH,
    exist_ok=True
)


# ================= LOAD DATASET AND MODELS =================

dataset = NuScenesDataset(
    PREPROCESSED_SAVE_PATH,
    allow_pickle=True
)

print(
    f"Total samples: {len(dataset)}"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

models = {}

for model_name, model_config in MODELS_CONFIG.items():

    display_name = model_config[
        "display_name"
    ]

    model_class = MODEL_CLASSES[
        model_name
    ]

    model_path = os.path.join(
        MODELS_PATH,
        model_config["model_file"]
    )

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

    model = model_class(
        hidden_dim=HIDDEN_DIM
    )

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

# ================= METRICS AND EVALUATION =================

def compute_ADE(
    pred,
    gt
):

    return torch.mean(
        torch.norm(
            pred - gt,
            dim=-1
        )
    ).item()


def compute_FDE(
    pred,
    gt
):

    return torch.mean(
        torch.norm(
            pred[:, -1]
            - gt[:, -1],
            dim=-1
        )
    ).item()

def evaluate_model(
    model,
    loader,
    test_type=None,
    noise_std=0.0
):

    ade_total = 0.0
    fde_total = 0.0
    smooth_total = 0.0
    collision_total = 0.0

    count = 0

    with torch.no_grad():

        for (
            ego,
            nbr,
            lane,
            gt
        ) in loader:

            ego = ego.to(
                DEVICE
            )

            nbr = nbr.to(
                DEVICE
            )

            lane = lane.to(
                DEVICE
            )

            gt = gt.to(
                DEVICE
            )

            if test_type == "zero_neighbor":

                nbr = torch.zeros_like(
                    nbr
                )

            elif test_type == "lane_removal":

                lane = torch.zeros_like(
                    lane
                )

            elif test_type == "neighbor_noise":

                noise = (
                    torch.randn_like(
                        nbr[:, :, :, :2]
                    )
                    * noise_std
                )

                nbr[
                    :, :, :, :2
                ] += noise

            elif test_type == "ego_noise":

                noise = (
                    torch.randn_like(
                        ego[:, :, :2]
                    )
                    * noise_std
                )

                ego[
                    :, :, :2
                ] += noise

            output = model(
                ego,
                nbr,
                lane
            )

            if isinstance(
                output,
                tuple
            ):

                pred = output[0]

            else:

                pred = output

            ade_total += compute_ADE(
                pred,
                gt
            )

            fde_total += compute_FDE(
                pred,
                gt
            )

            smooth_total += (
                compute_smoothness(
                    pred
                )
            )

            collision_total += (
                collision_rate(
                    pred,
                    nbr
                )
            )

            count += 1

    return {

        "ADE":
            ade_total / count,

        "FDE":
            fde_total / count,

        "Smoothness":
            smooth_total / count,

        "Collision":
            collision_total / count,
    }

def relative_degradation(
    original,
    tested
):

    return (
        (
            tested - original
        )
        / (
            original + 1e-6
        )
    ) * 100.0

def save_metrics_txt(
    path,
    baseline,
    tested
):

    with open(
        path,
        "w"
    ) as file:

        for model_name in baseline:

            display_name = MODELS_CONFIG[
                model_name
            ]["display_name"]

            file.write(
                "=" * 60
                + "\n"
            )

            file.write(
                f"{display_name}\n"
            )

            file.write(
                "=" * 60
                + "\n"
            )

            for metric in baseline[
                model_name
            ]:

                original = baseline[
                    model_name
                ][metric]

                tested_value = tested[
                    model_name
                ][metric]

                degradation = (
                    relative_degradation(
                        original,
                        tested_value
                    )
                )

                file.write(
                    f"{metric}\n"
                )

                file.write(
                    f"Original : "
                    f"{original:.4f}\n"
                )

                file.write(
                    f"Tested   : "
                    f"{tested_value:.4f}\n"
                )

                file.write(
                    f"Degrade% : "
                    f"{degradation:.2f}%\n\n"
                )

def plot_bar_line(
    metric_name,
    baseline,
    tested,
    save_path,
    title
):

    model_names = list(
        baseline.keys()
    )

    display_names = [
        MODELS_CONFIG[
            name
        ]["display_name"]
        for name in model_names
    ]

    baseline_values = [
        baseline[name][metric_name]
        for name in model_names
    ]

    tested_values = [
        tested[name][metric_name]
        for name in model_names
    ]

    x = np.arange(
        len(model_names)
    )

    plt.figure(
        figsize=(11, 7)
    )

    TITLE_SIZE = 20
    LABEL_SIZE = 15
    TICK_SIZE = 15
    LEGEND_SIZE = 15
    TEXT_SIZE = 15

    plt.bar(
        x,
        baseline_values,
        width=0.5,
        label="Original"
    )
    plt.plot(
        x,
        tested_values,
        marker="o",
        markersize=9,
        linewidth=3,
        linestyle="--",
        color="red",
        label="Tested"
    )
    max_tested = max(
        tested_values
    )

    offset = (
        max_tested * 0.03
        if max_tested != 0
        else 0.03
    )

    for i in range(
        len(model_names)
    ):

        degradation = (
            relative_degradation(
                baseline_values[i],
                tested_values[i]
            )
        )

        plt.text(
            x[i],
            tested_values[i]
            - 3 * offset,
            f"{degradation:.1f}%",
            fontsize=TEXT_SIZE,
            ha="center",
            va="bottom"
        )

    plt.xticks(
        x,
        display_names,
        fontsize=TICK_SIZE
    )

    plt.yticks(
        fontsize=TICK_SIZE
    )

    plt.ylabel(
        metric_name,
        fontsize=LABEL_SIZE
    )

    plt.title(
        title,
        fontsize=TITLE_SIZE
    )

    plt.grid(
        True
    )

    plt.legend(
        fontsize=LEGEND_SIZE
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_noise_curve(
    all_results,
    metric_name,
    save_path,
    title
):

    plt.figure(
        figsize=(11, 7)
    )

    TITLE_SIZE = 20
    LABEL_SIZE = 15
    TICK_SIZE = 15
    LEGEND_SIZE = 15

    for model_name in all_results:

        display_name = MODELS_CONFIG[
            model_name
        ]["display_name"]

        values = []

        for noise_level in NOISE_LEVELS:

            values.append(
                all_results[
                    model_name
                ][noise_level][
                    metric_name
                ]
            )

        plt.plot(
            NOISE_LEVELS,
            values,
            marker="o",
            markersize=8,
            linewidth=3,
            label=display_name
        )

    plt.xlabel(
        "Noise Standard Deviation",
        fontsize=LABEL_SIZE
    )

    plt.ylabel(
        metric_name,
        fontsize=LABEL_SIZE
    )

    plt.title(
        title,
        fontsize=TITLE_SIZE
    )

    plt.xticks(
        NOISE_LEVELS,
        fontsize=TICK_SIZE
    )

    plt.yticks(
        fontsize=TICK_SIZE
    )

    plt.grid(
        True
    )

    plt.legend(
        fontsize=LEGEND_SIZE
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

baseline_results = {}

print(
    "\nRunning baseline evaluation...\n"
)

for model_name, model in models.items():

    display_name = MODELS_CONFIG[
        model_name
    ]["display_name"]

    print(
        f"Evaluating: {display_name}"
    )

    baseline_results[
        model_name
    ] = evaluate_model(
        model,
        loader
    )

def run_single_test(
    test_name,
    test_type
):

    save_dir = os.path.join(
        ROBUSTNESS_PATH,
        test_name
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    tested_results = {}

    print(
        f"\nRunning {test_name}...\n"
    )

    for model_name, model in models.items():

        display_name = MODELS_CONFIG[
            model_name
        ]["display_name"]

        print(
            f"  {display_name}"
        )

        tested_results[
            model_name
        ] = evaluate_model(
            model,
            loader,
            test_type=test_type
        )

    save_metrics_txt(
        os.path.join(
            ROBUSTNESS_METRICS_PATH,
            f"{test_name}.txt"
        ),
        baseline_results,
        tested_results
    )

    metrics = [
        "ADE",
        "FDE",
        "Smoothness",
        "Collision"
    ]

    for metric in metrics:

        plot_bar_line(
            metric,
            baseline_results,
            tested_results,
            os.path.join(
                save_dir,
                f"{metric}.png"
            ),
            f"{test_name} - {metric}"
        )

def run_noise_test(
    test_name,
    noise_target
):

    save_dir = os.path.join(
        ROBUSTNESS_PATH,
        test_name
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    all_noise_results = {

        model_name: {}

        for model_name in models
    }

    for noise_level in NOISE_LEVELS:

        print(
            f"\nNoise level = "
            f"{noise_level}"
        )

        for model_name, model in models.items():

            display_name = MODELS_CONFIG[
                model_name
            ]["display_name"]

            print(
                f"  {display_name}"
            )

            all_noise_results[
                model_name
            ][noise_level] = (
                evaluate_model(
                    model,
                    loader,
                    test_type=noise_target,
                    noise_std=noise_level
                )
            )

    metrics_path = os.path.join(
        ROBUSTNESS_METRICS_PATH,
        f"{test_name}.txt"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        for model_name in all_noise_results:

            display_name = MODELS_CONFIG[
                model_name
            ]["display_name"]

            file.write(
                "=" * 70
                + "\n"
            )

            file.write(
                f"{display_name}\n"
            )

            file.write(
                "=" * 70
                + "\n"
            )

            baseline = baseline_results[
                model_name
            ]

            for noise_level in NOISE_LEVELS:

                file.write(
                    f"\nNoise STD = "
                    f"{noise_level}\n"
                )

                for metric in baseline:

                    original = baseline[
                        metric
                    ]

                    tested_value = (
                        all_noise_results[
                            model_name
                        ][noise_level][
                            metric
                        ]
                    )

                    degradation = (
                        relative_degradation(
                            original,
                            tested_value
                        )
                    )

                    file.write(
                        f"{metric}: "
                        f"{tested_value:.4f} "
                        f"(Deg="
                        f"{degradation:.2f}%)\n"
                    )

    metrics = [
        "ADE",
        "FDE",
        "Smoothness",
        "Collision"
    ]

    for metric in metrics:

        plot_noise_curve(
            all_noise_results,
            metric,
            os.path.join(
                save_dir,
                f"{metric}.png"
            ),
            f"{test_name} - {metric}"
        )

run_single_test(
    "Zero Neighbor Test",
    "zero_neighbor"
)

run_single_test(
    "Lane Removal Test",
    "lane_removal"
)

run_noise_test(
    "Neighbor Robustness Test",
    "neighbor_noise"
)

run_noise_test(
    "Ego Robustness Test",
    "ego_noise"
)

print(
    "\n======================================"
)

print(
    "ALL ROBUSTNESS TESTS COMPLETED"
)

print(
    "======================================"
)

print(
    f"Plots:"
    f"\n{ROBUSTNESS_PATH}"
)

print(
    f"Metrics:"
    f"\n{ROBUSTNESS_METRICS_PATH}"
)