import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils import count_parameters
from load_nuscenes import NuScenesDataset
from training_models import Model1_EgoOnly, Model2_Attention, Model3_TopK, Model4_Gated, Model5_GatedTopK

# =========================================================
# BENCHMARK
# =========================================================

def benchmark_models(models, loader, device="cpu", warmup=10, runs=5):

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    results = []

    for model_name, model in models.items():

        print(f"\nBenchmarking {model_name}...")

        model.to(device)
        model.eval()

        # -----------------------------
        # Warmup
        # -----------------------------
        with torch.no_grad():

            for i, (ego, nbr, lane, gt) in enumerate(loader):

                if i >= warmup:
                    break

                ego = ego.to(device)
                nbr = nbr.to(device)
                lane = lane.to(device)

                _ = model(ego, nbr, lane)

        # -----------------------------
        # Timed runs
        # -----------------------------
        batch_times = []

        for r in range(runs):

            with torch.no_grad():

                for ego, nbr, lane, gt in loader:

                    ego = ego.to(device)
                    nbr = nbr.to(device)
                    lane = lane.to(device)

                    start = time.perf_counter()

                    _ = model(ego, nbr, lane)

                    end = time.perf_counter()

                    batch_times.append(end - start)

        batch_times = np.array(batch_times)

        mean_batch_time = batch_times.mean()
        std_batch_time = batch_times.std()

        ms_per_batch = mean_batch_time * 1000

        ms_per_sample = (
            mean_batch_time / loader.batch_size
        ) * 1000

        fps = 1.0 / mean_batch_time

        params = count_parameters(model)

        results.append({
            "Model": model_name,
            "Parameters": params,
            "Mean Batch Time (ms)": ms_per_batch,
            "Std Batch Time (ms)": std_batch_time * 1000,
            "ms/sample": ms_per_sample,
            "FPS (batch-based)": fps
        })

        print(f"Done: {model_name}")

    return pd.DataFrame(results)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    device = "cpu"

    dataset = NuScenesDataset(
        "nuscenes_full_preprocessed_final.npz"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False
    )

    hidden_dim = 128

    models = {
        "Model1_EgoOnly": Model1_EgoOnly(hidden_dim),
        "Model2_Attention": Model2_Attention(hidden_dim),
        "Model3_TopK": Model3_TopK(hidden_dim),
        "Model4_Gated": Model4_Gated(hidden_dim),
        "Model5_GatedTopK": Model5_GatedTopK(hidden_dim)
    }

    results_df = benchmark_models(
        models=models,
        loader=val_loader,
        device=device,
        warmup=10,
        runs=5
    )

    print("\n======================================")
    print(results_df)
    print("======================================")

    csv_path = "benchmark_results_all_models.csv"

    results_df.to_csv(csv_path, index=False)

    print(f"\nCSV saved to: {csv_path}")
   