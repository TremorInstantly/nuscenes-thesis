import torch
from torch.utils.data import DataLoader
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from load_nuscenes import NuScenesDataset
from utils import trajectory_loss, save_training_metrics, collision_rate, per_horizon_error, compute_smoothness
from training_models import Model1_EgoOnly, Model2_Attention, Model3_TopK, Model4_Gated, Model5_GatedTopK
from config import PREPROCESSED_SAVE_PATH, MODELS_PATH, TRAINING_METRICS_PATH, MODEL_CONFIG

def train(model, train_loader, val_loader, model_name, model_save_path, epochs=500, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses, val_losses = [], []
    grad_norms = []
    train_ades = []
    val_ades = []

    train_fdes = []
    val_fdes = []
    smoothness_scores = []
    collision_rates = []
    horizon_errors = []
    
    best_val = float("inf")
    best_ade = float("inf")
    best_fde = float("inf")
    best_collision_rate = float("inf")
    best_smoothness = float("inf")
    best_horizon_error = None
    best_grad_norm = float("inf")

    best_train_loss = float("inf")
    best_train_ade = float("inf")
    best_train_fde = float("inf")
    best_epoch = 0
    patience = 15
    counter = 0
    
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_ade = 0
        total_fde = 0
        epoch_grad_norm = 0.0

        for ego, nbr, lane, gt in train_loader:
            ego, nbr, lane, gt = ego.to(device), nbr.to(device), lane.to(device), gt.to(device)

            pred, _, _, _ = model(ego, nbr, lane) #pred, gates, lane_weights, topk_idx
            loss, ade, fde = trajectory_loss(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            batch_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    batch_grad_norm += p.grad.norm().item()
            epoch_grad_norm += batch_grad_norm
            optimizer.step()

            total_loss += loss.item()
            total_ade += ade.item()
            total_fde += fde.item()

        avg_grad_norm = epoch_grad_norm / len(train_loader)
        grad_norms.append(avg_grad_norm)
        train_loss = total_loss / len(train_loader)
        train_ade = total_ade / len(train_loader)
        train_fde = total_fde / len(train_loader)

        train_losses.append(train_loss)
        train_ades.append(train_ade)
        train_fdes.append(train_fde)

        # Validation
        model.eval()
        val_loss_total = 0
        val_ade_total = 0
        val_fde_total = 0
        smoothness_total = 0
        collision_total = 0
        horizon_error_sum = None

        with torch.no_grad():
            for ego, nbr, lane, gt in val_loader:
                ego, nbr, lane, gt = ego.to(device), nbr.to(device), lane.to(device), gt.to(device)

                pred, _, _, _ = model(ego, nbr, lane)
                loss, ade, fde = trajectory_loss(pred, gt)

                val_loss_total += loss.item()
                val_ade_total += ade.item()
                val_fde_total += fde.item()
                smoothness_total += compute_smoothness(pred)
                collision_total += collision_rate(pred, nbr)
                batch_horizon_error = per_horizon_error(pred, gt)
                if horizon_error_sum is None:
                    horizon_error_sum = batch_horizon_error
                else:
                    horizon_error_sum += batch_horizon_error

        val_loss = val_loss_total / len(val_loader)
        val_ade = val_ade_total / len(val_loader)
        val_fde = val_fde_total / len(val_loader)
        avg_smoothness = smoothness_total / len(val_loader)
        avg_collision = collision_total / len(val_loader)
        avg_horizon_error = horizon_error_sum / len(val_loader)
        horizon_errors.append(avg_horizon_error)

        val_losses.append(val_loss)
        val_ades.append(val_ade)
        val_fdes.append(val_fde)
        smoothness_scores.append(avg_smoothness)
        collision_rates.append(avg_collision)
        
        epoch_time = time.time() - epoch_start
        
        # EARLY STOPPING
        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch + 1

            best_ade = val_ade
            best_fde = val_fde
            
            best_collision_rate = avg_collision
            
            best_smoothness = avg_smoothness
            best_grad_norm = avg_grad_norm
            best_train_loss = train_loss
            best_train_ade = train_ade
            best_train_fde = train_fde
            best_horizon_error = avg_horizon_error
            
            counter = 0
            torch.save(model.state_dict(),model_save_path)
        else:
            counter += 1
            
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Train ADE={train_ade:.4f} | "
            f"Val ADE={val_ade:.4f} | "
            f"Train FDE={train_fde:.4f} | "
            f"Val FDE={val_fde:.4f} | "
            f"Collision Rate={avg_collision:.4f} | "
            f"Time={epoch_time:.2f}s"
        )
        
        if counter > patience:
            print("Early stopping triggered")
            break

    total_time = time.time() - start_time

    print("\n======================================")
    print("FINAL TRAINING SUMMARY")
    print("======================================")

    print(f"Best Epoch                : {best_epoch}")

    print("\n----- LOSSES -----")
    print(f"Best Validation Loss      : {best_val:.4f}")
    print(f"Best Train Loss           : {best_train_loss:.4f}")

    print("\n----- TRAJECTORY METRICS -----")
    print(f"Best Train ADE            : {best_train_ade:.4f}")
    print(f"Best Validation ADE       : {best_ade:.4f}")

    print(f"Best Train FDE            : {best_train_fde:.4f}")
    print(f"Best Validation FDE       : {best_fde:.4f}")

    print("\n----- STABILITY METRICS -----")
    print(f"Best Gradient Norm        : {best_grad_norm:.4f}")
    print(f"Best Smoothness           : {best_smoothness:.4f}")

    print("\n----- SAFETY METRICS -----")
    print(f"Best Collision Rate       : {best_collision_rate:.4f}")

    print("\n----- EFFICIENCY METRICS -----")
    print(f"Total Training Time       : {total_time:.2f} sec")
    print(f"Average Epoch Time        : {total_time / len(train_losses):.2f} sec")
    
    print("\n----- HORIZON ERRORS -----")
    for i, err in enumerate(best_horizon_error):
        print(f"Horizon {i+1:02d} : {err:.4f}")
    print("======================================")

    return (
        train_losses,
        val_losses,
        train_ades,
        val_ades,
        train_fdes,
        val_fdes,
        grad_norms,
        smoothness_scores,
        collision_rates,
        horizon_errors
    )


if __name__ == "__main__":
    dataset = NuScenesDataset(PREPROCESSED_SAVE_PATH, allow_pickle=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    models = [ # Comment out models if necessary
        (
            "Model1_EgoOnly", Model1_EgoOnly(hidden_dim=64)
            ),
        (
            "Model2_Attention", Model2_Attention(hidden_dim=64)
            ), 
        (
            "Model3_TopK", Model3_TopK(hidden_dim=64) 
            ),
        ( 
            "Model4_Gated", Model4_Gated(hidden_dim=64)
            ), 
        (
            "Model5_GatedTopK", Model5_GatedTopK(hidden_dim=64)
            ), ]
    
    for model_name, model in models:
        
        model_config = MODEL_CONFIG[ model_name ]
        display_name = model_config[ "display_name" ]
        model_file = model_config[ "model_file" ]
        metrics_file = model_config[ "metrics_file" ]
        
        model_save_path = os.path.join( MODELS_PATH, model_file )
        metrics_save_path = os.path.join( TRAINING_METRICS_PATH, metrics_file )
        
        print("\n") 
        print("######################################") 
        print( f"STARTING TRAINING: {display_name}" ) 
        print("######################################")
        
        (
            train_losses,
            val_losses,
            train_ades,
            val_ades,
            train_fdes,
            val_fdes,
            grad_norms,
            smoothness_scores,
            collision_rates,
            horizon_errors
        ) = train(model, train_loader, val_loader, model_name=model_name, model_save_path=model_save_path, device=device)
    
        save_training_metrics(
            metrics_save_path,
            train_losses,
            val_losses,
            train_ades,
            val_ades,        
            train_fdes,
            val_fdes,
            grad_norms,
            smoothness_scores,
            collision_rates,
            horizon_errors
        )
        
        print( f"Model saved to: " f"{model_save_path}" ) 
        print( f"Metrics saved to: " f"{metrics_save_path}" )
    
