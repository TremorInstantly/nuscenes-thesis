import numpy as np
import math
import torch

def quat_to_yaw(q):
    w, x, y, z = q
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    return math.atan2(siny, cosy)
    

    
def compute_kinematics(traj):
    """
    traj: [(x,y,yaw), ...]
    returns: (T,5) -> x,y,v,a,yaw
    """

    out = []
    prev_v = 0.0
    
    # print("Len=",len(traj))

    for i in range(len(traj)):

        x, y, yaw = traj[i]

        x = float(x)
        y = float(y)
        yaw = float(yaw)

        if i == 0:
            # print("NO MOVING")
            
            v = 0.0
            a = 0.0
        else:
            # print("MOVING")
            px, py, _ = traj[i - 1]
            dx = x - px
            dy = y - py
            v = math.sqrt(dx * dx + dy * dy)
            a = v - prev_v

        prev_v = v
        # print("accel=",a)
        # print("vel=",v)
        out.append([x, y, v, a, yaw])

    return np.array(out, dtype=np.float32)
    


def get_neighbor_mask(neighbors):
    # neighbors: (B,K,T,5)
    mask = (neighbors.abs().sum(dim=-1).sum(dim=-1) > 0)  # (B,K)
    return mask
    
    
    
def save_training_metrics(
    save_path,
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
):

    np.savez(
        save_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_ades=np.array(train_ades),
        val_ades=np.array(val_ades),
        train_fdes=np.array(train_fdes),
        val_fdes=np.array(val_fdes),
        grad_norms=np.array(grad_norms),
        smoothness_scores=np.array(smoothness_scores),
        collision_rates=np.array(collision_rates),
        horizon_errors=np.array(horizon_errors)
    )

    print(f"\nMetrics saved to: {save_path}")

def load_training_metrics(path):

    data = np.load(path, allow_pickle=True)

    return {
        "train_losses": data["train_losses"],
        "val_losses": data["val_losses"],
        "train_ades": data["train_ades"],
        "val_ades": data["val_ades"],
        "train_fdes": data["train_fdes"],
        "val_fdes": data["val_fdes"],
        "grad_norms": data["grad_norms"],
        "smoothness_scores": data["smoothness_scores"],
        "collision_rates": data["collision_rates"],
        "horizon_errors": data["horizon_errors"]
    }
    
    
def trajectory_loss(pred, gt):
    """
    pred: (B, H, 2)
    gt: (B, H, 2)
    """

    ade = torch.mean(torch.norm(pred - gt, dim=-1))
    fde = torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1))

    return ade + fde, ade, fde

def compute_smoothness(traj):
    """
    traj: (B,H,2)
    """

    accel = traj[:, 2:] - 2 * traj[:, 1:-1] + traj[:, :-2]

    smoothness = torch.norm(accel, dim=-1).mean()

    return smoothness.item()

def collision_rate(pred, neighbors, threshold=2.0):
    """
    pred: (B,H,2)
    neighbors: (B,K,T,6)
    """

    B, H, _ = pred.shape

    nbr_pos = neighbors[:, :, -1, :2]

    collisions = 0

    for b in range(B):

        for h in range(H):

            ego_xy = pred[b, h]

            dist = torch.norm(
                nbr_pos[b] - ego_xy.unsqueeze(0),
                dim=-1
            )

            if (dist < threshold).any():
                collisions += 1
                break

    return collisions / B

def per_horizon_error(pred, gt):
    """
    pred: (B,H,2)
    gt: (B,H,2)
    """

    err = torch.norm(pred - gt, dim=-1)

    return err.mean(dim=0).cpu().numpy()

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )