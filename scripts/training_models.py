import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import get_neighbor_mask

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            dropout=0.2,
            num_layers=2
        )

    def forward(self, x):
        # x: (B, T, D)
        out, (h, _) = self.lstm(x)
        return h[-1], out  # (B, H), (B, T, H)
    
class FeatureEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ego_enc = TrajectoryEncoder(6, hidden_dim)
        self.nbr_enc = TrajectoryEncoder(6, hidden_dim)
        self.lane_enc = TrajectoryEncoder(2, hidden_dim)

    def forward(self, ego, neighbors, lanes):
        """
        ego: (B, T, 6)
        neighbors: (B, K, T, 6)
        lanes: (B, N, L, 2)
        """

        B, K, T, D = neighbors.shape
        _, N, L, _ = lanes.shape

        # Ego
        h_ego, ego_seq = self.ego_enc(ego)

        # Neighbors
        neighbors = neighbors.view(B*K, T, D)
        h_nbr, _ = self.nbr_enc(neighbors)
        h_nbr = h_nbr.view(B, K, -1)

        # Lanes
        lanes = lanes.view(B*N, L, 2)
        h_lane, _ = self.lane_enc(lanes)
        h_lane = h_lane.view(B, N, -1)

        return h_ego, ego_seq, h_nbr, h_lane
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, future_steps=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, future_steps * 2)
        )
        self.future_steps = future_steps

    def forward(self, z):
        out = self.fc(z)
        return out.view(-1, self.future_steps, 2)
    
class Model1_EgoOnly(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_dim)
        self.decoder = Decoder(hidden_dim, hidden_dim)

    def forward(self, ego, neighbors, lanes):
        h_ego, _, _, _ = self.encoder(ego, neighbors, lanes)
        return self.decoder(h_ego)
    
class Model2_Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_dim)
        self.decoder = Decoder(hidden_dim * 3, hidden_dim)

    def forward(self, ego, neighbors, lanes):
        mask = get_neighbor_mask(neighbors)

        h_ego, _, h_nbr, h_lane = self.encoder(ego, neighbors, lanes)

        attn_scores = torch.matmul(h_nbr, h_ego.unsqueeze(-1)).squeeze(-1)

        # Mask invalid neighbors
        attn_scores[~mask] = -1e9

        attn_weights = F.softmax(attn_scores, dim=1)
        h_nbr_agg = torch.sum(h_nbr * attn_weights.unsqueeze(-1), dim=1)

        h_lane_agg = torch.mean(h_lane, dim=1)

        z = torch.cat([h_ego, h_nbr_agg, h_lane_agg], dim=-1)
        return self.decoder(z)
    
class Model3_TopK(nn.Module):
    def __init__(self, hidden_dim, K=5):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_dim)
        self.decoder = Decoder(hidden_dim * 3, hidden_dim)
        self.K = K

    def forward(self, ego, neighbors, lanes):
        mask = get_neighbor_mask(neighbors)

        # RAW physics features (IMPORTANT)
        nbr_xy = neighbors[:, :, -1, :2]
        ego_xy = ego[:, -1, :2]

        nbr_v = neighbors[:, :, -1, 2]
        ego_v = ego[:, -1, 2]

        nbr_yaw = torch.atan2(neighbors[:, :, -1, 4], neighbors[:, :, -1, 5])
        ego_yaw = torch.atan2(ego[:, -1, 4], ego[:, -1, 5])

        dist = torch.norm(nbr_xy - ego_xy.unsqueeze(1), dim=-1)
        vel_diff = torch.abs(nbr_v - ego_v.unsqueeze(1))
        yaw_diff = torch.abs(nbr_yaw - ego_yaw.unsqueeze(1))

        score = 1/(dist+1e-3) + 0.5*vel_diff + 0.2*yaw_diff
        score[~mask] = -1e9

        _, idx = torch.topk(score, self.K, dim=1)

        h_ego, _, h_nbr, h_lane = self.encoder(ego, neighbors, lanes)

        idx_exp = idx.unsqueeze(-1).expand(-1, -1, h_nbr.size(-1))
        h_topk = torch.gather(h_nbr, 1, idx_exp)

        h_nbr_agg = torch.mean(h_topk, dim=1)
        h_lane_agg = torch.mean(h_lane, dim=1)

        z = torch.cat([h_ego, h_nbr_agg, h_lane_agg], dim=-1)
        return self.decoder(z)
    
class Model4_Gated(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2 + 3, 1)  # + physics features
        self.lane_fc = nn.Linear(hidden_dim, 1)
        self.decoder = Decoder(hidden_dim * 3, hidden_dim)

    def forward(self, ego, neighbors, lanes):
        mask = get_neighbor_mask(neighbors)

        # Physics features
        nbr_xy = neighbors[:, :, -1, :2]
        ego_xy = ego[:, -1, :2]

        nbr_v = neighbors[:, :, -1, 2]
        ego_v = ego[:, -1, 2]

        nbr_yaw = torch.atan2(neighbors[:, :, -1, 4], neighbors[:, :, -1, 5])
        ego_yaw = torch.atan2(ego[:, -1, 4], ego[:, -1, 5])

        dist = torch.norm(nbr_xy - ego_xy.unsqueeze(1), dim=-1)
        vel_diff = torch.abs(nbr_v - ego_v.unsqueeze(1))
        yaw_diff = torch.abs(nbr_yaw - ego_yaw.unsqueeze(1))

        physics = torch.stack([dist, vel_diff, yaw_diff], dim=-1)

        h_ego, ego_seq, h_nbr, h_lane = self.encoder(ego, neighbors, lanes)

        B, K, H = h_nbr.shape

        h_ego_exp = h_ego.unsqueeze(1).expand(-1, K, -1)
        gate_input = torch.cat([h_ego_exp, h_nbr, physics], dim=-1)

        gates = torch.sigmoid(self.gate(gate_input)).squeeze(-1)

        # Convert mask to float
        mask_f = mask.float()

        # Apply mask WITHOUT inplace modification
        gates = gates * mask_f

        h_nbr_agg = torch.sum(h_nbr * gates.unsqueeze(-1), dim=1)

        lane_scores = self.lane_fc(h_lane).squeeze(-1)
        lane_weights = torch.softmax(lane_scores, dim=1)
        h_lane_agg = torch.sum(h_lane * lane_weights.unsqueeze(-1), dim=1)

        h_temporal = torch.mean(ego_seq, dim=1)
        h_ego_enhanced = h_ego + h_temporal

        z = torch.cat([h_ego_enhanced, h_nbr_agg, h_lane_agg], dim=-1)
        z = z + torch.cat([h_ego, h_ego, h_ego], dim=-1)

        return self.decoder(z)
    
class Model5_GatedTopK(nn.Module):
    """
    Hybrid model:
    - learns interaction scores (gating)
    - applies Top-K sparsification on learned scores
    - aggregates selected neighbors only
    """

    def __init__(self, hidden_dim, K=5):
        super().__init__()

        self.encoder = FeatureEncoder(hidden_dim)
        self.K = K

        # learned interaction score (ego + neighbor + physics)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # lane attention
        self.lane_fc = nn.Linear(hidden_dim, 1)

        self.decoder = Decoder(hidden_dim * 3, hidden_dim)

    def forward(self, ego, neighbors, lanes):

        # ================= MASK VALID NEIGHBORS =================
        mask = get_neighbor_mask(neighbors)  # (B, K)

        # ================= ENCODE TRAJECTORIES =================
        h_ego, ego_seq, h_nbr, h_lane = self.encoder(ego, neighbors, lanes)

        B, K, H = h_nbr.shape

        # ================= PHYSICS FEATURES =================
        ego_xy = ego[:, -1, :2]
        nbr_xy = neighbors[:, :, -1, :2]

        ego_v = ego[:, -1, 2]
        nbr_v = neighbors[:, :, -1, 2]

        ego_yaw = torch.atan2(ego[:, -1, 4], ego[:, -1, 5])
        nbr_yaw = torch.atan2(neighbors[:, :, -1, 4], neighbors[:, :, -1, 5])

        dist = torch.norm(nbr_xy - ego_xy.unsqueeze(1), dim=-1)
        vel_diff = torch.abs(nbr_v - ego_v.unsqueeze(1))
        yaw_diff = torch.abs(nbr_yaw - ego_yaw.unsqueeze(1))

        physics = torch.stack([dist, vel_diff, yaw_diff], dim=-1)  # (B, K, 3)

        # ================= LEARNED INTERACTION SCORE (GATING) =================
        h_ego_exp = h_ego.unsqueeze(1).expand(-1, K, -1)

        mlp_input = torch.cat([h_ego_exp, h_nbr, physics], dim=-1)
        learn_score = self.score_mlp(mlp_input).squeeze(-1)

        score = learn_score

        score = score.masked_fill(~mask, -1e9)

        # ================= TOP-K SELECTION (STRUCTURAL SPARSITY) =================
        topk_vals, topk_idx = torch.topk(score, self.K, dim=1)

        # build binary mask
        sparse_mask = torch.zeros_like(score)
        sparse_mask.scatter_(1, topk_idx, 1.0)

        # final gated weights
        gates = torch.sigmoid(score) * sparse_mask

        # normalize for stability
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)

        # ================= NEIGHBOR AGGREGATION =================
        h_nbr_agg = torch.sum(h_nbr * gates.unsqueeze(-1), dim=1)

        # ================= LANE AGGREGATION =================
        lane_scores = self.lane_fc(h_lane).squeeze(-1)
        lane_weights = F.softmax(lane_scores, dim=1)
        h_lane_agg = torch.sum(h_lane * lane_weights.unsqueeze(-1), dim=1)

        # ================= TEMPORAL ENCODING =================
        h_temporal = torch.mean(ego_seq, dim=1)
        h_ego_enhanced = h_ego + h_temporal

        # ================= FINAL FUSION =================
        z = torch.cat([h_ego_enhanced, h_nbr_agg, h_lane_agg], dim=-1)
        
        z = z + torch.cat([h_ego, h_ego, h_ego], dim=-1)

        pred = self.decoder(z)

        return pred, gates, lane_weights, topk_idx