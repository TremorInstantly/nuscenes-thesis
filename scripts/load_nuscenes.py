import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from config import DATAROOT, VERSION
from torch.utils.data import Dataset

# ================= LOAD =================
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False) # Load raw nuscenes dataset from local storage

MAPS = {
    name: NuScenesMap(dataroot=DATAROOT, map_name=name)
    for name in [
        "boston-seaport",
        "singapore-onenorth",
        "singapore-hollandvillage",
        "singapore-queenstown"
    ]
}

class NuScenesDataset(Dataset): # To Load preprocessed dataset for training
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)

        self.ego_hist = data["ego_hist"]
        self.ego_fut = data["ego_fut"]
        self.neighbors = data["neighbors"]
        self.lanes = data["lanes"]
        filtered_ego_hist = []
        filtered_ego_fut = []
        filtered_neighbors = []
        filtered_lanes = []
        threshold = 1.0  # meters
        removed = 0
        for i in range(len(self.ego_hist)):
            fut = np.array(self.ego_fut[i], dtype=np.float32)
            displacement = np.linalg.norm(
                fut[-1, :2] - fut[0, :2]
            )
            if displacement < threshold:
                removed += 1
                continue
            filtered_ego_hist.append(self.ego_hist[i])
            filtered_ego_fut.append(self.ego_fut[i])
            filtered_neighbors.append(self.neighbors[i])
            filtered_lanes.append(self.lanes[i])

        self.ego_hist = filtered_ego_hist
        self.ego_fut = filtered_ego_fut
        self.neighbors = filtered_neighbors
        self.lanes = filtered_lanes
        print(f"Removed stationary scenes: {removed}")
        print(f"Remaining scenes: {len(self.ego_hist)}")
        all_ego = []
        for ego in self.ego_hist:
            ego = np.array(ego, dtype=np.float32)
            all_ego.append(ego[:, 2:4])
        all_ego = np.concatenate(all_ego, axis=0)
        self.global_mean = all_ego.mean(axis=0)
        self.global_std = all_ego.std(axis=0)
        self.global_std[self.global_std < 1e-6] = 1.0
        print("Global Mean:", self.global_mean)
        print("Global Std :", self.global_std)

    def __len__(self):
        return len(self.ego_hist)

    def normalize(self, x):
        x = x.astype(np.float32)
        return (
            (x - self.global_mean)
            / self.global_std
        )

    def to_egocentric(self, ego, neighbors, lanes, fut):
        ref_x, ref_y = ego[-1, 0], ego[-1, 1]

        ego[:, 0] -= ref_x
        ego[:, 1] -= ref_y

        neighbors[:, :, 0] -= ref_x
        neighbors[:, :, 1] -= ref_y

        lanes[:, :, 0] -= ref_x
        lanes[:, :, 1] -= ref_y
        
        fut[:, 0] -= ref_x
        fut[:, 1] -= ref_y

        return ego, neighbors, lanes, fut

    def process_yaw(self, arr):
        yaw = arr[..., 4]
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        return np.concatenate([arr[..., :4], sin_yaw[..., None], cos_yaw[..., None]], axis=-1)

    def __getitem__(self, idx):
        ego = np.array(self.ego_hist[idx], dtype=np.float32)
        fut = np.array(self.ego_fut[idx], dtype=np.float32)
        nbr = np.array(self.neighbors[idx], dtype=np.float32)
        lane = np.array(self.lanes[idx], dtype=np.float32)

        K = 5
        T = ego.shape[0]

        # Convert object array safely
        nbr_list = self.neighbors[idx]

        # Empty neighbors
        if len(nbr_list) == 0:
            nbr = np.zeros((K, T, 5), dtype=np.float32)

        else:
            nbr = np.array(nbr_list, dtype=np.float32)

        # Ensure 3D
        if nbr.ndim == 2:
            nbr = nbr[None, ...]

        # Pad to fixed K
        if nbr.shape[0] < K:
            pad = np.zeros((K - nbr.shape[0], T, 5), dtype=np.float32)
            nbr = np.concatenate([nbr, pad], axis=0)

        elif nbr.shape[0] > K:
            nbr = nbr[:K]

        ego, nbr, lane, fut = self.to_egocentric(ego, nbr, lane, fut)

        # Normalize v,a
        ego[:, 2:4] = self.normalize(ego[:, 2:4])
        nbr[:, :, 2:4] = self.normalize(nbr[:, :, 2:4])

        # Convert yaw → sin/cos
        ego = self.process_yaw(ego)      # (T,6)
        nbr = self.process_yaw(nbr)      # (K,T,6)

        return (
            torch.tensor(ego, dtype=torch.float32),
            torch.tensor(nbr, dtype=torch.float32),
            torch.tensor(lane, dtype=torch.float32),
            torch.tensor(fut, dtype=torch.float32)
        )