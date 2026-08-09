import numpy as np
from tqdm import tqdm
from config import  T, H, K, N_LANES, LANE_POINTS, RADIUS, PREPROCESSED_SAVE_PATH
from load_nuscenes import nusc, MAPS
from utils import quat_to_yaw, compute_kinematics

# ================= FULL AGENT TRAJECTORIES =================
def build_agent_tracks(scene):
    tracks = {}
    token = scene["first_sample_token"]
    while token != "":
        sample = nusc.get("sample", token)
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            inst = ann["instance_token"]
            x, y = ann["translation"][:2]
            yaw = quat_to_yaw(ann["rotation"])
            if inst not in tracks:
                tracks[inst] = []
            tracks[inst].append((x, y, yaw))
        token = sample["next"]

    return tracks

# ================= NEIGHBORS =================
def get_neighbors(sample, ego_xy):
    agents = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        inst = ann["instance_token"]
        x, y = ann["translation"][:2]
        dist = np.linalg.norm([x - ego_xy[0], y - ego_xy[1]])
        agents.append((dist, inst))
    agents.sort(key=lambda x: x[0])

    return [a[1] for a in agents[:K]]

# ================= LANES =================
def resample(points, num):
    points = np.array(points)
    if len(points) < 2:
        return np.zeros((num, 2))
    d = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(-1))
    d = np.insert(np.cumsum(d), 0, 0)
    if d[-1] == 0:
        return np.repeat(points[:1], num, axis=0)
    t = np.linspace(0, d[-1], num)

    out = []
    for ti in t:
        idx = np.searchsorted(d, ti) - 1
        idx = np.clip(idx, 0, len(points) - 2)
        r = (ti - d[idx]) / (d[idx + 1] - d[idx] + 1e-6)
        p = points[idx] * (1 - r) + points[idx + 1] * r
        out.append(p)

    return np.array(out)

def get_lane_polylines(nusc_map, x, y):

    lanes_out = []
    records = nusc_map.get_records_in_radius(
        x, y, RADIUS,
        layer_names=["lane"]
    )
    lane_tokens = records.get("lane", [])
    for lane_token in lane_tokens:
        try:
            lane = nusc_map.get("lane", lane_token)
            nodes = lane.get("exterior_node_tokens", [])
            pts = []
            for n in nodes:
                node = nusc_map.get("node", n)
                pts.append([node["x"], node["y"]])
            if len(pts) > 2:
                lanes_out.append(resample(pts, LANE_POINTS))
        except:
            continue

    return lanes_out

# ================= MAIN =================
ego_hist_all = []
ego_fut_all = []
neighbors_all = []
lanes_all = []

for scene in tqdm(nusc.scene):
    log = nusc.get("log", scene["log_token"])
    nusc_map = MAPS[log["location"]]
    traj_lookup = build_agent_tracks(scene)
    # ego trajectory timeline
    token = scene["first_sample_token"]
    traj = []
    while token != "":
        sample = nusc.get("sample", token)
        ego = nusc.get("ego_pose", sample["data"]["LIDAR_TOP"])
        x, y = ego["translation"][:2]
        yaw = quat_to_yaw(ego["rotation"])
        traj.append((token, x, y, yaw))
        token = sample["next"]
    if len(traj) < T + H:
        continue
    
    max_i = len(traj) - (T + H)
    for i in range(max_i):
        # ================= EGO =================
        ego_hist = compute_kinematics(
            [(x, y, yaw) for (_, x, y, yaw) in traj[i:i + T]]
        )
        ego_fut = [[x, y] for (_, x, y, _) in traj[i + T:i + T + H]]
        ego_xy = traj[i + T - 1][1:3]
        sample_token = traj[i + T - 1][0]
        sample = nusc.get("sample", sample_token)

        # ================= NEIGHBORS =================
        neighbor_tokens = get_neighbors(sample, ego_xy)
        neigh_data = []
        ego_start_token = traj[i][0]
        ego_end_token = traj[i + T - 1][0]
        ego_samples = []

        # sample timeline for alignment
        token = ego_start_token
        for _ in range(T):
            if token == "":
                break
            ego_samples.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]

        for nt in neighbor_tokens:
            agent_full = traj_lookup.get(nt, [])
            aligned_seq = []
            for t in range(T):
                if t >= len(ego_samples):
                    aligned_seq.append([0, 0, 0])
                    continue
                ego_token = ego_samples[t]
                sample = nusc.get("sample", ego_token)
                
                found = False
                for ann_token in sample["anns"]:
                    ann = nusc.get("sample_annotation", ann_token)
                    if ann["instance_token"] == nt:
                        x, y = ann["translation"][:2]
                        yaw = quat_to_yaw(ann["rotation"])
                        aligned_seq.append([x, y, yaw])
                        found = True
                        break

                if not found:
                    aligned_seq.append([0, 0, 0])

            neigh_data.append(compute_kinematics(aligned_seq))

        # ================= LANES =================
        lanes = get_lane_polylines(nusc_map, ego_xy[0], ego_xy[1])
        while len(lanes) < N_LANES:
            lanes.append(np.zeros((LANE_POINTS, 2), dtype=np.float32))
        lanes = lanes[:N_LANES]

        # ================= STORE VALUES =================
        ego_hist_all.append(ego_hist)
        ego_fut_all.append(ego_fut)
        neighbors_all.append(neigh_data)
        lanes_all.append(lanes)

# ================= SAVE =================

ego_hist_all = np.array(ego_hist_all, dtype=object)
ego_fut_all = np.array(ego_fut_all, dtype=object)
neighbors_all = np.array(neighbors_all, dtype=object)
lanes_all = np.array(lanes_all, dtype=object)

np.savez_compressed(
    PREPROCESSED_SAVE_PATH,
    ego_hist=ego_hist_all,
    ego_fut=ego_fut_all,
    neighbors=neighbors_all,
    lanes=lanes_all
)

print("Saved:", PREPROCESSED_SAVE_PATH)