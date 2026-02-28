import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, f1_score, precision_score, recall_score
)


# =========================
# 0) Reproducibility
# =========================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random seed set to {seed}")


# =========================
# 1) CONFIG
# =========================
SEQ_LEN = 24
METEO_LEN = 72
HORIZONS = [3, 6, 12, 24]
EPOCHS = 70
LR = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
TEST_SIZE = 0.2

EPS_Q = 1e-6
SUPERVISE_ALL_ACTUAL_NODES = True
TARGET_NODE = "K"
EVENT_CONTROL_NODE = "K"

USE_DISTANCE_DECAY = True
EDGE_W_CLIP = 20.0

VNODE_BG_RATIO = 0.2

# 只预测 TN/TP
W_TN = 1.0
W_TP = 1.0

# gate loss
LAMBDA_GATE = 1.0
GATE_WARMUP_EPOCHS = 15

# 气象输入模式
METEO_MODE = "full"

# 输出：是否额外导出所有“实测点”的预测（默认只导出K点）
EXPORT_ALL_ACTUAL_NODES = False

MODEL_NAME = "PredGate"


# =========================
# 2) Standard Scaler
# =========================
class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, x: np.ndarray):
        self.mean = float(np.mean(x))
        self.std = float(np.std(x))
        if self.std < 1e-6:
            self.std = 1.0

    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray):
        return x * self.std + self.mean


# =========================
# 3) Time features (sin/cos)
# =========================
def make_time_features(time_series: pd.Series) -> pd.DataFrame:
    t = pd.to_datetime(time_series, errors="coerce")
    if t.isna().any():
        t = t.ffill().bfill()

    hour = t.dt.hour.values.astype(np.float32)
    dow = t.dt.dayofweek.values.astype(np.float32)
    doy = t.dt.dayofyear.values.astype(np.float32)

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin  = np.sin(2 * np.pi * dow / 7.0)
    dow_cos  = np.cos(2 * np.pi * dow / 7.0)
    doy_sin  = np.sin(2 * np.pi * doy / 365.0)
    doy_cos  = np.cos(2 * np.pi * doy / 365.0)

    return pd.DataFrame({
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin,   "dow_cos": dow_cos,
        "doy_sin": doy_sin,   "doy_cos": doy_cos
    })


# =========================
# 4) Data Loading
# =========================
def load_flow(flow_file, nodes_df):
    df = pd.read_csv(flow_file)
    if "time_step" not in df.columns:
        raise ValueError("flow_data.csv 必须包含 time_step 列")

    out = pd.DataFrame()
    out["time_step"] = df["time_step"].copy()

    data = df.drop(columns=["time_step"], errors="ignore")
    data = data.apply(pd.to_numeric, errors="coerce").ffill().bfill()

    for n in nodes_df["node_id"].tolist():
        out[n] = data[n] if n in data.columns else np.nan

    return out


def load_quality_TN_TP(qual_file, nodes_df):
    df = pd.read_csv(qual_file)
    if "time_step" not in df.columns:
        raise ValueError("quality_data.csv 必须包含 time_step 列")

    tn_df = pd.DataFrame()
    tp_df = pd.DataFrame()
    tn_df["time_step"] = df["time_step"].copy()
    tp_df["time_step"] = df["time_step"].copy()

    data = df.drop(columns=["time_step"], errors="ignore")

    for n in nodes_df["node_id"].tolist():
        tn_df[n] = pd.to_numeric(data[n], errors="coerce") if n in data.columns else np.nan
        tp_col = f"{n}_TP"
        tp_df[n] = pd.to_numeric(data[tp_col], errors="coerce") if tp_col in data.columns else np.nan

    tn_df = tn_df.ffill().bfill()
    tp_df = tp_df.ffill().bfill()

    if tp_df.drop(columns=["time_step"]).isna().all().all():
        raise ValueError("未检测到任何 *_TP 列。请确认 quality_data.csv 已加入 TP 列。")

    return tn_df, tp_df


def load_meteo(meteo_file, mode="full"):
    df = pd.read_csv(meteo_file)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    time_col = None
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            if c.lower() == "time_step":
                time_col = c
                break

    if time_col is not None:
        t = df[time_col].copy()
    else:
        t = None

    time_like_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "datetime", "stamp"])]
    df = df.drop(columns=time_like_cols, errors="ignore")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how='all')
    df = df.ffill().bfill().fillna(0.0)

    rename_map = {
        "temp2m": "Temp2m",
        "rh2m": "RH2m",
        "rain": "Rain",
        "pressure": "Pressure",
        "et0": "ET0",
        "wind10m": "Wind10m"
    }
    df.rename(columns={c: rename_map.get(c.lower(), c) for c in df.columns}, inplace=True)

    required = ["Temp2m", "RH2m", "Rain", "Pressure", "ET0", "Wind10m"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"气象数据缺少关键列: {missing}")

    out = df[required].astype(np.float32).copy()

    if mode == "no_rain":
        out["Rain"] = 0.0
    elif mode == "rain_only":
        for c in required:
            if c != "Rain":
                out[c] = 0.0
    elif mode == "full":
        pass
    else:
        raise ValueError(f"未知 METEO_MODE={mode}")

    if t is not None:
        out.insert(0, "time_step", t.values)

    return out


def load_landuse(file, nodes_df):
    df = pd.read_csv(file)
    df["farmland"] = df.get("Cropland", 0)
    df["builtup"] = df.get("Traffic route", 0) + df.get("Building", 0)
    df["forest"] = df.get("Tree cover", 0)
    df["water"] = df.get("Water", 0)
    df["other"] = df.get("Barren and sparse vegetation", 0)

    df = df.set_index("node_id").reindex(nodes_df["node_id"]).fillna(0.0)
    x = torch.tensor(df[["farmland", "builtup", "forest", "water", "other"]].values.astype(np.float32))
    m = x.abs().max()
    if m > 0:
        x = x / m
    return x


def load_distance(distance_file, nodes_df):
    df = pd.read_csv(distance_file)
    if df.columns[0] != "node_id":
        raise ValueError("Distance.csv 第 1 列必须为 node_id")
    df = df.set_index("node_id").reindex(nodes_df["node_id"])
    df = df[nodes_df["node_id"]]

    dist = torch.tensor(df.values.astype(np.float32))
    dist[dist < 1e-6] = 1e-6

    positive = dist[dist > 0]
    L = positive.mean().item() if positive.numel() > 0 else 1.0
    decay = torch.exp(-dist / (L + 1e-6))
    return dist, decay


# =========================
# 5) Align by time_step + append time features to meteo
# =========================
def align_by_time_step(flow_df, tn_df, tp_df, meteo_df):
    flow_df = flow_df.copy()
    tn_df = tn_df.copy()
    tp_df = tp_df.copy()
    meteo_df = meteo_df.copy()

    flow_df["time_dt"] = pd.to_datetime(flow_df["time_step"], errors="coerce")
    tn_df["time_dt"]   = pd.to_datetime(tn_df["time_step"], errors="coerce")
    tp_df["time_dt"]   = pd.to_datetime(tp_df["time_step"], errors="coerce")

    flow_df = flow_df.dropna(subset=["time_dt"])
    tn_df   = tn_df.dropna(subset=["time_dt"])
    tp_df   = tp_df.dropna(subset=["time_dt"])

    flow_i = flow_df.set_index("time_dt")
    tn_i   = tn_df.set_index("time_dt")
    tp_i   = tp_df.set_index("time_dt")

    idx = flow_i.index.intersection(tn_i.index).intersection(tp_i.index)
    if len(idx) == 0:
        raise ValueError("time_step 对齐后交集为空：请检查三份数据的 time_step 格式/覆盖范围是否一致。")

    flow_i = flow_i.loc[idx]
    tn_i   = tn_i.loc[idx]
    tp_i   = tp_i.loc[idx]

    flow_aligned = flow_i.reset_index(drop=False).rename(columns={"time_dt": "time_dt"})
    tn_aligned   = tn_i.reset_index(drop=False).rename(columns={"time_dt": "time_dt"})
    tp_aligned   = tp_i.reset_index(drop=False).rename(columns={"time_dt": "time_dt"})

    node_cols = [c for c in flow_df.columns if c not in ["time_step", "time_dt"]]
    flow_aligned = flow_aligned[["time_step"] + node_cols].reset_index(drop=True)
    tn_aligned   = tn_aligned[["time_step"] + node_cols].reset_index(drop=True)
    tp_aligned   = tp_aligned[["time_step"] + node_cols].reset_index(drop=True)

    if "time_step" in meteo_df.columns:
        meteo_df = meteo_df.copy()
        meteo_df["time_dt"] = pd.to_datetime(meteo_df["time_step"], errors="coerce")
        meteo_df = meteo_df.dropna(subset=["time_dt"]).set_index("time_dt")

        idx2 = meteo_df.index.intersection(pd.to_datetime(flow_aligned["time_step"], errors="coerce"))
        flow_time_dt = pd.to_datetime(flow_aligned["time_step"], errors="coerce")
        keep_mask = flow_time_dt.isin(idx2)

        flow_aligned = flow_aligned.loc[keep_mask].reset_index(drop=True)
        tn_aligned   = tn_aligned.loc[keep_mask].reset_index(drop=True)
        tp_aligned   = tp_aligned.loc[keep_mask].reset_index(drop=True)
        flow_time_dt = flow_time_dt.loc[keep_mask].reset_index(drop=True)

        met = meteo_df.loc[flow_time_dt.values]
        meteo_aligned_6 = met.drop(columns=["time_step"], errors="ignore").reset_index(drop=True)
    else:
        T = min(len(flow_aligned), len(meteo_df))
        flow_aligned = flow_aligned.iloc[:T].reset_index(drop=True)
        tn_aligned   = tn_aligned.iloc[:T].reset_index(drop=True)
        tp_aligned   = tp_aligned.iloc[:T].reset_index(drop=True)
        meteo_aligned_6 = meteo_df.iloc[:T].reset_index(drop=True)

    time_feat = make_time_features(flow_aligned["time_step"])
    meteo_aligned = pd.concat([meteo_aligned_6.reset_index(drop=True),
                               time_feat.reset_index(drop=True)], axis=1)

    return flow_aligned, tn_aligned, tp_aligned, meteo_aligned


# =========================
# 6) River Network
# =========================
def create_river_network():
    nodes_data = {
        'node_id': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
        'x': [0, 2, 0, 0, 4, 2, 6, 2, 8, 6, 10],
        'y': [2, 2, 1, 3, 2, 1, 2, 3, 2, 1, 2],
        'node_type': ['actual', 'virtual', 'actual', 'actual',
                      'virtual', 'actual', 'virtual', 'actual',
                      'virtual', 'actual', 'actual']
    }
    edges_data = {
        'source': ['A', 'C', 'D', 'B', 'F', 'B', 'E', 'H', 'E', 'G', 'J', 'G', 'I'],
        'target': ['B', 'B', 'B', 'E', 'E', 'E', 'G', 'G', 'G', 'I', 'I', 'I', 'K']
    }
    return pd.DataFrame(nodes_data), pd.DataFrame(edges_data)


# =========================
# 7) Virtual nodes estimation
# =========================
def estimate_virtual_nodes(flow_df, tn_df, tp_df, nodes_df, decay_matrix, bg_ratio=VNODE_BG_RATIO):
    est_flow = flow_df.copy()
    est_tn = tn_df.copy()
    est_tp = tp_df.copy()

    node_ids = nodes_df["node_id"].tolist()
    node_to_idx = {n: i for i, n in enumerate(node_ids)}

    for n in node_ids:
        if n not in est_flow.columns:
            est_flow[n] = np.nan
        if n not in est_tn.columns:
            est_tn[n] = np.nan
        if n not in est_tp.columns:
            est_tp[n] = np.nan

    mix_rules = {
        "B": ["A", "C", "D"],
        "E": ["B", "F"],
        "G": ["E", "H"],
        "I": ["G", "J"],
    }

    def mix_quality(src_vals, flow_vals, alphas):
        flow_abs = np.abs(flow_vals)
        weight = flow_abs * alphas[None, :]
        mean_up = np.nanmean(src_vals, axis=1)

        Q_total = np.nansum(flow_abs, axis=1)
        Q_bg = bg_ratio * Q_total
        C_bg = mean_up

        num = np.nansum(weight * src_vals, axis=1)
        den = np.nansum(weight, axis=1)
        eps = 1e-3
        out = (num + Q_bg * C_bg) / (den + Q_bg + eps)

        bad = (~np.isfinite(out)) | ((den + Q_bg) < eps)
        out[bad] = mean_up[bad]
        out = np.clip(out, 0.0, 1e6)
        return out

    for tgt, srcs in mix_rules.items():
        valid_srcs = [s for s in srcs if s in est_flow.columns]
        if not valid_srcs:
            continue

        flow_vals = est_flow[valid_srcs].values.astype(np.float32)
        tn_vals = est_tn[valid_srcs].values.astype(np.float32)
        tp_vals = est_tp[valid_srcs].values.astype(np.float32)

        alphas = np.array([float(decay_matrix[node_to_idx[s], node_to_idx[tgt]].item())
                           for s in valid_srcs], dtype=np.float32)

        est_tn[tgt] = mix_quality(tn_vals, flow_vals, alphas)
        est_tp[tgt] = mix_quality(tp_vals, flow_vals, alphas)

        flow_sum = np.nansum(flow_vals, axis=1)
        sign_sum = np.sign(flow_sum)
        Q_total = np.nansum(np.abs(flow_vals), axis=1)
        Q_bg = bg_ratio * Q_total
        est_flow[tgt] = flow_sum + sign_sum * Q_bg

    est_flow = est_flow.ffill().bfill().fillna(0.0)
    est_tn = est_tn.ffill().bfill().fillna(0.0)
    est_tp = est_tp.ffill().bfill().fillna(0.0)

    return est_flow, est_tn, est_tp


# =========================
# 8) Dataset builder (no leakage split)
# =========================
def create_time_series_dataset(
    nodes_df,
    flow_df, tn_df, tp_df,
    meteo_df,
    landuse_tensor,
    decay_matrix,
    seq_len=SEQ_LEN,
    meteo_len=METEO_LEN,
    test_size=TEST_SIZE
):
    node_ids = nodes_df["node_id"].tolist()
    N = len(node_ids)
    node_to_idx = {n: i for i, n in enumerate(node_ids)}

    Q_raw = torch.tensor(flow_df[node_ids].values.astype(np.float32).T)  # [N,T]
    TN_raw = torch.tensor(tn_df[node_ids].values.astype(np.float32).T)
    TP_raw = torch.tensor(tp_df[node_ids].values.astype(np.float32).T)

    Qmag_raw = torch.log1p(Q_raw.abs())

    T = Q_raw.shape[1]
    max_h = max(HORIZONS)
    last_start = T - seq_len - max_h
    if last_start <= 0:
        raise ValueError("时间序列长度太短，无法构造多步样本。")

    split_point = int(T * (1 - test_size))

    qmag_scaler = StandardScaler()
    tn_scaler = StandardScaler()
    tp_scaler = StandardScaler()

    qmag_scaler.fit(Qmag_raw[:, :split_point].reshape(-1).numpy())
    tn_scaler.fit(TN_raw[:, :split_point].reshape(-1).numpy())
    tp_scaler.fit(TP_raw[:, :split_point].reshape(-1).numpy())

    Qmag = torch.tensor(qmag_scaler.transform(Qmag_raw.numpy()), dtype=torch.float32)
    TN = torch.tensor(tn_scaler.transform(TN_raw.numpy()), dtype=torch.float32)
    TP = torch.tensor(tp_scaler.transform(TP_raw.numpy()), dtype=torch.float32)

    meteo_tensor = torch.tensor(meteo_df.values.astype(np.float32))
    meteo_dim = meteo_tensor.shape[1]

    LU = landuse_tensor
    LU_expand = LU.unsqueeze(1).repeat(1, seq_len, 1)
    Fdim = 3 + LU.shape[1]

    rules = [
        ("A", "A", "B"),
        ("C", "C", "B"),
        ("D", "D", "B"),
        ("B", "B", "E"),
        ("F", "F", "E"),
        ("E", "E", "G"),
        ("H", "H", "G"),
        ("G", "G", "I"),
        ("J", "J", "I"),
        ("I", "I", "K"),
    ]

    def build_edge_at_time(graph_t: int):
        edge_list, w_list = [], []
        Q_t = Q_raw[:, graph_t]

        for ctrl, u_name, v_name in rules:
            c = node_to_idx[ctrl]
            u = node_to_idx[u_name]
            v = node_to_idx[v_name]

            Qc = float(Q_t[c].item())
            if (not np.isfinite(Qc)) or abs(Qc) < EPS_Q:
                continue

            if Qc > 0:
                src, tgt = u, v
            else:
                src, tgt = v, u

            decay = float(decay_matrix[src, tgt].item()) if USE_DISTANCE_DECAY else 1.0
            dyn = np.log1p(abs(Qc))
            dyn = min(dyn, EDGE_W_CLIP)
            w = dyn * decay

            edge_list.append([src, tgt])
            w_list.append(w)

        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(w_list, dtype=torch.float32)
            edge_weight = edge_weight / (edge_weight.max() + 1e-6)

        return edge_index, edge_weight

    def get_meteo_seq(end_idx):
        end = min(end_idx, meteo_tensor.shape[0])
        start = max(0, end - meteo_len)
        seq = meteo_tensor[start:end]
        if seq.shape[0] < meteo_len:
            pad = seq[:1].repeat(meteo_len - seq.shape[0], 1)
            seq = torch.cat([pad, seq], dim=0)
        return seq

    gate_node_idx = node_to_idx[EVENT_CONTROL_NODE]

    train_list, test_list = [], []

    for start in range(0, last_start + 1):
        end = start + seq_len
        target_ts = [(end - 1) + h for h in HORIZONS]

        graph_t = end - 1
        edge_index, edge_weight = build_edge_at_time(graph_t)

        x = torch.zeros(N, seq_len, Fdim, dtype=torch.float32)
        x[:, :, 0] = Qmag[:, start:end]
        x[:, :, 1] = TN[:, start:end]
        x[:, :, 2] = TP[:, start:end]
        x[:, :, 3:] = LU_expand

        y_tn = torch.stack([TN[:, t] for t in target_ts], dim=1)
        y_tp = torch.stack([TP[:, t] for t in target_ts], dim=1)
        y_quality = torch.stack([y_tn, y_tp], dim=-1)  # [N,H,2]

        gate_y = []
        for t in target_ts:
            qk = float(Q_raw[gate_node_idx, t].item())
            gate_y.append(1.0 if abs(qk) > EPS_Q else 0.0)
        gate_y = torch.tensor(gate_y, dtype=torch.float32)

        meteo_seq = get_meteo_seq(end)

        d = Data(
            x=x,
            y_quality=y_quality,
            meteo=meteo_seq,
            edge_index=edge_index,
            edge_attr=edge_weight,
            gate_y=gate_y
        )
        d.node_ids = node_ids
        d.start_idx = start
        d.end_idx = end
        d.graph_t = graph_t
        d.meteo_dim = meteo_dim

        if (end - 1 + max_h) < split_point:
            train_list.append(d)
        elif start >= split_point:
            test_list.append(d)
        else:
            pass

    scalers = {"Qmag": qmag_scaler, "TN": tn_scaler, "TP": tp_scaler}
    return train_list, test_list, scalers, meteo_dim


# =========================
# 9) Model: PredGate
#    - 预测 gate_logits: [B,H]
#    - 用 gate_prob 条件化（通过 gate_inj 加到 meteo_node 上）
# =========================
class MeteoTransformer(nn.Module):
    def __init__(self, meteo_in_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.inp = nn.Linear(meteo_in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, meteo_seq_b):
        z = self.inp(meteo_seq_b)
        z = self.enc(z)
        return z.mean(dim=1)


class PredGateSTGNN(nn.Module):
    def __init__(self, node_feat_dim, num_horizons, num_nodes, meteo_in_dim,
                 gru_hidden=64, meteo_dim=64, gcn_hidden=64):
        super().__init__()
        self.num_h = num_horizons
        self.num_nodes = num_nodes
        self.meteo_in_dim = meteo_in_dim
        self.meteo_dim = meteo_dim
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden

        self.gru = nn.GRU(input_size=node_feat_dim, hidden_size=gru_hidden, batch_first=True)

        self.meteo_enc = MeteoTransformer(meteo_in_dim=meteo_in_dim, d_model=meteo_dim, nhead=4, num_layers=2)
        self.meteo_proj = nn.Linear(meteo_dim, gcn_hidden)

        # 闸门注入：H -> gcn_hidden，然后加到 meteo_node 上（保证结构与 NoGate 尽量一致）
        self.gate_inj = nn.Sequential(
            nn.Linear(num_horizons, 128),
            nn.ReLU(),
            nn.Linear(128, gcn_hidden)
        )

        self.gcn1 = GCNConv(gru_hidden + gcn_hidden, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.act = nn.ReLU()

        self.quality_head = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.ReLU(),
            nn.Linear(gcn_hidden // 2, num_horizons * 2)
        )

        # gate head：per graph [H]
        self.gate_head = nn.Sequential(
            nn.Linear(meteo_dim + gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_horizons)
        )

    def forward(self, data, k_idx: int, meteo_len: int):
        x = data.x.float()  # [BN,L,F]
        BN, L, _ = x.shape
        B = int(data.num_graphs)

        gru_out, _ = self.gru(x)
        node_short = gru_out[:, -1, :]  # [BN,gru_hidden]
        node_short_bnh = node_short.view(B, self.num_nodes, -1)
        k_feat = node_short_bnh[:, k_idx, :]  # [B,gru_hidden]

        meteo = data.meteo.float().view(B, meteo_len, self.meteo_in_dim)
        meteo_embed = self.meteo_enc(meteo)  # [B,meteo_dim]

        gate_logits = self.gate_head(torch.cat([meteo_embed, k_feat], dim=-1))  # [B,H]
        gate_prob = torch.sigmoid(gate_logits)  # [B,H]

        meteo_node = self.meteo_proj(meteo_embed)  # [B,gcn_hidden]
        meteo_node = meteo_node[data.batch]        # [BN,gcn_hidden]

        gate_node = self.gate_inj(gate_prob)       # [B,gcn_hidden]
        gate_node = gate_node[data.batch]          # [BN,gcn_hidden]
        meteo_node = meteo_node + gate_node

        h = torch.cat([node_short, meteo_node], dim=-1)
        h = self.act(self.gcn1(h, data.edge_index, data.edge_attr))
        h = self.act(self.gcn2(h, data.edge_index, data.edge_attr))

        y_hat_quality = self.quality_head(h).view(BN, self.num_h, 2)
        return y_hat_quality, gate_logits


# =========================
# 10) Metrics + Export
# =========================
def compute_reg_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan
    return rmse, mae, r2


def select_gate_thresholds_per_horizon(gate_true, gate_prob, step=0.01):
    H = gate_true.shape[1]
    thresholds = np.zeros(H, dtype=np.float32)
    grid = np.arange(0.05, 0.96, step)
    for i in range(H):
        yt = gate_true[:, i].astype(int)
        yp = gate_prob[:, i]
        best_f1, best_th = -1.0, 0.5
        for th in grid:
            yhat = (yp >= th).astype(int)
            f1 = f1_score(yt, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        thresholds[i] = best_th
    return thresholds


def _as_int_list(x, B: int):
    if x is None:
        return [None] * B
    if torch.is_tensor(x):
        x = x.view(-1).detach().cpu().numpy().tolist()
        return [int(v) for v in x]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    return [int(x)] * B


def export_predictions_csv(
    out_csv: str,
    model_name: str,
    split_name: str,
    node_name: str,
    ys_norm: np.ndarray,      # [S,H,2]
    ps_norm: np.ndarray,      # [S,H,2]
    scalers: dict,
    time_steps: np.ndarray,   # [T]
    end_idxs: list,           # length S
    gate_true: np.ndarray = None,  # [S,H]
    gate_prob: np.ndarray = None,  # [S,H]
    gate_th: np.ndarray = None     # [H]
):
    TN_true = scalers["TN"].inverse(ys_norm[:, :, 0])
    TN_pred = scalers["TN"].inverse(ps_norm[:, :, 0])
    TP_true = scalers["TP"].inverse(ys_norm[:, :, 1])
    TP_pred = scalers["TP"].inverse(ps_norm[:, :, 1])

    rows = []
    S = TN_true.shape[0]
    for s in range(S):
        end = int(end_idxs[s])
        for i, h in enumerate(HORIZONS):
            t_idx = (end - 1) + h
            t_step = time_steps[t_idx] if (0 <= t_idx < len(time_steps)) else None

            row = {
                "model": model_name,
                "split": split_name,
                "node": node_name,
                "sample": s,
                "end_idx": end,
                "horizon_h": h,
                "target_t_idx": t_idx,
                "target_time_step": t_step,
                "TN_true": float(TN_true[s, i]),
                "TN_pred": float(TN_pred[s, i]),
                "TP_true": float(TP_true[s, i]),
                "TP_pred": float(TP_pred[s, i]),
            }

            if gate_true is not None and gate_prob is not None:
                row["gate_true"] = int(gate_true[s, i])
                row["gate_prob"] = float(gate_prob[s, i])
                if gate_th is not None:
                    th = float(gate_th[i])
                    row["gate_th"] = th
                    row["gate_pred"] = int(gate_prob[s, i] >= th)
                else:
                    row["gate_th"] = np.nan
                    row["gate_pred"] = np.nan
            else:
                row["gate_true"] = np.nan
                row["gate_prob"] = np.nan
                row["gate_th"] = np.nan
                row["gate_pred"] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_csv} | rows={len(df)}")


# =========================
# 11) Train/Eval (PredGate)
# =========================
def train_and_eval_pred_gate(nodes_df, train_list, test_list, scalers, num_nodes, meteo_dim, target_node=TARGET_NODE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_ids = train_list[0].node_ids
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    k_idx = node_to_idx[target_node]

    if SUPERVISE_ALL_ACTUAL_NODES:
        actual_nodes = nodes_df[nodes_df["node_type"] == "actual"]["node_id"].tolist()
        sup_idxs = [node_to_idx[n] for n in actual_nodes if n in node_to_idx]
    else:
        sup_idxs = [k_idx]

    sup_mask = torch.zeros(num_nodes, dtype=torch.bool)
    sup_mask[sup_idxs] = True

    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False)

    node_feat_dim = train_list[0].x.shape[-1]
    model = PredGateSTGNN(
        node_feat_dim=node_feat_dim,
        num_horizons=len(HORIZONS),
        num_nodes=num_nodes,
        meteo_in_dim=meteo_dim
    ).to(device)

    # gate pos_weight
    all_gate = []
    for d in train_list:
        all_gate.append(d.gate_y.numpy())
    all_gate = np.stack(all_gate, axis=0)
    pos = all_gate.sum()
    neg = all_gate.size - pos
    pos_weight_gate = torch.tensor([neg / (pos + 1e-6)], device=device, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    print(f"\n==================== RUN {MODEL_NAME} ====================")
    print(f"[INFO] train={len(train_list)} | test={len(test_list)} | meteo_dim={meteo_dim}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        sum_main, sum_gate, sum_total, cnt = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            y_hat_quality, gate_logits = model(batch, k_idx=k_idx, meteo_len=METEO_LEN)

            B = int(batch.num_graphs)
            yq = batch.y_quality.float().view(B, num_nodes, len(HORIZONS), 2)
            predq = y_hat_quality.view(B, num_nodes, len(HORIZONS), 2)

            sup = sup_mask.to(device)
            y_sup = yq[:, sup, :, :]
            p_sup = predq[:, sup, :, :]

            loss_tn = F.mse_loss(p_sup[..., 0], y_sup[..., 0])
            loss_tp = F.mse_loss(p_sup[..., 1], y_sup[..., 1])
            loss_main = W_TN * loss_tn + W_TP * loss_tp

            gate_y = batch.gate_y.float().view(B, -1)
            loss_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_y, pos_weight=pos_weight_gate)

            warm = min(1.0, ep / float(GATE_WARMUP_EPOCHS))
            gate_lambda = LAMBDA_GATE * warm

            loss = loss_main + gate_lambda * loss_gate

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            sum_main += float(loss_main.item())
            sum_gate += float(loss_gate.item())
            sum_total += float(loss.item())
            cnt += 1

        epoch_total = sum_total / max(cnt, 1)
        scheduler.step(epoch_total)

        if ep % 5 == 0:
            print(f"Epoch {ep:03d} | main={sum_main/max(cnt,1):.4f} | gate={sum_gate/max(cnt,1):.4f} "
                  f"| total={sum_total/max(cnt,1):.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    def collect(loader, node_index: int):
        model.eval()
        ys_q, ps_q = [], []
        end_idxs_all = []
        gates_true, gates_prob = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                y_hat_quality, gate_logits = model(batch, k_idx=k_idx, meteo_len=METEO_LEN)
                B = int(batch.num_graphs)

                end_list = _as_int_list(getattr(batch, "end_idx", None), B)
                end_idxs_all.extend(end_list)

                yq = batch.y_quality.float().view(B, num_nodes, len(HORIZONS), 2)
                predq = y_hat_quality.view(B, num_nodes, len(HORIZONS), 2)

                ys_q.append(yq[:, node_index, :, :].cpu().numpy())
                ps_q.append(predq[:, node_index, :, :].cpu().numpy())

                gate_y = batch.gate_y.float().view(B, -1)
                gates_true.append(gate_y.cpu().numpy())
                gates_prob.append(torch.sigmoid(gate_logits).cpu().numpy())

        ys_q = np.concatenate(ys_q, axis=0)
        ps_q = np.concatenate(ps_q, axis=0)
        gates_true = np.concatenate(gates_true, axis=0)
        gates_prob = np.concatenate(gates_prob, axis=0)
        return ys_q, ps_q, end_idxs_all, gates_true, gates_prob

    train_pack_k = collect(train_loader, k_idx)
    test_pack_k = collect(test_loader, k_idx)

    # thresholds from TRAIN
    train_yk, train_pk, train_endk, train_gt, train_gp = train_pack_k
    gate_th = select_gate_thresholds_per_horizon(train_gt, train_gp, step=0.01)
    print("\n[INFO] Gate thresholds selected from TRAIN (max F1):")
    for h, th in zip(HORIZONS, gate_th):
        print(f"  Horizon {h}h: th={th:.2f}")

    return model, train_pack_k, test_pack_k, gate_th, node_to_idx


def print_quality_metrics(title, y_true_norm, y_pred_norm, scalers):
    TN_true = scalers["TN"].inverse(y_true_norm[:, :, 0])
    TN_pred = scalers["TN"].inverse(y_pred_norm[:, :, 0])
    TP_true = scalers["TP"].inverse(y_true_norm[:, :, 1])
    TP_pred = scalers["TP"].inverse(y_pred_norm[:, :, 1])

    print(f"\n==================== {title} | 水质回归指标（按步长） ====================\n")
    for i, h in enumerate(HORIZONS):
        rmse_tn, mae_tn, r2_tn = compute_reg_metrics(TN_true[:, i], TN_pred[:, i])
        rmse_tp, mae_tp, r2_tp = compute_reg_metrics(TP_true[:, i], TP_pred[:, i])
        print(f"[H={h:2d}h] TN: RMSE={rmse_tn:.4f} MAE={mae_tn:.4f} R2={r2_tn:.4f} | "
              f"TP: RMSE={rmse_tp:.4f} MAE={mae_tp:.4f} R2={r2_tp:.4f}")


def print_gate_metrics(title, gate_true, gate_prob, gate_th):
    print(f"\n==================== {title} | 闸门预测指标（按步长） ====================\n")
    for i, h in enumerate(HORIZONS):
        yt = gate_true[:, i].astype(int)
        yp = gate_prob[:, i]
        th = float(gate_th[i])

        try:
            pr_auc = float(average_precision_score(yt, yp))
        except Exception:
            pr_auc = np.nan

        yhat = (yp >= th).astype(int)
        f1 = float(f1_score(yt, yhat, zero_division=0))
        prec = float(precision_score(yt, yhat, zero_division=0))
        rec = float(recall_score(yt, yhat, zero_division=0))
        open_rate = float(yt.mean())

        print(f"[H={h:2d}h] PR-AUC={pr_auc:.4f} | F1={f1:.4f} P={prec:.4f} R={rec:.4f} | th={th:.2f} | open_rate={open_rate:.3f}")


# =========================
# 12) MAIN
# =========================
if __name__ == "__main__":
    set_seed(2025)

    base_path = "D:/海宁图神经网络水质预测/test"
    flow_file = os.path.join(base_path, "flow_data.csv")
    qual_file = os.path.join(base_path, "quality_data.csv")
    meteo_file = os.path.join(base_path, "meteorology.csv")
    landuse_file = os.path.join(base_path, "LandUse.csv")
    distance_file = os.path.join(base_path, "Distance.csv")

    out_dir = os.path.join(base_path, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    nodes_df, _ = create_river_network()
    num_nodes = len(nodes_df)

    flow_df = load_flow(flow_file, nodes_df)
    tn_df, tp_df = load_quality_TN_TP(qual_file, nodes_df)
    meteo_df = load_meteo(meteo_file, mode=METEO_MODE)
    landuse_tensor = load_landuse(landuse_file, nodes_df)
    _, decay_matrix = load_distance(distance_file, nodes_df)

    flow_df, tn_df, tp_df, meteo_df = align_by_time_step(flow_df, tn_df, tp_df, meteo_df)
    flow_df, tn_df, tp_df = estimate_virtual_nodes(flow_df, tn_df, tp_df, nodes_df, decay_matrix)

    train_list, test_list, scalers, meteo_dim = create_time_series_dataset(
        nodes_df,
        flow_df, tn_df, tp_df,
        meteo_df,
        landuse_tensor,
        decay_matrix,
        seq_len=SEQ_LEN,
        meteo_len=METEO_LEN,
        test_size=TEST_SIZE
    )

    print(f"[INFO] METEO_MODE={METEO_MODE} | meteo_dim={meteo_dim} | train={len(train_list)} | test={len(test_list)}")

    model, train_pack_k, test_pack_k, gate_th, node_to_idx = train_and_eval_pred_gate(
        nodes_df, train_list, test_list, scalers, num_nodes, meteo_dim, target_node=TARGET_NODE
    )

    train_yk, train_pk, train_endk, train_gt, train_gp = train_pack_k
    test_yk,  test_pk,  test_endk,  test_gt,  test_gp  = test_pack_k

    print_quality_metrics(f"{MODEL_NAME}/TRAIN(K)", train_yk, train_pk, scalers)
    print_gate_metrics(f"{MODEL_NAME}/TRAIN(K)", train_gt, train_gp, gate_th)

    print_quality_metrics(f"{MODEL_NAME}/TEST(K)", test_yk, test_pk, scalers)
    print_gate_metrics(f"{MODEL_NAME}/TEST(K)", test_gt, test_gp, gate_th)

    time_steps = flow_df["time_step"].values

    # 导出K点（含 gate）
    export_predictions_csv(
        out_csv=os.path.join(out_dir, f"{MODEL_NAME}_TRAIN_K.csv"),
        model_name=MODEL_NAME,
        split_name="TRAIN",
        node_name=TARGET_NODE,
        ys_norm=train_yk,
        ps_norm=train_pk,
        scalers=scalers,
        time_steps=time_steps,
        end_idxs=train_endk,
        gate_true=train_gt,
        gate_prob=train_gp,
        gate_th=gate_th
    )
    export_predictions_csv(
        out_csv=os.path.join(out_dir, f"{MODEL_NAME}_TEST_K.csv"),
        model_name=MODEL_NAME,
        split_name="TEST",
        node_name=TARGET_NODE,
        ys_norm=test_yk,
        ps_norm=test_pk,
        scalers=scalers,
        time_steps=time_steps,
        end_idxs=test_endk,
        gate_true=test_gt,
        gate_prob=test_gp,
        gate_th=gate_th
    )

    # 可选：导出全部实测点
    if EXPORT_ALL_ACTUAL_NODES:
        actual_nodes = nodes_df[nodes_df["node_type"] == "actual"]["node_id"].tolist()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        test_loader = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False)

        for n in actual_nodes:
            idx = node_to_idx[n]
            model.eval()
            ys_q, ps_q = [], []
            end_idxs_all = []
            gates_true, gates_prob = [], []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    y_hat_quality, gate_logits = model(batch, k_idx=node_to_idx[TARGET_NODE], meteo_len=METEO_LEN)
                    B = int(batch.num_graphs)
                    end_list = _as_int_list(getattr(batch, "end_idx", None), B)
                    end_idxs_all.extend(end_list)

                    yq = batch.y_quality.float().view(B, num_nodes, len(HORIZONS), 2)
                    predq = y_hat_quality.view(B, num_nodes, len(HORIZONS), 2)

                    ys_q.append(yq[:, idx, :, :].cpu().numpy())
                    ps_q.append(predq[:, idx, :, :].cpu().numpy())

                    gate_y = batch.gate_y.float().view(B, -1)
                    gates_true.append(gate_y.cpu().numpy())
                    gates_prob.append(torch.sigmoid(gate_logits).cpu().numpy())

            ys_q = np.concatenate(ys_q, axis=0)
            ps_q = np.concatenate(ps_q, axis=0)
            gates_true = np.concatenate(gates_true, axis=0)
            gates_prob = np.concatenate(gates_prob, axis=0)

            export_predictions_csv(
                out_csv=os.path.join(out_dir, f"{MODEL_NAME}_TEST_{n}.csv"),
                model_name=MODEL_NAME,
                split_name="TEST",
                node_name=n,
                ys_norm=ys_q,
                ps_norm=ps_q,
                scalers=scalers,
                time_steps=time_steps,
                end_idxs=end_idxs_all,
                gate_true=gates_true,
                gate_prob=gates_prob,
                gate_th=gate_th
            )
