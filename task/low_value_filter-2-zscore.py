"""
使用 z-score 场和局部外环对比作为前景概率, 进行 graph cut 切割高 z-score 区域
created at 2026-04-13
"""
from pathlib import Path
import meshio
import numpy as np
from collections import deque
import maxflow

from const import *
from utilize.mesh_io import read_mshv2_triangular

# demo_1, 长条状
# hard_ele_list = [5, 6, 31, 86, 94, 124, 125, 131, 136, 140, 145, 151, 177, 
#                  179, 185, 213, 219, 252, 255]
# demo_2, 圆形状
# hard_ele_list = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
# tissue_1, 圆形状 + 长条状
# hard_ele_list = [48, 55, 63, 91, 94, 126, 128, 138, 174, 201, 204, 247, 250, 254, 255, 
#                  256, 297, 327, 330, 379, 390, 391, 393, 396, 397, 401, 403, 412] + \
#                 [238, 278, 287, 288, 359, 365, 366, 367, 370, 374, 414]
# tissue_2, 长条状
# hard_ele_list = [66, 125, 138, 193, 197, 211, 219, 231, 234, 280, 284, 285, 290, 
#                  298, 301, 344, 345, 370, 376]

# real experiment, 圆形状
# hard_ele_list = [28, 30, 32, 35, 36, 37, 61, 64, 65, 79, 84, 139, 160, 458, 459, 686, 687]  # sponge_1 
# hard_ele_list = [51, 52, 53, 55, 61, 66, 67, 68, 123, 129, 132, 140, 143, 472, 473, 474, 475] # sponge_2
hard_ele_list = [51, 52, 54, 55, 62, 64, 66, 105, 153, 154, 155, 156] # sponge_2 (repeat)
# hard_ele_list = [7, 18, 32, 33, 55, 57, 67, 70, 71, 78, 90, 94] # silicone

### 从sim的数据中读取网格信息 ###
# mesh_path = MESH_DIR / "pd_stretch_tissue_mesh_init.msh"
# mesh_path = "data/results/sim-stiff-area-est/hard-3/pd_stretch_tissue_mesh_init.msh"
# V, F = read_mshv2_triangular(mesh_path)

mesh_data_file = np.load(DATA_DIR / "demo" / "real_track" / "05030806" / "tissue_mesh.npz")
neighbors = mesh_data_file['neighbors']
V, F = mesh_data_file['vertices'], mesh_data_file['faces']

mesh_data = {"V": V, "F": F}

# fixed_ele_list = np.loadtxt(OUTPUT_DIR / "background_ele_list.csv", delimiter=",", dtype=int).tolist()
# fixed_ele_list = np.loadtxt("data/results/sim-stiff-area-est/hard-3/background_ele_list.csv", delimiter=",", dtype=int).tolist()
fixed_ele_list = []
# fixed_node_list = list(range(40, 65))
# fixed_node_list = list(range(32, 44)) + list(range(59, 64))
fixed_node_list = list(range(30, 43)) + list(range(58, 63))     # sponge_2 (repeat)
# fixed_node_list = list(range(8, 11)) + list(range(30, 35))
for i, cell in enumerate(F):
    if any(v in fixed_node_list for v in cell):
        fixed_ele_list.append(i)
print(f"Fixed elements: {fixed_ele_list}")

zscore = np.loadtxt(OUTPUT_DIR / "zscore_mean.csv", delimiter=",")

assert zscore.shape[0] == F.shape[0], "Z-score vector length must match number of cells (triangles)."
n_cells = len(F)

edge_to_cells = {}
for i, cell in enumerate(F):
    for edge in [tuple(sorted((cell[0], cell[1]))), tuple(sorted((cell[1], cell[2]))), tuple(sorted((cell[2], cell[0])))]:
        edge_to_cells.setdefault(edge, []).append(i)

cell_neighbors = {i: [] for i in range(n_cells)}
for edge, cells in edge_to_cells.items():
    if len(cells) == 2:
        u, v = cells[0], cells[1]
        cell_neighbors[u].append(v)
        cell_neighbors[v].append(u)

SMOOTH_ITERATIONS = 1
ALPHA = 0.3
smoothed_zscore = zscore.copy()
for iteration in range(SMOOTH_ITERATIONS):
    temp_zscore = smoothed_zscore.copy()
    for i in range(n_cells):
        neighbors_idx = cell_neighbors[i]
        if len(neighbors_idx) > 0:
            neighbor_avg = np.mean(temp_zscore[neighbors_idx])
            smoothed_zscore[i] = (1.0 - ALPHA) * temp_zscore[i] + ALPHA * neighbor_avg


def build_face_adjacency(F):
    """构建三角面片的邻接表（共享边的面互为邻居）"""
    edge_to_faces = {}
    for fi, face in enumerate(F):
        v0, v1, v2 = face
        for edge in [(v0, v1), (v1, v2), (v2, v0)]:
            edge_sorted = tuple(sorted(edge))
            edge_to_faces.setdefault(edge_sorted, []).append(fi)
            
    adj_faces = [[] for _ in range(len(F))]
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            adj_faces[faces[0]].append(faces[1])
            adj_faces[faces[1]].append(faces[0])
    return adj_faces

def get_edge_length(V, F, fi, fj):
    """获取两个相邻面片共享边的长度"""
    set_fi = set(F[fi])
    set_fj = set(F[fj])
    shared_verts = list(set_fi.intersection(set_fj))
    if len(shared_verts) == 2:
        return np.linalg.norm(V[shared_verts[0]] - V[shared_verts[1]])
    return 0.0

def robust_sigmoid_score(x, center=None, scale=None, gain=2.5):
    if center is None:
        center = np.median(x)
    if scale is None:
        scale = np.percentile(x, 75) - np.percentile(x, 25)
    z = (x - center) / (scale + 1e-10)
    z = np.clip(gain * z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))

def get_ring_neighbors_range(fi, adj_faces, k_min, k_max):
    visited = set([fi])
    queue = deque([(fi, 0)])
    result = []
    while queue:
        curr, dist = queue.popleft()
        if k_min <= dist <= k_max:
            result.append(curr)
        if dist == k_max:
            continue
        for nb in adj_faces[curr]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    if fi in result:
        result.remove(fi)
    return result

def extract_high_value_region_v2(V, F, zscore, fixed_ele_list,
                                 outer_ring=(2, 4),
                                 lambda_smooth=1.0):
    num_faces = len(F)
    fixed_ele_set = set(fixed_ele_list)

    adj_faces = build_face_adjacency(F)

    # --------------------------------------------------
    # 1) 主区域证据：单元本身高 z-score
    # 高 z-score 区域 => 分数越大越像前景
    # --------------------------------------------------
    value_score_raw = zscore
    prob_value = robust_sigmoid_score(value_score_raw, gain=2.8)

    valid_mask = np.ones(num_faces, dtype=bool)
    valid_mask[list(fixed_ele_set)] = False

    # --------------------------------------------------
    # 2) 辅助区域证据：外环对比
    # 当前值 - 外环均值 越大，说明当前位置处在局部高值峰中
    # --------------------------------------------------
    local_score = np.zeros(num_faces)
    k_min, k_max = outer_ring

    global_mad = np.median(np.abs(zscore[valid_mask] - np.median(zscore[valid_mask])))
    mad_floor = 0.25 * global_mad

    for i in range(num_faces):
        outer = get_ring_neighbors_range(i, adj_faces, k_min, k_max)
        if len(outer) == 0:
            local_score[i] = 0.0
            continue

        outer_vals = zscore[outer]
        med_outer = np.median(outer_vals)
        mad_outer = np.median(np.abs(outer_vals - med_outer))
        denom = max(1.4826 * mad_outer, mad_floor, 1e-10)
        local_score[i] = (zscore[i] - med_outer) / denom

    # 以 0 为中心更合理：0 表示与外环持平；>0 表示高于外环
    prob_local = robust_sigmoid_score(local_score, center=0.0, scale=1.0, gain=1.4)

    # --------------------------------------------------
    # 3) 融合：不要乘法，用加权和
    # prob_value 主导，prob_local 辅助
    # --------------------------------------------------
    eps = 1e-6
    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    # fused_score = 1.4 * logit(prob_value) + 0.5 * logit(prob_local)
    # fused_score = .3 * logit(prob_value) + 1. * logit(prob_local)
    fused_score = 0.2 * logit(prob_value) + 1 * logit(prob_local) - 1.0
    fused_score = np.clip(fused_score, -30, 30)
    prob_fg = 1.0 / (1.0 + np.exp(-fused_score))

    # --------------------------------------------------
    # 4) 构造前景/背景种子
    # 前景：高 z-score 核心
    # 背景：低 z-score 区 + fixed background
    # --------------------------------------------------
    # fg_local_z = 1.0

    fg_seed = valid_mask & (prob_fg >= np.quantile(prob_fg[valid_mask], 0.90))
    # bg_seed = valid_mask & (prob_fg <= np.quantile(prob_fg[valid_mask], 0.10))
    bg_seed = valid_mask & \
            (zscore <= np.quantile(zscore[valid_mask], 0.25)) & \
            (local_score <= 0.0)
    bg_seed[list(fixed_ele_set)] = True

    # --------------------------------------------------
    # 5) Graph Cut
    # --------------------------------------------------
    g = maxflow.Graph[float]()
    node_ids = g.add_nodes(num_faces)

    # N-links
    zscore_diff_sq = []
    for i in range(num_faces):
        for j in adj_faces[i]:
            if i < j:
                zscore_diff_sq.append((zscore[i] - zscore[j]) ** 2)
    beta = 1.0 / (2.0 * np.mean(zscore_diff_sq) + 1e-10) if len(zscore_diff_sq) > 0 else 1.0

    edge_lengths = []
    for i in range(num_faces):
        for j in adj_faces[i]:
            if i < j:
                edge_lengths.append(get_edge_length(V, F, i, j))
    mean_edge_len = np.mean(edge_lengths) if len(edge_lengths) > 0 else 1.0

    for i in range(num_faces):
        for j in adj_faces[i]:
            if i < j:
                edge_len = get_edge_length(V, F, i, j)
                w = lambda_smooth * (edge_len / (mean_edge_len + 1e-10)) * \
                    np.exp(-beta * (zscore[i] - zscore[j]) ** 2)
                g.add_edge(node_ids[i], node_ids[j], w, w)

    # T-links
    cost_fg = -np.log(np.clip(prob_fg, eps, 1.0))
    cost_bg = -np.log(np.clip(1.0 - prob_fg, eps, 1.0))
    K = max(cost_fg.max(), cost_bg.max()) * 100.0

    # for i in range(num_faces):
    #     if i in fixed_ele_set or bg_seed[i]:
    #         g.add_tedge(node_ids[i], K, 0.0)   # 强背景（source, gc_labels==0）
    #     elif fg_seed[i]:
    #         g.add_tedge(node_ids[i], 0.0, K)   # 强前景（sink, gc_labels==1）
    #     else:
    #         g.add_tedge(node_ids[i], cost_fg[i], cost_bg[i])

    for i in range(num_faces):
        if i in fixed_ele_set:
            g.add_tedge(node_ids[i], K, 0.0)   # 强背景
        else:
            g.add_tedge(node_ids[i], cost_fg[i], cost_bg[i])

    g.maxflow()
    gc_labels = g.get_grid_segments(node_ids).astype(np.int32)

    # 强制 fixed 保持背景
    gc_labels[list(fixed_ele_set)] = 0

    # high_prob_bg = np.where((prob_fg >= 0.9) & (gc_labels == 0))[0]
    # print("high prob but background:", high_prob_bg)
    # print("prob_fg:", prob_fg[high_prob_bg])
    # print("fg_seed:", fg_seed[high_prob_bg])
    # print("bg_seed:", bg_seed[high_prob_bg])
    # print("zscore:", zscore[high_prob_bg])
    # print("local_score:", local_score[high_prob_bg])

    return local_score, gc_labels, prob_fg, prob_value, prob_local, fg_seed, bg_seed

local_score, gc_labels, prob_fg, prob_value, prob_local, fg_seed, bg_seed = extract_high_value_region_v2(
    V, F, smoothed_zscore, fixed_ele_list,
)

print(f"分割的目标单元数: {np.sum(gc_labels == 1)}/{len(gc_labels)}; 实际单元数 (hard label): {len(hard_ele_list)}")

np.savetxt(OUTPUT_DIR / "low_value_indices.csv", np.where(gc_labels == 1)[0], delimiter=",", fmt="%d")
print(f"High z-score region indices saved to compatibility path {OUTPUT_DIR / 'low_value_indices.csv'}")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                facecolors=smoothed_zscore, shading='flat',
                edgecolors='k', linewidth=0.6, cmap="RdBu_r")
plt.colorbar(label='Smoothed z-score field')
plt.title('Smoothed z-score field')
plt.axis('equal')

plt.subplot(122)
cmap_custom = plt.get_cmap('tab10', 2)
plt.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                facecolors=gc_labels.astype(int), shading='flat', 
                edgecolors='k', linewidth=0.6, cmap=cmap_custom, vmin=0, vmax=1)
plt.colorbar(ticks=[0, 1], label='Segment (0=Background, 1=High Z-score)')
# plt.gca().set_yticklabels(['Background', 'High Z-score'])
plt.axis('equal')
plt.title('High Z-score Region')
plt.show()
plt.savefig(f"{VISUALIZATION_DIR}/low_region_prob.svg", transparent=True, format='svg', dpi=300, bbox_inches='tight')
print(f"High z-score region visualization saved to compatibility path {VISUALIZATION_DIR}/low_region_prob.svg")

cells = [("triangle", F.astype(np.int32))]
mesh = meshio.Mesh(
V[:, :2],
cells,
cell_data={
    "SmoothedZScore": [smoothed_zscore],
    "GC_Label": [gc_labels.astype(int)],
    "Prob_FG": [prob_fg],
    "Prob_Value": [prob_value],
    "Prob_Local": [prob_local],
}
)
mesh.write(f"{OUTPUT_DIR}/mesh_low_value.vtu")


# exit()
# ==========================================
# 论文结果美化代码 (可选)
# ==========================================
from matplotlib.colors import ListedColormap
plt.style.use('default')
params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(params)

colors = ['#F0F0F2', '#1F77B4'] # 蓝色表示高 z-score 区域
cmap_custom = ListedColormap(colors)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
im2 = ax1.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                    facecolors=gc_labels.astype(int), shading='flat', 
                    edgecolors='k', linewidth=0.6, cmap=cmap_custom, vmin=0, vmax=1)
ax1.set_aspect('equal')
ax1.axis('off')
cbar1 = plt.colorbar(im2, ax=ax1, fraction=0.046, shrink=0.5, aspect=15)
cbar1.ax.tick_params(length=0)
cbar1.ax.set_yticklabels([])

plt.savefig(f"{ARTICLE_VIS_DIR}/low_value_region.svg", transparent=True, format='svg', dpi=300, bbox_inches='tight')
print(f"High z-score region visualization saved to compatibility path {ARTICLE_VIS_DIR}/low_value_region.svg")


# ==========================================
# 独立于分类算法的纯场量评估 (Field-Level Metrics)
# ==========================================
from sklearn.metrics import roc_auc_score, average_precision_score
print("\nStep: Evaluating Field Separability vs Ground Truth...")

def get_band_mask_from_gt(gt_mask, adj_faces, k_min=1, k_max=3):
    gt_idx = np.where(gt_mask)[0]
    band_mask = np.zeros(len(gt_mask), dtype=bool)

    for fi in gt_idx:
        outer = get_ring_neighbors_range(fi, adj_faces, k_min, k_max)
        band_mask[outer] = True

    band_mask[gt_mask] = False
    return band_mask

### Local ###
adj_faces = build_face_adjacency(F)

gt_mask = np.zeros(n_cells, dtype=bool)
gt_mask[hard_ele_list] = True

band_mask = get_band_mask_from_gt(gt_mask, adj_faces, 1, 3)
band_mask[fixed_ele_list] = False
print(f"GT region: {np.where(gt_mask==True)}")
print(f"Band region: {np.where(band_mask==True)}")

field_score = smoothed_zscore

local_eval_mask = gt_mask | band_mask
y_true_local = gt_mask[local_eval_mask]
y_score_local = field_score[local_eval_mask]

band_roc_auc = roc_auc_score(y_true_local, y_score_local)
band_pr_auc = average_precision_score(y_true_local, y_score_local)

target_vals = field_score[gt_mask]
band_vals = field_score[band_mask]

med_t = np.median(target_vals)
med_b = np.median(band_vals)
mad_t = np.median(np.abs(target_vals - med_t)) + 1e-10
mad_b = np.median(np.abs(band_vals - med_b)) + 1e-10
local_robust_cnr = np.abs(med_t - med_b) / (1.4826 * (mad_t + mad_b) + 1e-10)

mean_t, mean_b = np.mean(target_vals), np.mean(band_vals)
var_t, var_b = np.var(target_vals), np.var(band_vals)
local_fdr = (mean_t - mean_b) ** 2 / (var_t + var_b + 1e-10)

print(f"mean_t: {mean_t:.4f}, mean_b: {mean_b:.4f}, var_t: {var_t:.4f}, var_b: {var_b:.4f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.hist(target_vals, bins=50, alpha=0.5, label='GT')
plt.hist(band_vals, bins=50, alpha=0.5, label='Band')
plt.legend()
plt.savefig(f"field_value_distribution-0.svg", transparent=False, format='svg', dpi=300, bbox_inches='tight')

print(f"[Local Evaluation] (Using GT band as reference)")
print(f"  Band ROC-AUC: {band_roc_auc:.4f}")
print(f"  Band PR-AUC: {band_pr_auc:.4f}")
print(f"  Local Robust CNR: {local_robust_cnr:.4f}")
print(f"  Local FDR: {local_fdr:.4f}")

print(f"[Global Evaluation]")
# ---------------------------------------------------------
# 1. Ground Truth mask
# ---------------------------------------------------------
gt_mask = np.zeros(n_cells, dtype=bool)
gt_mask[hard_ele_list] = True

# ---------------------------------------------------------
# 2. Evaluation mask
# ---------------------------------------------------------
eval_mask = np.ones(n_cells, dtype=bool)
eval_mask[fixed_ele_list] = False

# ---------------------------------------------------------
# 3. Field score
# 目标是高 z-score 区域，所以 z-score 越大越像目标
# ---------------------------------------------------------
field_score = smoothed_zscore

y_true = gt_mask[eval_mask]
y_score = field_score[eval_mask]

# ---------------------------------------------------------
# 4. ROC-AUC / PR-AUC
# ---------------------------------------------------------
try:
    roc_auc = roc_auc_score(y_true, y_score)
    print(f"  [Field Metric 1] ROC-AUC: {roc_auc:.4f} (Closer to 1.0 is better)")
except ValueError:
    print("  [Field Metric 1] ROC-AUC: N/A")

try:
    pr_auc = average_precision_score(y_true, y_score)
    print(f"  [Field Metric 2] PR-AUC: {pr_auc:.4f} (Higher is better)")
except ValueError:
    print("  [Field Metric 2] PR-AUC: N/A")

# ---------------------------------------------------------
# 5. Robust CNR / FDR
# ---------------------------------------------------------
target_vals = field_score[gt_mask & eval_mask]
bg_vals = field_score[(~gt_mask) & eval_mask]

if len(target_vals) > 0 and len(bg_vals) > 0:
    # Robust CNR
    med_t = np.median(target_vals)
    med_b = np.median(bg_vals)

    mad_t = np.median(np.abs(target_vals - med_t)) + 1e-10
    mad_b = np.median(np.abs(bg_vals - med_b)) + 1e-10

    robust_cnr = np.abs(med_t - med_b) / (1.4826 * (mad_t + mad_b) + 1e-10)
    print(f"  [Field Metric 3] Robust CNR: {robust_cnr:.4f} (Higher is better)")

    # FDR
    mean_t = np.mean(target_vals)
    mean_b = np.mean(bg_vals)
    var_t = np.var(target_vals)
    var_b = np.var(bg_vals)

    fdr = (mean_t - mean_b) ** 2 / (var_t + var_b + 1e-10)
    print(f"  [Field Metric 4] FDR: {fdr:.4f} (Higher is better)")
else:
    print("  [Field Metric 3] Robust CNR: N/A")
    print("  [Field Metric 4] FDR: N/A")

# print("\nStep: Evaluating Segmentation Quality vs Ground Truth...")

# ---------------------------------------------------------
# 1. Ground Truth mask
# ---------------------------------------------------------
gt_mask = np.zeros(n_cells, dtype=bool)
gt_mask[hard_ele_list] = True

# ---------------------------------------------------------
# 2. Evaluation mask
# 排除 fixed 区域，避免人为硬约束影响评估
# ---------------------------------------------------------
eval_mask = np.ones(n_cells, dtype=bool)
eval_mask[fixed_ele_list] = False

# ---------------------------------------------------------
# 3. Prediction mask
# 这里沿用当前代码的约定：gc_labels == 1 表示 high z-score region
# ---------------------------------------------------------
pred_mask = (gc_labels == 1)

# 只在有效区域内评估
gt_eval = gt_mask[eval_mask]
pred_eval = pred_mask[eval_mask]

# ---------------------------------------------------------
# 4. Confusion terms
# ---------------------------------------------------------
tp = np.sum(pred_eval & gt_eval)
fp = np.sum(pred_eval & (~gt_eval))
fn = np.sum((~pred_eval) & gt_eval)
tn = np.sum((~pred_eval) & (~gt_eval))

eps = 1e-10

# ---------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------
dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
iou = tp / (tp + fp + fn + eps)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
specificity = tn / (tn + fp + eps)
f1 = 2.0 * precision * recall / (precision + recall + eps)

# ---------------------------------------------------------
# 6. Print
# ---------------------------------------------------------
print(f"[Global Segmentation Evaluation]")
print(f"  [Seg Metric 1] Dice:       {dice:.4f}")
print(f"  [Seg Metric 2] IoU:        {iou:.4f}")
print(f"  [Seg Metric 3] Precision:  {precision:.4f}")
print(f"  [Seg Metric 4] Recall:     {recall:.4f}")
print(f"  [Seg Metric 5] Specificity:{specificity:.4f}")
print(f"  [Seg Metric 6] F1-score:   {f1:.4f}")
