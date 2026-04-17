"""
在信息空间取对数和面积归一化后
使用梯度和最低值的概率乘法作为前景概率，进行graph cut切割
created at 2026-04-13
"""
from pathlib import Path
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
# tissue_2， 长条状
# hard_ele_list = [66, 125, 138, 193, 197, 211, 219, 231, 234, 280, 284, 285, 290, 
#                  298, 301, 344, 345, 370, 376]

# hard_ele_list = [28, 30, 32, 35, 36, 37, 61, 64, 65, 79, 84, 139, 160, 458, 459, 686, 687]  # sponge_1 
# hard_ele_list = [44, 51, 52, 53, 55, 61, 66, 67, 68, 69, 123, 132, 139, 486, 487, 488, 489]  # sponge_2
hard_ele_list = [51, 52, 53, 55, 61, 66, 67, 68, 123, 129, 132, 140, 143, 486, 487, 488, 489]

### 从sim的数据中读取网格信息 ###
# mesh_path = MESH_DIR / "pd_stretch_tissue_mesh_init_2.msh"
# mesh_path = "data/results/sim-stiff-area-est/hard-3/pd_stretch_tissue_mesh_init.msh"
# V, F = read_mshv2_triangular(mesh_path)

mesh_data_file = np.load(DATA_DIR / "demo" / "real_track" / "04170230" / "tissue_mesh.npz")
neighbors = mesh_data_file['neighbors']
V, F = mesh_data_file['vertices'], mesh_data_file['faces']
mesh_data = {"V": V, "F": F}

# cell_size = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)

# fixed_ele_list = np.loadtxt(OUTPUT_DIR / "background_ele_list.csv", delimiter=",", dtype=int).tolist()
# fixed_ele_list = np.loadtxt("data/results/sim-stiff-area-est/hard-3/background_ele_list.csv", delimiter=",", dtype=int).tolist()
fixed_ele_list = []
# fixed_node_list = list(range(40, 65))
# fixed_node_list = list(range(31, 43)) + list(range(59, 64))
fixed_node_list = list(range(32, 44)) + list(range(59, 66))
for i, cell in enumerate(F):
    if any(v in fixed_node_list for v in cell):
        fixed_ele_list.append(i)

# variance_path = OUTPUT_DIR / "Y_ekf.csv"
# variance_path = "data/results/sim-stiff-area-est/hard-3/Y_ekf.csv"
variance_path = Path(OUTPUT_DIR) / "Y_sum_ekf.csv"

variance_mat = np.loadtxt(variance_path, delimiter=",")
variance = np.log10(np.sqrt(np.diag(variance_mat)))
# variance_normalized = variance / cell_size * 1.e-4 # 归一化方差，考虑单元大小对不确定性的影响
# np.savetxt(OUTPUT_DIR / "variance_log.csv", variance, delimiter=",")

assert variance.shape[0] == F.shape[0], "Variance vector length must match number of cells (triangles)."
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
smoothed_variance = variance.copy()
for iteration in range(SMOOTH_ITERATIONS):
    temp_variance = smoothed_variance.copy()
    for i in range(n_cells):
        neighbors_idx = cell_neighbors[i]
        if len(neighbors_idx) > 0:
            neighbor_avg = np.mean(temp_variance[neighbors_idx])
            smoothed_variance[i] = (1.0 - ALPHA) * temp_variance[i] + ALPHA * neighbor_avg


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

def get_k_ring_neighbors(fi, adj_faces, k):
    """获取拓扑距离为 k 环内的所有面片索引（不包含中心面片 fi)"""
    visited = set([fi])
    queue = deque([(fi, 0)])
    neighbors = []
    while queue:
        curr, dist = queue.popleft()
        if dist > 0:
            neighbors.append(curr)
        if dist == k:
            continue
        for nb in adj_faces[curr]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return neighbors

def get_edge_length(V, F, fi, fj):
    """获取两个相邻面片共享边的长度"""
    set_fi = set(F[fi])
    set_fj = set(F[fj])
    shared_verts = list(set_fi.intersection(set_fj))
    if len(shared_verts) == 2:
        return np.linalg.norm(V[shared_verts[0]] - V[shared_verts[1]])
    return 0.0

def extract_low_value_pure_continuous(V, F, variance, fixed_ele_list, k_ring=2, lambda_smooth=5.0):
    """
    :param sensitivity: 控制对低值区域的敏感程度。越大，越容易把稍低于周围的区域划入前景
    """
    num_faces = len(F)
    fixed_ele_set = set(fixed_ele_list)
    
    print("Step 1: 计算局部对比度 S(i)...")
    adj_faces = build_face_adjacency(F)
    S = np.zeros(num_faces)
    for i in range(num_faces):
        neighbors = get_k_ring_neighbors(i, adj_faces, k_ring)
        if len(neighbors) > 0:
            # weights = cell_size[neighbors]
            # weighted_mean = np.sum(variance[neighbors] * weights) / np.sum(weights)
            neighbor_mean = np.mean(variance[neighbors])
            S[i] = variance[i] - neighbor_mean
            
    print("Step 2: 使用 Sigmoid 函数将对比度转化为概率（无需阈值）...")

    IQR_grad = np.percentile(S, 75) - np.percentile(S, 25)
    k_grad = 5.0 / (IQR_grad + 1e-10)

    IQR_value = np.percentile(variance, 75) - np.percentile(variance, 25)
    k_value = 2.0 / (IQR_value + 1e-10)

    prob_grad = 1.0 / (1.0 + np.exp(k_grad * (S - np.median(S))))
    prob_value = 1.0 / (1.0 + np.exp(k_value * (variance - np.median(variance))))

    prob_fg = prob_value * prob_grad  # 直接乘法融合两种信息，得到前景概率
    
    print("Step 3: 构建 Maxflow...")
    g = maxflow.Graph[float]()
    node_ids = g.add_nodes(num_faces)
    
    # --- N-links (平滑项) ---
    var_diff_sq = []
    for i in range(num_faces):
        for j in adj_faces[i]:
            if i < j:
                var_diff_sq.append((variance[i] - variance[j])**2)
    beta = 1.0 / (2.0 * np.mean(var_diff_sq) + 1e-10) if var_diff_sq else 1.0

    mean_edge_len = np.mean([get_edge_length(V, F, i, j) for i in range(num_faces) for j in adj_faces[i] if i < j])
        
    for i in range(num_faces):
        for j in adj_faces[i]:
            if i < j:
                edge_len = get_edge_length(V, F, i, j)
                # w = lambda_smooth * edge_len * np.exp(-beta * (variance[i] - variance[j])**2)
                w = (1 * edge_len / mean_edge_len) * np.exp(-beta * (variance[i] - variance[j])**2) * lambda_smooth
                # print(f"Adding edge between faces {i} and {j} with weight {w}")
                g.add_edge(node_ids[i], node_ids[j], w, w)
                
    # --- T-links ---
    eps = 1e-3
    cost_fg = -np.log(prob_fg + eps)      # 分配给前景的代价
    cost_bg = -np.log(1.0 - prob_fg + eps) # 分配给背景的代价
    
    K = max(cost_fg.max(), cost_bg.max()) * 100 
    
    for i in range(num_faces):
        if i in fixed_ele_set:
            # 只有固定的背景单元使用硬约束
            g.add_tedge(node_ids[i], 0, K)
        else:
            g.add_tedge(node_ids[i], cost_fg[i], cost_bg[i])
            
    print("Step 4: 运行最大流/最小割...")
    g.maxflow()
    gc_labels = g.get_grid_segments(node_ids).astype(np.int32)
    gc_labels[list(fixed_ele_set)] = 0
    
    print(f"完成！检测到低值区域单元数: {np.sum(gc_labels == 1)}")
    return S, gc_labels, prob_fg, prob_value, prob_grad


S, gc_labels, prob_fg, prob_value, prob_grad = extract_low_value_pure_continuous(V, F, smoothed_variance, fixed_ele_list, k_ring=2, lambda_smooth=1.)

np.savetxt(OUTPUT_DIR / "low_value_indices.csv", np.where(gc_labels == 1)[0], delimiter=",", fmt="%d")
print(f"Low-value region indices saved to {OUTPUT_DIR / 'low_value_indices.csv'}")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                facecolors=prob_fg, shading='flat', 
                edgecolors='k', linewidth=0.6, cmap="RdBu_r")
plt.colorbar(label='Local Contrast S(i)')
plt.axis('equal')
plt.title('Local Contrast S(i) Histogram')

plt.subplot(122)
cmap_custom = plt.get_cmap('tab10', 2)
plt.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                facecolors=gc_labels.astype(int), shading='flat', 
                edgecolors='k', linewidth=0.6, cmap=cmap_custom, vmin=0, vmax=1)
plt.colorbar(ticks=[0, 1], label='Segment (0=Background, 1=Low Value)')
# plt.gca().set_yticklabels(['Background', 'Low Value'])
plt.axis('equal')
plt.title('Variance Histogram')
plt.show()
plt.savefig(f"{VISUALIZATION_DIR}/low_region_prob.svg", transparent=True, format='svg', dpi=300, bbox_inches='tight')
print(f"Low-value region visualization saved to {VISUALIZATION_DIR}/low_region_prob.svg")


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

colors = ['#F0F0F2', '#1F77B4'] # 蓝色表示低值区域
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
print(f"Low-value region visualization saved to {ARTICLE_VIS_DIR}/low_value_region.svg")


# ==========================================
# 独立于分类算法的纯场量评估 (Field-Level Metrics)
# ==========================================
from sklearn.metrics import roc_auc_score, average_precision_score
print("\nStep: Evaluating Raw Uncertainty Field vs Ground Truth...")

# 需要根据数据本身的空间调整表述，比如Log空间还是Linear空间
value = prob_value

# 1. 准备真实的 Ground Truth Mask (如果之前没定义的话)
gt_mask = np.zeros(n_cells, dtype=bool)
# hard_ele_list 是真实目标单元的索引列表
gt_mask[hard_ele_list] = True

# bg_mask = np.ones(n_cells, dtype=bool)
# bg_mask[hard_ele_list] = False
# bg_mask = (~gt_mask)

# 排除 fixed 节点带来的强偏置干扰（它们在物理上不参与变形评估）
eval_mask = np.ones(n_cells, dtype=bool)
eval_mask[fixed_ele_list] = False

# 提取用于评估的真值和标量场
y_true = gt_mask[eval_mask]
# 这里使用对数方差或线性方差都可以，因为 AUC 只看排序。
# 但对于 CNR，我们使用之前确定的具备尺度不变性的对数方差 (variance)
y_score = value[eval_mask] 

# ---------------------------------------------------------
# 指标 1: ROC-AUC (无需阈值的排序绝对准度)
# ---------------------------------------------------------
try:
    auc_score = roc_auc_score(y_true, y_score)
    print(f"  [Field Metric 1] ROC-AUC: {auc_score:.4f} (Closer to 1.0 is better)")
except ValueError:
    print("  [Field Metric 1] ROC-AUC: N/A (Only one class present in y_true)")

# ---------------------------------------------------------
# 指标 2: Average Precision (应对小目标的极度严苛指标)
# ---------------------------------------------------------
try:
    ap_score = average_precision_score(y_true, y_score)
    print(f"  [Field Metric 2] Average Precision (PR-AUC): {ap_score:.4f} (Higher is better)")
except ValueError:
    print("  [Field Metric 2] Average Precision: N/A")

# ---------------------------------------------------------
# 指标 3: GT-based CNR (物理场信噪比)
# ---------------------------------------------------------
gt_target_values = value[gt_mask & eval_mask]
gt_bg_values = value[(~gt_mask) & eval_mask]

if len(gt_target_values) > 0 and len(gt_bg_values) > 0:
    mean_gt_target = np.mean(gt_target_values)
    mean_gt_bg = np.mean(gt_bg_values)
    std_gt_bg = np.std(gt_bg_values)
    std_gt_target = np.std(gt_target_values)
    
    gt_cnr = np.abs(mean_gt_target - mean_gt_bg) / (std_gt_bg + std_gt_target + 1e-10)
    print(f"  [Field Metric 3] GT-based CNR: {gt_cnr:.4f} (Higher means better signal contrast)")
else:
    print("  [Field Metric 3] GT-based CNR: N/A (Missing target or background regions)")

# 提取线性尺度（物理标准差）和对数尺度的数据
# 记得 eval_mask 已经排除了 fixed_ele_list 的干扰
linear_std = value

gt_linear_target = linear_std[gt_mask & eval_mask]
gt_linear_bg = linear_std[(~gt_mask) & eval_mask]

gt_log_target = value[gt_mask & eval_mask]
gt_log_bg = value[(~gt_mask) & eval_mask]

if len(gt_log_target) > 0 and len(gt_log_bg) > 0:
    # ---------------------------------------------------------
    # 指标 4: GT-based FDR (费舍尔判别比)
    # 使用对数空间计算，保证尺度不变性
    # ---------------------------------------------------------
    # mean_gt_log_target = np.mean(gt_log_target)
    # mean_gt_log_bg = np.mean(gt_log_bg)
    # var_gt_log_target = np.var(gt_log_target)
    # var_gt_log_bg = np.var(gt_log_bg)
    
    # fdr_gt = ((mean_gt_log_target - mean_gt_log_bg)**2) / (var_gt_log_target + var_gt_log_bg + 1e-10)
    # print(f"  [Field Metric 4] (Log) GT-based FDR: {fdr_gt:.4f} (Higher means better statistical separability)")

    mean_t = np.mean(gt_linear_target)
    mean_b = np.mean(gt_linear_bg)
    var_t = np.var(gt_linear_target)
    var_b = np.var(gt_linear_bg)

    fdr_gt = ((mean_t - mean_b) ** 2) / (var_t + var_b + 1e-10)
    print(f"  [Field Metric 4] (Linear) GT-based FDR: {fdr_gt:.4f} (Higher means better statistical separability)")

    # ---------------------------------------------------------
    # 指标 5: GT-based Background Suppression Ratio (背景抑制比)
    # 使用线性空间计算，反映真实的物理数值比例变化
    # ---------------------------------------------------------
    mean_gt_linear_target = np.mean(gt_linear_target)
    mean_gt_linear_bg = np.mean(gt_linear_bg)
    
    linear_ratio = mean_gt_linear_target / (mean_gt_linear_bg + 1e-10)
    linear_distance = np.mean(gt_linear_target) - np.mean(gt_linear_bg)
    # geometric_ratio = np.exp(np.mean(gt_log_target) - np.mean(gt_log_bg))
    print(f"  [Field Metric 5] GT-based Background Suppression Distance: {linear_distance:.4f} (Higher means target is relatively enhanced)")
    print(f"  [Field Metric 6] GT-based Background Suppression Ratio: {linear_ratio:.4f} (Higher means target is relatively enhanced)")

else:
    print("  [Field Metrics] N/A (Missing target or background regions in GT)")
