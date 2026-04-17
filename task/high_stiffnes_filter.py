"""
在取对数后的标准差空间中, 使用GMM对单元进行三分类, 并结合Graph Cut方法提取高不确定性区域.

Date: 2026-01-23
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import maxflow
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pathlib import Path

from utilize.mesh_io import read_mshv2_triangular
from const import MESH_DIR, OUTPUT_DIR, VISUALIZATION_DIR, ARTICLE_VIS_DIR


# tissue_2， 长条状
hard_ele_list = [66, 125, 138, 193, 197, 211, 219, 231, 234, 280, 284, 285, 290, 
                 298, 301, 344, 345, 370, 376]
# demo_1, 长条状
# hard_ele_list = [5, 6, 31, 86, 94, 124, 125, 131, 136, 140, 145, 151, 177, 
#                  179, 185, 213, 219, 252, 255]
# demo_2, 圆形状
# hard_ele_list = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
# tissue_1, 圆形状 + 长条状
# hard_ele_list = [48, 55, 63, 91, 94, 126, 128, 138, 174, 201, 204, 247, 250, 254, 255, 
#                  256, 297, 327, 330, 379, 390, 391, 393, 396, 397, 401, 403, 412] + \
#                 [238, 278, 287, 288, 359, 365, 366, 367, 370, 374, 414]

# 要把fixed相关的单元去除计算,并且直接分类为背景

# neighbors_path = MESH_DIR / "pd_stretch_demo_mesh_neighbors.csv"
# neighbors = np.loadtxt(neighbors_path, delimiter=",")

fixed_ele_list = np.loadtxt(OUTPUT_DIR / "background_ele_list.csv", delimiter=",", dtype=int).tolist()
# fixed_ele_list = []

variance_path = OUTPUT_DIR / "P_diag_ekf.csv"
variance_vec = np.loadtxt(variance_path, delimiter=",")
variance = np.log10(np.sqrt(variance_vec))
# variance = np.sqrt(variance_vec)

# variance = np.loadtxt(f"{OUTPUT_DIR}/estimated_stiffness_ekf.csv", delimiter=",")
# variance = np.log10(variance)

# variance = np.loadtxt(f"stretch_weight_update.csv", delimiter=",")

mesh_path = MESH_DIR / "pd_stretch_tissue_mesh_init_2.msh"
# mesh_path = OUTPUT_DIR / "20260225_220420" / "pd_contact[39, 77]_step010.msh"

V, F = read_mshv2_triangular(mesh_path)
mesh_data = {"V": V, "F": F}

assert variance.shape[0] == F.shape[0], "Variance vector length must match number of cells (triangles)."

n_cells = len(F)
edge_to_cells = {}
for i, cell in enumerate(F):
    for edge in [tuple(sorted((cell[0], cell[1]))), tuple(sorted((cell[1], cell[2]))), tuple(sorted((cell[2], cell[0])))]:
        edge_to_cells.setdefault(edge, []).append(i)

# ==========================================
# 1.5. 网格标量场平滑 (Topological Smoothing)
# ==========================================
print("Step 0.5: Smoothing scalar field...")

# a. 构建单元到单元的邻接表 (Cell-to-Cell Adjacency)
cell_neighbors = {i: [] for i in range(n_cells)}
for edge, cells in edge_to_cells.items():
    if len(cells) == 2:
        u, v = cells[0], cells[1]
        cell_neighbors[u].append(v)
        cell_neighbors[v].append(u)

# b. 设置平滑参数
SMOOTH_ITERATIONS = 1   # 迭代次数 (建议 1~3 次)
ALPHA = 0.4             # 平滑强度 (0: 不平滑, 1: 完全等于邻居均值)

smoothed_variance = variance.copy()

for iteration in range(SMOOTH_ITERATIONS):
    # 使用 temp 数组，确保每一轮更新都是基于上一轮的同步状态
    temp_variance = smoothed_variance.copy() 
    
    for i in range(n_cells):
        neighbors_idx = cell_neighbors[i]
        if len(neighbors_idx) > 0:
            # 计算相邻单元的均值
            neighbor_avg = np.mean(temp_variance[neighbors_idx])
            # 公式：(1 - alpha) * 当前值 + alpha * 邻居均值
            smoothed_variance[i] = (1.0 - ALPHA) * temp_variance[i] + ALPHA * neighbor_avg

# 将平滑后的结果覆盖回原变量，供后续 GMM 和 Graph Cut 使用
variance = smoothed_variance

print(f"  - Applied Laplacian smoothing with alpha={ALPHA}, iterations={SMOOTH_ITERATIONS}")

# ==========================================
# 2. GMM 概率建模 (Data Term) —— 排除 fixed_cells
# ==========================================
print("Step 1: Running GMM...")

# GMM 输入需要是 (n_samples, n_features)
X = variance.reshape(-1, 1)
X_mask = np.delete(X, fixed_ele_list, axis=0)     # 去除与 fixed 相关的单元

# 拟合两个高斯分布：背景 vs 高不确定性区域
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# 获取两个分布的均值，判断哪个是“高不确定性”类
means = gmm.means_.flatten()
high_uncertainty_idx = np.argmax(means) # 均值大的那个是目标索引
low_uncertainty_idx = np.argmin(means)
background_idx = 3 - high_uncertainty_idx - low_uncertainty_idx # 剩下的一个索引

print(f"  - Low Uncert Mean: {means[low_uncertainty_idx]:.4f}")
print(f"  - Background Mean: {means[background_idx]:.4f}")
print(f"  - High Uncert Mean: {means[high_uncertainty_idx]:.4f}")

# 计算属于每一类的后验概率
probs = gmm.predict_proba(X)
prob_high = probs[:, high_uncertainty_idx] # 属于高不确定性的概率
prob_bg = probs[:, background_idx]         # 属于背景的概率
prob_low = probs[:, low_uncertainty_idx]  # 属于低不确定性的概率

prob_sink = 1.0 - prob_high

# 防止 log(0) 错误，加一个极小值
epsilon = 1e-2
prob_high = np.clip(prob_high, epsilon, 1.0 - epsilon)
prob_bg = np.clip(prob_sink, epsilon, 1.0 - epsilon)

# 计算 t-link (Terminal Link) 的能量项 (负对数似然)
# Graph Cut 最小化能量，所以概率越高，能量(代价)应该越低
# Source (S) = High Uncertainty, Sink (T) = Background
# 如果割断 S->i 的边，意味着 i 变成了 Background (T)。代价应该是 "i 属于 High 的概率" 对应的某种惩罚？
# 修正逻辑：
# Capacity(S->i): 保持连接S的意愿。如果 P(High) 很大，我们希望保留 S->i，切断 i->T。
# 所以 Cap(S->i) 应该大，Cap(i->T) 应该小。
# 公式：Cap = -ln(1 - P) 或者直接用 -ln(P_other)
# Cap(S->i) = -ln(P(Background))  <-- 如果 P(BG) 很小，代价巨大，很难切断
# Cap(i->T) = -ln(P(High))        <-- 如果 P(High) 很大，代价很小，容易切断（归为S）
data_term_weight_to_source = -np.log(prob_sink)  
data_term_weight_to_sink = -np.log(prob_high) 

# ==========================================
# 3. Graph Cut 构建与求解 (Smoothness Term)
# ==========================================
print("Step 2: Building Graph...")

# Lambda: 平滑系数。调节它来控制“分割块”的连续程度。
# 越大，区域越整块（可能丢失细节）；越小，越破碎（接近单纯的阈值法）。
# 经验值通常在 1.0 到 10.0 之间，需要根据 variance 的数值范围微调。
LAMBDA = 8.0
# SIGMA = 0.5 # 用于高斯核计算相邻差异的权重

diff_sq_sum = 0.0
num_edges = 0

for edge, cells in edge_to_cells.items():
    if len(cells) == 2:
        u, v = cells[0], cells[1]
        diff = variance[u] - variance[v]
        diff_sq_sum += diff ** 2
        num_edges += 1

# 计算 beta = 1 / (2 * sigma^2) = 1 / <(diff)^2>
if num_edges > 0:
    beta = 1.0 / (diff_sq_sum / num_edges + 1e-10)
    # beta = 1. / (2 * SIGMA**2) # 直接使用预设的 SIGMA 来计算 beta
else:
    beta = 0 # Should not happen

print(f"Auto-calculated Beta (Sensitivity): {beta:.4f}")

# 初始化 PyMaxflow 图
# estimate number of edges (每条边最多共享2个三角形)
est_n_edges = len(edge_to_cells)
g = maxflow.Graph[float](n_cells, est_n_edges)

# 添加节点
nodes = g.add_nodes(n_cells)

# 添加 n-links (Neighbor Links) —— 平滑项
# 遍历所有的边，找到相邻的两个三角形
for edge, cells in edge_to_cells.items():
    if len(cells) == 2:
        u, v = cells[0], cells[1]
        
        # 计算相邻单元的不确定性差异
        diff = variance[u] - variance[v]
        
        # 定义平滑权重：差异越小，连接越强（越难切开）；差异越大，连接越弱（是边界）
        # 使用经典的高斯核函数
        weight = LAMBDA * np.exp(-beta * (diff**2))
        
        # 添加双向边 (无向图)
        g.add_edge(nodes[u], nodes[v], weight, weight)

# 添加 t-links (Terminal Links) —— 数据项
for i in range(n_cells):
    # 设置节点分类到 Source (High Uncertainty) 和 Sink (Background) 的权重
    if i in fixed_ele_list:
        # 直接将与 fixed 相关的单元分类为背景 (Sink)
        g.add_tedge(nodes[i], 0, 1e10) # Source 权重为0，Sink 权重非常大，强制分类为背景
    else:
        g.add_tedge(nodes[i], data_term_weight_to_source[i], data_term_weight_to_sink[i])

print("Step 3: Calculating Max Flow / Min Cut...")
flow = g.maxflow()

print(f"  - Max Flow Value: {flow}")

# ==========================================
# 4. 提取结果
# ==========================================
# g.get_segment(node) 返回 0 (Source/High) 或 1 (Sink/Background)
# 注意：PyMaxflow 的 get_segment 定义通常是：
# 0 means the node is in the SOURCE set (我们在上面定义 Source 为 High Uncertainty)
# 1 means the node is in the SINK set (Background)

segment_results = []
for i in range(n_cells):
    segment_results.append(g.get_segment(nodes[i]))

segment_results = np.array(segment_results)

# 转换成 boolean mask: True 表示 High Uncertainty Region
# PyMaxflow: segment=0 -> Source, segment=1 -> Sink
high_uncertainty_mask = (segment_results == 0)

num_high = np.sum(high_uncertainty_mask)
print(f"Step 4: Done. Found {num_high} cells in high uncertainty region.")

# ==========================================
# 5. 可视化/保存 (可选)
# ==========================================
# 如果你想保存出哪些 ID 是高不确定性的
high_uncertainty_indices = np.where(high_uncertainty_mask)[0]
np.savetxt(OUTPUT_DIR / "high_uncertainty_indices.csv", high_uncertainty_indices, fmt="%d")

fig, ax1 = plt.subplots(figsize=(6, 6))
cmap_custom = plt.get_cmap('tab10', 2)
im3 = ax1.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                    facecolors=high_uncertainty_mask.astype(int), cmap=cmap_custom, edgecolors='k')
cbar3 = fig.colorbar(im3, ax=ax1, ticks=[0, 1])
cbar3.ax.set_yticklabels(['Background', 'High Variance'])
ax1.set_title("High Variance Region")
ax1.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{VISUALIZATION_DIR}/high_uncertainty_region.svg", dpi=300)
print(f"Visualization saved to {VISUALIZATION_DIR}/high_uncertainty_region.svg")
# plt.show()

true_pos = np.intersect1d(high_uncertainty_indices, hard_ele_list)
precision = len(true_pos) / num_high if num_high > 0 else 0
recall = len(true_pos) / len(hard_ele_list) if len(hard_ele_list) > 0 else 0
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

# ==========================================
# 论文结果美化代码
# ==========================================

from matplotlib.colors import ListedColormap

plt.style.use('default') # 重置
params = {
    'font.family': 'serif',        # 衬线体 (Times New Roman 等)
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], # 显式指定
    'mathtext.fontset': 'stix',    # 数学公式字体更像 LaTeX
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,      # 去掉顶部边框 (更简洁)
    'axes.spines.right': False,    # 去掉右侧边框
}
plt.rcParams.update(params)

colors = ['#F0F0F2', '#B81502'] # E3F2FD 1F77B4 B81502
cmap_custom = ListedColormap(colors)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
im2 = ax1.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                    facecolors=high_uncertainty_mask.astype(int), shading='flat', 
                    edgecolors='k', linewidth=0.6, cmap=cmap_custom, vmin=0, vmax=1)
# ax1.set_title('(b) Graph Cut Segmentation (Output)', pad=15)
ax1.set_aspect('equal')
ax1.axis('off')

# 自定义 Colorbar 的刻度
cbar1 = plt.colorbar(im2, ax=ax1, fraction=0.046, shrink=0.5, aspect=15)
cbar1.ax.tick_params(length=0)
cbar1.ax.set_yticklabels([])

plt.savefig(f"{ARTICLE_VIS_DIR}/high_uncertainty_region.svg", transparent=True, format='svg', dpi=300, bbox_inches='tight')


# ==========================================
# 独立于分类算法的纯场量评估 (Field-Level Metrics)
# ==========================================
print("\nStep: Evaluating Raw Uncertainty Field vs Ground Truth...")

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
y_score = variance[eval_mask] 

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
# 指标 3: GT-based Log-CNR (物理场信噪比)
# ---------------------------------------------------------
gt_target_values = variance[gt_mask & eval_mask]
gt_bg_values = variance[(~gt_mask) & eval_mask]

if len(gt_target_values) > 0 and len(gt_bg_values) > 0:
    mean_gt_target = np.mean(gt_target_values)
    mean_gt_bg = np.mean(gt_bg_values)
    std_gt_bg = np.std(gt_bg_values)
    
    gt_log_cnr = np.abs(mean_gt_target - mean_gt_bg) / (std_gt_bg + 1e-10)
    print(f"  [Field Metric 3] GT-based Log-CNR: {gt_log_cnr:.4f} (Higher means better signal contrast)")
else:
    print("  [Field Metric 3] GT-based Log-CNR: N/A (Missing target or background regions)")

# 提取线性尺度（物理标准差）和对数尺度的数据
# 记得 eval_mask 已经排除了 fixed_ele_list 的干扰
linear_std = np.sqrt(variance_vec) 

gt_linear_target = linear_std[gt_mask & eval_mask]
gt_linear_bg = linear_std[(~gt_mask) & eval_mask]

gt_log_target = variance[gt_mask & eval_mask]
gt_log_bg = variance[(~gt_mask) & eval_mask]

if len(gt_log_target) > 0 and len(gt_log_bg) > 0:
    # ---------------------------------------------------------
    # 指标 4: GT-based FDR (费舍尔判别比)
    # 使用对数空间计算，保证尺度不变性
    # ---------------------------------------------------------
    mean_gt_log_target = np.mean(gt_log_target)
    mean_gt_log_bg = np.mean(gt_log_bg)
    var_gt_log_target = np.var(gt_log_target)
    var_gt_log_bg = np.var(gt_log_bg)
    
    fdr_gt = ((mean_gt_log_target - mean_gt_log_bg)**2) / (var_gt_log_target + var_gt_log_bg + 1e-10)
    print(f"  [Field Metric 4] GT-based FDR: {fdr_gt:.4f} (Higher means better statistical separability)")

    # ---------------------------------------------------------
    # 指标 5: GT-based Background Suppression Ratio (背景抑制比)
    # 使用线性空间计算，反映真实的物理数值比例变化
    # ---------------------------------------------------------
    mean_gt_linear_target = np.mean(gt_linear_target)
    mean_gt_linear_bg = np.mean(gt_linear_bg)
    
    ratio_gt = mean_gt_linear_target / (mean_gt_linear_bg + 1e-10)
    print(f"  [Field Metric 5] GT-based Suppression Ratio: {ratio_gt:.4f} (Higher means target is relatively enhanced)")

else:
    print("  [Field Metrics] N/A (Missing target or background regions in GT)")