"""
在取对数后的标准差空间中, 使用GMM对单元进行三分类, 并结合Graph Cut方法提取高不确定性区域.

Date: 2026-01-23
"""
import numpy as np
import maxflow  # pip install PyMaxflow
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pathlib import Path

from utilize.mesh_io import read_mshv2_triangular
from const import MESH_DIR, OUTPUT_DIR, VISUALIZATION_DIR, ARTICLE_VIS_DIR

hard_ele_list = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
# hard_ele_list = [5, 6, 31, 86, 94, 124, 125, 131, 136, 140, 145, 151, 177, 
#                  179, 185, 213, 219, 252, 255]
# hard_ele_list = [48, 55, 63, 91, 94, 126, 128, 138, 174, 201, 204, 247, 250, 254, 255, 
#                  256, 297, 327, 330, 379, 390, 391, 393, 396, 397, 401, 403, 412] + \
#                 [238, 278, 287, 288, 359, 365, 366, 367, 370, 374, 414]

neighbors_path = MESH_DIR / "pd_stretch_demo_mesh_neighbors.csv"
neighbors = np.loadtxt(neighbors_path, delimiter=",")

variance_path = OUTPUT_DIR / "P_diag_ekf.csv"
variance_vec = np.loadtxt(variance_path, delimiter=",")
variance = np.log10(np.sqrt(variance_vec))

variance = np.loadtxt(f"{OUTPUT_DIR}/estimated_stiffness_ekf.csv", delimiter=",")
variance = np.log10(variance)

variance = np.loadtxt(f"{OUTPUT_DIR}/stretch_weight_update.csv", delimiter=",")

mesh_path = MESH_DIR / "pd_stretch_demo_mesh_init.msh"

V, F = read_mshv2_triangular(mesh_path)
mesh_data = {"V": V, "F": F}

n_cells = len(F)
edge_to_cells = {}
for i, cell in enumerate(F):
    for edge in [tuple(sorted((cell[0], cell[1]))), tuple(sorted((cell[1], cell[2]))), tuple(sorted((cell[2], cell[0])))]:
        edge_to_cells.setdefault(edge, []).append(i)

# ==========================================
# 2. GMM 概率建模 (Data Term)
# ==========================================
print("Step 1: Running GMM...")

# GMM 输入需要是 (n_samples, n_features)
X = variance.reshape(-1, 1)

# 改为拟合两个高斯分布：低不确定性区域 vs 高不确定性区域
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# 获取两个分布的均值，判断哪个是“高不确定性”类
means = gmm.means_.flatten()
high_uncertainty_idx = np.argmax(means)  # 均值大的作为高不确定性区域 (Source)
low_uncertainty_idx = np.argmin(means)   # 均值小的作为低不确定性区域 (Sink)

print(f"  - Low Uncert Mean: {means[low_uncertainty_idx]:.4f}")
print(f"  - High Uncert Mean: {means[high_uncertainty_idx]:.4f}")

# 计算属于每一类的后验概率
probs = gmm.predict_proba(X)
prob_high = probs[:, high_uncertainty_idx]  # 属于高不确定性的概率
prob_low = probs[:, low_uncertainty_idx]    # 属于低不确定性的概率

# 将低不确定性区域作为 Graph Cut 的背景 (Sink)
prob_sink = prob_low

# 防止 log(0) 错误，加一个极小值
epsilon = 1e-10
prob_high = np.clip(prob_high, epsilon, 1.0 - epsilon)
prob_sink = np.clip(prob_sink, epsilon, 1.0 - epsilon)

# 计算 t-link (Terminal Link) 的能量项 (负对数似然)
# Source (S) = High Uncertainty, Sink (T) = Low Uncertainty
data_term_weight_to_source = -np.log(prob_sink)  
data_term_weight_to_sink = -np.log(prob_high)

# ==========================================
# 3. Graph Cut 构建与求解 (Smoothness Term)
# ==========================================
print("Step 2: Building Graph...")

# Lambda: 平滑系数。调节它来控制“分割块”的连续程度。
# 越大，区域越整块（可能丢失细节）；越小，越破碎（接近单纯的阈值法）。
# 经验值通常在 1.0 到 10.0 之间，需要根据 variance 的数值范围微调。
LAMBDA = 3.0
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
# 批量添加以提高效率
for i in range(n_cells):
    # add_tedge(node, capacity_from_source, capacity_to_sink)
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
# np.savetxt(OUTPUT_DIR / "high_uncertainty_indices.csv", high_uncertainty_indices, fmt="%d")

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

colors = ['#F0F0F2', '#B81502'] # D9534F 912C2C
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