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
from const import MESH_DIR, OUTPUT_DIR, VISUALIZATION_DIR

neighbors_path = MESH_DIR / "pd_stretch_demo_mesh_neighbors.csv"
neighbors = np.loadtxt(neighbors_path, delimiter=",")

variance_path = OUTPUT_DIR / "P_diag_ekf.csv"
variance_vec = np.loadtxt(variance_path, delimiter=",")

mesh_path = MESH_DIR / "pd_stretch_demo_mesh_init.msh"

V, F = read_mshv2_triangular(mesh_path)
mesh_data = {"V": V, "F": F}

n_cells = len(F)
edge_to_cells = {}
for i, cell in enumerate(F):
    for edge in [tuple(sorted((cell[0], cell[1]))), tuple(sorted((cell[1], cell[2]))), tuple(sorted((cell[2], cell[0])))]:
        edge_to_cells.setdefault(edge, []).append(i)

variance = np.log10(np.sqrt(variance_vec))

# ==========================================
# 2. GMM 概率建模 (Data Term)
# ==========================================
print("Step 1: Running GMM...")

# GMM 输入需要是 (n_samples, n_features)
X = variance.reshape(-1, 1)

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

# 防止 log(0) 错误，加一个极小值
epsilon = 1e-10
prob_high = np.clip(prob_high, epsilon, 1.0 - epsilon)
prob_bg = np.clip(prob_bg, epsilon, 1.0 - epsilon)

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
data_term_weight_to_source = -np.log(prob_bg)  
data_term_weight_to_sink = -np.log(prob_high) 

# ==========================================
# 3. Graph Cut 构建与求解 (Smoothness Term)
# ==========================================
print("Step 2: Building Graph...")

# Lambda: 平滑系数。调节它来控制“分割块”的连续程度。
# 越大，区域越整块（可能丢失细节）；越小，越破碎（接近单纯的阈值法）。
# 经验值通常在 1.0 到 10.0 之间，需要根据 variance 的数值范围微调。
LAMBDA = 5.0 
SIGMA = 0.5 # 用于高斯核计算相邻差异的权重

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
        weight = LAMBDA * np.exp(-(diff**2) / (2 * SIGMA**2))
        
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
cbar3.ax.set_yticklabels(['Low (0)', 'High (1)'])
ax1.set_title("High Variance Region")
ax1.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{VISUALIZATION_DIR}/high_uncertainty_region.svg", dpi=300)
# plt.show()