import pickle
import numpy as np
import matplotlib.pyplot as plt

from utilize.mesh_io import read_mshv2_triangular
from const import *

with open(OUTPUT_DIR / "evaluation_results_3_1.pkl", 'rb') as f:
    loaded_results = pickle.load(f)

max_entropy_reduction = max(loaded_results.items(), key=lambda x: x[1]['entropy_reduction'])
print(f"Contact Index with Max Entropy Reduction: {max_entropy_reduction[0]}")
print(f"Max Entropy Reduction: {max_entropy_reduction[1]['entropy_reduction']:.4f}")
print(f"Corresponding Action: {max_entropy_reduction[1]['optimal_action']}")

max_loss_increment = max(loaded_results.items(), key=lambda x: x[1]['fun_increment'])
print(f"\nContact Index with Max Loss Increment: {max_loss_increment[0]}")
print(f"Max Loss Increment: {max_loss_increment[1]['fun_increment']:.4f}")
print(f"Corresponding Action: {max_loss_increment[1]['optimal_action']}")

# ==========================================
# 1. 提取数据与相关性计算
# ==========================================
contact_indices = list(loaded_results.keys())
fun_increments = [np.sqrt(loaded_results[idx]['fun_increment']) for idx in contact_indices]
entropy_reductions = [loaded_results[idx]['entropy_reduction'] for idx in contact_indices]

# 计算皮尔逊相关系数 (Pearson correlation)
correlation_matrix = np.corrcoef(fun_increments, entropy_reductions)
corr_coeff = correlation_matrix[0, 1]
print(f"Correlation coefficient between Loss Increment and Entropy Reduction: {corr_coeff:.4f}")

# ==========================================
# 2. 绘制散点图与趋势线
# ==========================================
plt.style.use('default')
params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(params)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.scatter(fun_increments, entropy_reductions, color='#3498db', alpha=0.8, edgecolors='k', s=50)

# 添加线性拟合趋势线
m, b = np.polyfit(fun_increments, entropy_reductions, 1)
ax.plot(np.array(fun_increments), m * np.array(fun_increments) + b, 
        color='#e74c3c', linestyle='--', linewidth=2, label=f'Trend Line (r={corr_coeff:.2f})')

ax.set_xlabel('Loss Increment (Objective Function)')
ax.set_ylabel('Entropy Reduction (Information Gain)')
ax.set_title('Correlation: Loss Increment vs Entropy Reduction', pad=15)
ax.legend(frameon=False)

# 保存相关性图
plt.savefig(VISUALIZATION_DIR / "correlation_loss_vs_entropy.svg", transparent=False, format='svg', dpi=300, bbox_inches='tight')


# ==========================================
# 3. 提取 Top-N 结果并准备协方差数据
# ==========================================
import seaborn as sns

MESH_PATH = MESH_DIR / "pd_stretch_tissue_mesh_init_2.msh"
node_pos_np, triangles = read_mshv2_triangular(MESH_PATH)

TOP_N = 3  # 你可以修改为你想要显示的 Top N 数量

# 按 fun_increment 降序排序
sorted_by_loss = sorted(loaded_results.items(), key=lambda x: x[1]['fun_increment'], reverse=True)
top_n_contacts = sorted_by_loss[:TOP_N]

# 提取最优接触点对应的协方差矩阵的对角线（即每个单元的刚度方差）
best_contact_idx = top_n_contacts[0][0]
best_covariance = loaded_results[best_contact_idx]['covariance']
# 提取对角线作为 variance (为了防止数值精度出现负数或 0，限制最小值为 1e-10)
sigma_k = np.clip(np.diag(best_covariance), a_min=1e-10, a_max=None)

np.savetxt(OUTPUT_DIR/"P_diag_new.csv", sigma_k, delimiter=",")

# ==========================================
# 4. 绘制网格与动作向量 (结合你的代码)
# ==========================================
cmap_custom = sns.color_palette("RdYlBu_r", as_cmap=True)

variance = sigma_k.copy()
v_min, v_max = variance.min(), variance.max()
# 自动计算对数刻度范围
tick_locs = np.arange(np.floor(np.log10(v_min)), np.ceil(np.log10(v_max)) + 1)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

# 绘制带有方差颜色的三角形网格
# 提示: 请确保 node_pos_np 和 triangles 已经在当前作用域中加载
im1 = ax1.tripcolor(node_pos_np[:, 0], node_pos_np[:, 1], triangles, 
                    facecolors=np.log10(variance), shading='flat', 
                    edgecolors='#333333', linewidth=1., alpha=0.9, cmap=cmap_custom,    
                    vmin=tick_locs[0], vmax=tick_locs[-1])

ax1.set_aspect('equal')
ax1.axis('off')

# 自定义 Colorbar 的刻度
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.01, shrink=0.8, aspect=20, fraction=0.046)
cbar1.ax.tick_params(length=0, labelsize=14)
cbar1.set_ticks(tick_locs)
cbar1.set_ticklabels([f"$10^{{ {int(loc):d} }}$" for loc in tick_locs])

# ==========================================
# 5. 在网格上叠加绘制 Top-N 最优动作
# ==========================================
# 动作的可视化缩放系数
VISUAL_SCALE = 0.5  

colors_for_actions = ['#27ae60', '#8e44ad', '#f39c12'] # Top 1, 2, 3 的箭头颜色

for i, (contact_idx, data) in enumerate(top_n_contacts):
    action = data['optimal_action']
    
    # 假设 contact_idx 可以直接作为索引在 node_pos_np 中找到对应的物理坐标
    # 如果你的 contact_idx 是面(face)索引，这里需要改为求单元格中心坐标
    start_x = node_pos_np[contact_idx, 0]
    start_y = node_pos_np[contact_idx, 1]
    
    dx = action[0] * VISUAL_SCALE
    dy = action[1] * VISUAL_SCALE
    
    # 绘制动作箭头
    ax1.arrow(start_x, start_y, dx, dy, 
              head_width=0.005, head_length=0.005, 
              fc=colors_for_actions[i % len(colors_for_actions)], 
              ec='black', linewidth=1.5, zorder=5)
    
    # 在箭头起点旁边添加文本标注
    # ax1.text(start_x - 0.01, start_y + 0.01, f'Top {i+1}', 
    #          color=colors_for_actions[i % len(colors_for_actions)], 
    #          fontsize=12, fontweight='bold', zorder=6,
    #          bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# 保存可视化结果
plt.savefig(VISUALIZATION_DIR / "stiffness_variance_with_actions.svg", transparent=False, format='svg', dpi=300, bbox_inches='tight')