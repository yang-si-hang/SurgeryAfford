"""
将实验中每次拉伸的EKF结果中的 Y 矩阵（即 stiffness estimate 的协方差矩阵）相加，得到一个总的 Y 矩阵。
然后可视化这个总的 Y 矩阵的对角线元素（即每个单元 stiffness estimate 的方差），以展示哪些区域的 stiffness estimate 更不确定。
Date: 2026-04-04
"""
from pathlib import Path
import os
import glob
import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from const import *
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular


FILE_DIR = DATA_DIR / "demo" / "real_track" / "05030806"
mesh_data_file = np.load(FILE_DIR / "tissue_mesh.npz")
neighbors = mesh_data_file['neighbors']
V, F = mesh_data_file['vertices'], mesh_data_file['faces']
mesh_data = {"V": V, "F": F}

# 1. 查找所有以 "Y_ekf" 开头的 csv 文件
pattern = os.path.join(FILE_DIR, "Y_ekf*.csv")
file_list = sorted(glob.glob(pattern))

if not file_list:
    print(f"在文件夹 '{FILE_DIR}' 中未找到以 'Y_ekf' 开头的 CSV 文件。")
else:
    print(f"共找到 {len(file_list)} 个文件：")
    for f in file_list:
        print(f"  {f}")

    # 2. 读取所有矩阵并相加
    total_matrix = None
    for file_path in file_list:
        # 读取 CSV 为 DataFrame，再转为 numpy 矩阵
        mat = np.loadtxt(file_path, delimiter=",")

        if total_matrix is None:
            total_matrix = mat
        else:
            total_matrix = total_matrix + mat

        print(f"  已读取: {os.path.basename(file_path)}, 形状: {mat.shape}")

c, lower = cho_factor(total_matrix)
P_mat = cho_solve((c, lower), np.eye(total_matrix.shape[0]))
np.savetxt(Path(FILE_DIR) / "Y_sum_ekf.csv", total_matrix, delimiter=",")
np.savetxt(Path(OUTPUT_DIR) / "Y_sum_ekf.csv", total_matrix, delimiter=",")
np.savetxt(Path(OUTPUT_DIR) / "P_sum_ekf.csv", P_mat, delimiter=",")

cells = [("triangle", F.astype(np.int32))]
mesh = meshio.Mesh(
V[:, :2],
cells,
cell_data={
    "Uncertainty": [np.log10(np.sqrt(np.diag(P_mat)))],
    "Information": [np.log10(np.sqrt(np.diag(total_matrix)))],
}
)
mesh.write(f"{OUTPUT_DIR}/mesh_with_stiffness.vtu")
print(f"Saved mesh with stiffness information to {OUTPUT_DIR}/mesh_with_stiffness.vtu")

triang = mtri.Triangulation(V[:, 0], V[:, 1], F)

plt.figure(figsize=(6, 6))
plt.tripcolor(triang, facecolors=np.log10(np.sqrt(np.diag(total_matrix))), cmap='RdBu_r', edgecolors='k')
plt.colorbar(label='Delta Y Diagonal')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Delta Y Diagonal per Element')

out_path_deltay = Path(VISUALIZATION_DIR) / f"stiffness_inform-real.svg"
plt.savefig(out_path_deltay, dpi=300)
plt.close()
print(f"Saved delta Y diagonal visualization to {out_path_deltay}")


plt.figure(figsize=(6, 6))
plt.tripcolor(triang, facecolors=np.log10(np.sqrt(np.diag(P_mat))), cmap='RdBu_r', edgecolors='k')
plt.colorbar(label='P Diagonal')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'P Diagonal per Element')

out_path_deltay = Path(VISUALIZATION_DIR) / f"stiffness_variance-real.svg"
plt.savefig(out_path_deltay, dpi=300)
plt.close()
print(f"Saved P diagonal visualization to {out_path_deltay}")

exit()

# ==========================================
# 论文结果美化代码
# ==========================================
from matplotlib.colors import ListedColormap
import seaborn as sns

plt.style.use('default') # 重置
params = {
    'font.family': 'serif',        # 衬线体 (Times New Roman 等)
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], # 显式指定
    'mathtext.fontset': 'stix',    # 数学公式字体更像 LaTeX
    'axes.labelsize': 14,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.spines.top': False,      # 去掉顶部边框 (更简洁)
    'axes.spines.right': False,    # 去掉右侧边框
}
plt.rcParams.update(params)

cmap_custom = sns.color_palette("RdYlBu_r", as_cmap=True)

inform = np.sqrt(np.diag(total_matrix.copy()))
v_min, v_max = inform.min(), inform.max()
tick_locs = np.arange(np.ceil(np.log10(v_min)), np.floor(np.log10(v_max))+2)
fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
im1 = ax1.tripcolor(triang, facecolors=np.log10(inform), shading='flat',
                    edgecolors='#333333', linewidth=1., alpha=0.9, cmap=cmap_custom,    # 
                    vmin=tick_locs[0], vmax=tick_locs[-1])
# ax1.set_title('(b) Graph Cut Segmentation (Output)', pad=15)
ax1.set_aspect('equal')
ax1.axis('off')

# 自定义 Colorbar 的刻度
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.01, shrink=0.8, aspect=20, fraction=0.046)
cbar1.ax.tick_params(length=0, labelsize=14)

cbar1.set_ticks(tick_locs)
cbar1.set_ticklabels([f"$10^{{ {int(loc):d} }}$" for loc in tick_locs])

plt.savefig(f"{ARTICLE_VIS_DIR}/stiffness_inform.svg", transparent=True, format='svg', dpi=300, bbox_inches='tight')