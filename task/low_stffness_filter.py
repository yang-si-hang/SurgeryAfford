"""
在刚度和方差空间中, 使用双阈值法对单元进行分类出低刚度区区域 (低刚度和低方差), 并可视化结果

Date: 2026-01-23
"""
import numpy as np
import matplotlib.pyplot as plt

from utilize.mesh_io import read_mshv2_triangular
from const import MESH_DIR, OUTPUT_DIR, VISUALIZATION_DIR


if __name__ == "__main__":
    neighbors_path = MESH_DIR / "pd_stretch_demo_mesh_neighbors.csv"
    neighbors = np.loadtxt(neighbors_path, delimiter=",")

    variance_path = OUTPUT_DIR / "P_diag_ekf.csv"
    variance_vec = np.loadtxt(variance_path, delimiter=",")
    variance_vec = np.sqrt(variance_vec)

    stiffness_path = OUTPUT_DIR / "estimated_stiffness_ekf.csv"
    stiffness = np.loadtxt(stiffness_path, delimiter=",")

    mesh_path = MESH_DIR / "pd_stretch_demo_mesh_init.msh"
    
    V, F = read_mshv2_triangular(mesh_path)
    mesh_data = {"V": V, "F": F}

    n_cells = len(F)
    edge_to_cells = {}
    for i, cell in enumerate(F):
        for edge in [tuple(sorted((cell[0], cell[1]))), tuple(sorted((cell[1], cell[2]))), tuple(sorted((cell[2], cell[0])))]:
            edge_to_cells.setdefault(edge, []).append(i)

    edge_list = []
    for edge, cell_indices in edge_to_cells.items():
        if len(cell_indices) == 2:
            u, v = cell_indices
            edge_list.append((u, v))
        else:
            # 边界边，只连接同一单元的循环边
            pass

    adjacency_list = [[] for _ in range(n_cells)]
    for edge in edge_list:
        u, v = edge
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    ### 设定双阈值 ###
    stiffness_low = np.percentile(stiffness, 20)
    variance_low = np.percentile(variance_vec, 20)

    stiffness_lablels = np.array([0 if s < stiffness_low else 1 for s in stiffness])
    variance_labels = np.array([0 if v < variance_low else 1 for v in variance_vec])

    union_labels = np.zeros_like(stiffness_lablels)
    for i in range(len(stiffness_lablels)):
        if stiffness_lablels[i] == 0 and variance_labels[i] == 0:
            union_labels[i] = 0  # Low
        else:
            union_labels[i] = 1  # Mid

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    ax1.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                  facecolors=stiffness_lablels, cmap='viridis', edgecolors='k')
    ax1.set_title("Stiffness Labels")
    fig.colorbar(ax1.collections[0], ax=ax1, label='Stiffness Variance')
    ax1.set_aspect('equal')

    cmap_custom = plt.get_cmap('tab10', 2)
    im2 = ax2.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                        facecolors=variance_labels, cmap=cmap_custom, edgecolors='k')
    cbar = fig.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Low (0)', 'Mid (1)'])
    ax2.set_title("Variance Labels")
    ax2.set_aspect('equal')

    im3 = ax3.tripcolor(mesh_data["V"][:, 0], mesh_data["V"][:, 1], mesh_data["F"], 
                        facecolors=union_labels, cmap=cmap_custom, edgecolors='k')
    cbar3 = fig.colorbar(im3, ax=ax3, ticks=[0, 1])
    cbar3.ax.set_yticklabels(['Union Low (0)', 'Others (1)'])
    ax3.set_title("Low Stiffness & Low Variance Region")
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/stiffness&variance_low.svg", dpi=300)

    # plt.show()