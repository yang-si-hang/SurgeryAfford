""" 用于可视化三角形网格单元上的属性值
created on 2025-11-16
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from const import MESH_DIR, OUTPUT_DIR, ROOT_DIR
from utilize.mesh_io import read_mshv2_triangular


def visualize_triangle_property(nodes_pos:np.ndarray, triangles:np.ndarray, cell_attrs:np.ndarray,
                                cmap='RdBu_r', edgecolors='k',
                                title='2D Triangle Mesh Visualization',
                                colorbar_label='Attribute', output_path=None,
                                dpi=150, show_fig=False):
    """
    可视化三角形网格单元属性。
    Args:
      nodes_pos: (N,2) array-like, 节点坐标
      triangles: (M,3) array-like, 三角形拓扑（节点索引，从0开始）
      cell_attrs: (M,) array-like, 每个单元的属性值（按三角形顺序）
      output_path: 若提供则保存为该路径（如 "triangle.svg"），否则不保存
      show_fig: 若为 True 则调用 plt.show()
    Returns:
      (fig, ax) matplotlib 对象
    """
    nodes_pos = np.asarray(nodes_pos)
    triangles = np.asarray(triangles, dtype=int)
    cell_attrs = np.asarray(cell_attrs)

    if nodes_pos.ndim != 2 or nodes_pos.shape[1] < 2:
        raise ValueError("nodes_pos must be shape (N,2) or greater.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be shape (M,3).")
    if cell_attrs.shape[0] != triangles.shape[0]:
        raise ValueError("cell_attrs length must match number of triangles.")

    x = nodes_pos[:, 0]
    y = nodes_pos[:, 1]

    triang = mtri.Triangulation(x, y, triangles)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pc = ax.tripcolor(triang, facecolors=cell_attrs, cmap=cmap, edgecolors=edgecolors)
    fig.colorbar(pc, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_aspect('equal')

    if output_path:
        fig.savefig(output_path, dpi=dpi)
    if show_fig:
        plt.show()

    return fig, ax