""" mesh_util.py
一些网格处理的实用函数
created on 2025-11-4
"""
import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from scipy.spatial import Delaunay

def extract_edge_from_face(F: np.ndarray) -> np.ndarray:
    """ 从三角形面片索引 F 中提取唯一的边索引。

    Args:
        F (np.ndarray): 形状为 (M, 3) 的面片索引数组。

    Returns:
        np.ndarray: 形状为 (NumUniqueEdges, 2) 的唯一边索引数组。
    """
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("输入的面片 F 必须是 (M, 3) 形状。")
    
    # 1. 提取所有边 (3M, 2)
    # (M, 3) -> (M, 3, 2) -> (3M, 2)
    edges = np.vstack([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]]
    ])
    
    # 2. 排序每条边的索引，使 (i, j) 和 (j, i) 一致
    edges = np.sort(edges, axis=1)
    
    # 3. 找到唯一的边
    unique_edges = np.unique(edges, axis=0)
    
    return unique_edges

def mesh_obj_tri(obj_shape:List[float], seed_size:float)->Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """ 将二维对象生成三角形网格
    Args:
        obj_shape (List[float]): [length, width]
        seed_size (float): 网格尺寸
    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]: 0-based, 节点、边、单元
    """
    length, width = obj_shape

    length_n = int(length / seed_size)
    width_n = int(width / seed_size)

    length_n = length_n if abs(length - length_n * seed_size) < 1.e-6 else length_n + 1
    width_n = width_n if abs(width - width_n * seed_size) < 1.e-6 else width_n + 1

    xx, yy = np.meshgrid(np.linspace(0, length, length_n+1), np.linspace(0, width, width_n+1))
    xx_pad = xx.flatten('C')
    yy_pad = yy.flatten('C')
    node = np.array([xx_pad, yy_pad], dtype=float).T         # dim: N*2

    tri = Delaunay(node)
    element = np.sort(tri.simplices, axis=1)

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_temp = tuple(sorted(simplices[[i, (i + 1) % 3]]))
            edge_set.add(edge_temp)

    edge = np.array(list(edge_set), dtype=int)

    return node, edge, element

def find_boundary_edges(F: np.ndarray) -> np.ndarray:
    """
    从一个三角网格的面片列表 F 中，查找所有边界边。
    一条边是边界边，如果它在整个网格中只出现一次（即只属于一个三角形）。

    Args:
        F (np.ndarray): 形状为 (M, 3) 的面片索引数组。

    Returns:
        np.ndarray: (NumBoundaryEdges, 2) 形状的数组，包含所有边界边的索引。
    """
    if F is None or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("输入的面片 F 必须是 (M, 3) 形状的 NumPy 数组。")

    # 1. 提取所有边 (3*M, 2)， M 是三角形数量
    edges = np.vstack([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]]
    ])
    
    # 2. 排序每条边的索引，使 (i, j) 和 (j, i) 一致
    edges = np.sort(edges, axis=1)
    
    # 3. 找到唯一的边，并 *计算它们的出现次数*
    #    unique_edges: (NumUnique, 2)
    #    counts: (NumUnique,)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    
    # 4. 筛选出所有只出现一次的边，这些就是边界边
    boundary_edges = unique_edges[counts == 1]
    
    return boundary_edges

def find_boundary_node_indices(F: np.ndarray) -> np.ndarray:
    """
    从一个三角网格的面片列表 F 中，查找所有边界节点的索引。
    一个节点是边界节点，如果它位于一条边界边上。
    此函数通过先查找边界边，再从边界边中提取节点来实现。

    Args:
        F (np.ndarray): 形状为 (M, 3) 的面片索引数组。

    Returns:
        np.ndarray: (NumBoundaryNodes,) 形状的一维数组，包含所有边界节点的索引，已排序且唯一。
    """
    
    # 1. [第一步] 调用函数找到所有的边界边
    #    (NumBoundaryEdges, 2)
    boundary_edges = find_boundary_edges(F)
    
    # 2. [第二步] 从边界边中提取所有节点索引
    #    .flatten() 将 (NumBoundaryEdges, 2) 展平为 (NumBoundaryEdges * 2,)
    #    np.unique() 会自动排序并返回唯一的节点索引
    boundary_node_indices = np.unique(boundary_edges.flatten())
    
    return boundary_node_indices