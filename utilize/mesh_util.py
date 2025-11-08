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

def read_mshv2_triangular(filename:str):
    """ read Gmsh file with version 2.2 and return

    Returns:
        nodes (npt.Ndarray): (N, 3)，存储节点坐标 \\
        triangles (npt.Ndarray): (T, 3)，存储三角形单元的节点索引(0-based)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 去掉每行末尾的换行符，方便处理
    lines = [line.strip() for line in lines]

    # 1. 找到 "$Nodes" 段落并读取节点
    try:
        idx_nodes_start = lines.index('$Nodes')
    except ValueError:
        raise ValueError("未在文件中找到 $Nodes 段落，请检查文件格式。")

    # 读取节点数 N
    N = int(lines[idx_nodes_start + 1])
    # 读取 N 行节点数据
    node_data_lines = lines[idx_nodes_start + 2 : idx_nodes_start + 2 + N]

    # 将节点数据存储到一个列表中
    nodes = []
    for line in node_data_lines:
        parts = line.split()
        # parts[0] 是节点编号(1-based)，此处可忽略
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        nodes.append([x, y, z])
    nodes = np.array(nodes, dtype=float)

    # 2. 找到 "$Elements" 段落并读取单元
    try:
        idx_elements_start = lines.index('$Elements')
    except ValueError:
        raise ValueError("未在文件中找到 $Elements 段落，请检查文件格式。")

    # 读取单元数 M
    M = int(lines[idx_elements_start + 1])
    # 读取 M 行单元数据
    elem_data_lines = lines[idx_elements_start + 2 : idx_elements_start + 2 + M]

    # 用于存储三角形单元
    triangles = []

    for line in elem_data_lines:
        parts = line.split()
        # parts[0] 是单元编号
        elm_type = int(parts[1])      # 单元类型
        num_tags = int(parts[2])      # 标签数量

        # Gmsh v2 格式中，三角形的 elementType == 2
        if elm_type == 2:
            # 节点编号开始位置 = 3 + num_tags
            node_indices_start = 3 + num_tags
            # 对于三角形，后面有 3 个节点编号
            n1 = int(parts[node_indices_start])   - 1  # 转换为0-based索引
            n2 = int(parts[node_indices_start+1]) - 1
            n3 = int(parts[node_indices_start+2]) - 1
            triangles.append([n1, n2, n3])

    triangles = np.array(triangles, dtype=int)

    return nodes, triangles

def write_mshv2_triangular(filename:str, nodes:npt.NDArray, triangles:npt.NDArray):
    """ 写入版本为2的三角形网格的.msh文件
    mshv2中，采用1-based索引
    
    Args:
        nodes (npt.NDArray): 节点列表，每个元素为 (x, y) 或 (x, y, z)
        triangles (npt.NDArray): 三角形面片列表，每个元素为 (i, j, k), 假定索引从0开始
        filename (str): 输出文件名
    """
    with open(filename, "w") as f:
        # 写入 MeshFormat 部分
        f.write("$MeshFormat\n")
        # 版本号2.2，文件类型0（ASCII），数据大小8
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # 写入 Nodes 部分
        f.write("$Nodes\n")
        f.write("{}\n".format(len(nodes)))
        for i, node in enumerate(nodes, start=1):
            # 如果节点只有两个坐标，则默认 z=0.0
            if node.shape[0] == 2:
                x, y = node
                z = 0.0
            else:
                x, y, z = node
            f.write("{:d} {:.8f} {:.8f} {:.8f}\n".format(i, x, y, z))
        f.write("$EndNodes\n")
        
        # 写入 Elements 部分
        f.write("$Elements\n")
        f.write("{}\n".format(triangles.shape[0]))
        for i, tri in enumerate(triangles, start=1):
            # Gmsh中，三角形单元的类型为2
            # 此处 tags 数量设为0（可以根据需要增加物理区域等信息）
            # 注意：将0开始的索引转换为1开始
            n1, n2, n3 = tri
            f.write("{} 2 0 {} {} {}\n".format(i, n1+1, n2+1, n3+1))
        f.write("$EndElements\n")

def mesh_obj_tri(obj_shape:List[float], seed_size:float)->Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """将二维对象生成三角形网格
    Args:
        obj_shape (List[float]): [length, width]
        seed_size (float): 网格尺寸
    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]: 节点、边、单元
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