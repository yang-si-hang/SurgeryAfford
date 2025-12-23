""" 包含网格文件的读写接口函数
msh，vtu
created on 2025-11-12
"""
import numpy as np
import numpy.typing as npt
import meshio
from typing import Tuple, List

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
        filename (str): 输出文件路径
        nodes (npt.NDArray): 节点列表，每个元素为 (x, y) 或 (x, y, z)
        triangles (npt.NDArray): 三角形面片列表，每个元素为 (i, j, k), 假定索引从0开始
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

def write_vtu(mesh_file:str, pos:npt.NDArray, write_name:str):
    """ Write the node position to a .vtu file baased on the initial mesh file
    Args:
        mesh_file (str): The initial mesh file path
        pos (npt.NDArray): The node position
        write_path (str): The write file path
    """
    _, triangles = read_mshv2_triangular(mesh_file)

    cells_write = [("triangle", triangles)]
    mesh = meshio.Mesh(points=pos, cells=cells_write)
    mesh.write(f"{write_name}")