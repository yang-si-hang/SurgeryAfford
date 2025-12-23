""" 尝试统一mask-to-mesh-to-model的流程
最终计算affordance值
created on 2025-11-7
"""
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import shapely
import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

from const import OUTPUT_DIR, MESH_DIR
from gen_mesh import MaskTo2DMesh
from diffpd_2d_candi_contact import Soft2DNocontact
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular
from utilize.mesh_util import mesh_obj_tri, find_boundary_node_indices

def build_convex_hull_polygon(points: np.ndarray) -> Polygon:
    """
    接收一个 (N, 2) 的点集，计算它们的凸包，并返回一个 Shapely 的 Polygon 对象。

    Args:
        points: 形状为 (N, 2)，代表 N 个二维点。

    Returns:
        shapely.geometry.Polygon: 一个代表点集凸包的多边形对象。
    """
    if points.shape[0] < 3:
        raise ValueError(f"至少需要3个点来构建凸包，但只收到了 {points.shape[0]} 个。")
        
    try:
        # 1. 使用 SciPy 计算凸包
        hull = ConvexHull(points)
        
        # 2. 提取凸包的顶点坐标 (按顺序)
        hull_vertices = points[hull.vertices]
        hull_polygon = Polygon(hull_vertices)   # 3. 使用 Shapely 创建多边形对象
        
        return hull_polygon
        
    except Exception as e:
        print(f"构建凸包时出错: {e}")   # 在 QHull 错误时 (例如所有点共线) 返回空
        return Polygon()

def calculate_distances_to_hull(hull_polygon: Polygon, 
                                query_points: np.ndarray) -> np.ndarray:
    """
    计算一个或多个点到预先计算好的凸包多边形的距离。
    - 如果点在凸包内部或边界上，距离为 0。
    - 如果点在凸包外部，距离为最短欧几里得距离。

    Args:
        hull_polygon: 从 build_convex_hull_polygon() 得到的凸包对象。
        query_points: 一个 NumPy 数组，形状为 (K, 2)，代表 K 个要查询的点。

    Returns:
        np.ndarray: 一个形状为 (K,) 的数组，包含每个点到凸包的距离。
    """
    # 1. 将 (K, 2) 的 NumPy 数组转换为 Shapely 的点数组。(这比用 for 循环创建 Point() 对象快得多)
    shapely_points = shapely.points(query_points)
    
    # 2. 矢量化计算所有点到多边形的距离
    #    shapely.distance() 自动处理内部/外部点：
    #    - 内部点 -> 0.0
    #    - 外部点 -> 最短距离
    distances = shapely.distance(hull_polygon, shapely_points)
    
    return distances

@ti.data_oriented
class Soft2D(Soft2DNocontact):
    def __init__(self, shape, fix:List[int], candi_contact:List[int], 
                 E:float, nu:float, dt:float, density:float, **kwargs):
        super().__init__(shape, fix, E, nu, dt, density, **kwargs)
        self.candi_contact = candi_contact
        self.CON_N:int = 1
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)
        self.control_hull = dict()
        self.construct_control_convexhull()

        self.precomputation()
        self.substep(0)
        self.construct_g_hessian()
        self.A_pre, self.B_pre = self.model_gradient_const()

    def construct_control_convexhull(self):
        """ 构建控制点的凸包多边形，用于后续距离计算
        """
        candi_contact = self.candi_contact
        fix = self.fix_particle_list
        node_np = self.node_pos.to_numpy()
        fix_pos = node_np[fix]
        for i, idx in enumerate(candi_contact):
            all_points = np.vstack([fix_pos, node_np[idx]])
            hull_polygon = build_convex_hull_polygon(all_points)
            self.control_hull[idx] = hull_polygon

    def construct_controllability(self, contact_idx, query_points:np.ndarray):
        """ 计算控制点对query_points的可控性，基于距离的反比
        Args:
            query_points: np.ndarray, (M, 2) M为查询点数量。查询点的位置。
        Returns:
            controlability: np.ndarray, (M, ) control convexhull对查询点的可控性矩阵
        """
        s = 1.e-1  # 控制距离与可控性之间的比例系数。TODO：需要多次调整
        M = query_points.shape[0]
        controlability = np.zeros((M,), dtype=np.float64)

        hull_polygon = self.control_hull[contact_idx]
        distances = calculate_distances_to_hull(hull_polygon, query_points)
        # 转化为可控性值，距离越近可控性越高
        controlability = 1.0 / (s * distances + 1.)

        return controlability

    def model_gradient(self, contact_idx:int):
        """ 直接将contact相关的项加到A_pre和B_pre上，得到最终的A和B矩阵
        """
        A = self.A_pre.copy()
        A[contact_idx*2, contact_idx*2] += self.positional_weight
        A[contact_idx*2+1, contact_idx*2+1] += self.positional_weight

        B = self.B_pre[:, contact_idx*2:contact_idx*2+2].copy() # 取出对应contact点的两列
        B[contact_idx*2, 0] += self.positional_weight
        B[contact_idx*2+1, 1] += self.positional_weight

        dq_dcontact_np = np.linalg.solve(A, B)
        return dq_dcontact_np
    
    def construct_loss_gradient(self, feature_pos:np.ndarray):
        # 默认为多个点对之间的向量长度距离，专门针对缝合任务
        M = feature_pos.shape[0]
        if M % 2 != 0:
            raise ValueError("feature_pos的点数必须为偶数。")
        
        loss = 0.0
        gradient = np.zeros(feature_pos.size, dtype=np.float64)
        # 以下均为手工推导结果
        for i in range(M // 2):
            p1 = feature_pos[2*i]
            p2 = feature_pos[2*i + 1]
            dir_vec = p1 - p2

            loss += dir_vec @ dir_vec

            gradient[4*i:4*i+2] = 2 * dir_vec @ np.eye(2)
            gradient[4*i+2:4*i+4] = -2 * dir_vec @ np.eye(2)

        return loss, gradient

    def compute_task_afford(self, feature_pos:np.ndarray, points_j:np.ndarray):
        """ 
        Args:
            feature_pos: np.ndarray, (M, 2) M为特征点数量。特征点的位置。
            points_j: np.ndarray, (2*M, 2*N) M为特征点数量，N为节点数量。特征点对于节点的梯度矩阵
        """
        results = {}
        afford_vec = np.zeros((len(self.candi_contact), 2), dtype=np.float64)
        candi_contact = self.candi_contact

        loss, grad = self.construct_loss_gradient(feature_pos)
        for i, idx in enumerate(candi_contact):
            controllability = self.construct_controllability(idx, feature_pos)
            controllability_matrix = np.kron(np.diag(controllability), np.eye(2))  # (2M, 2M)
            dq_dcontact = self.model_gradient(idx)
            dloss_dcontact = grad @ controllability_matrix @ points_j @ dq_dcontact  # 链式法则
            afford = np.linalg.norm(dloss_dcontact)
            results[idx] = afford
            afford_vec[i] = dloss_dcontact
            # print(f"loss gradient: {grad}")
            # np.savetxt(f"points_j.csv", points_j, fmt="%.8f", delimiter=",")
            # np.savetxt(f"dq_dcontact_{idx}.csv", dq_dcontact, fmt="%.8f", delimiter=",")
            print(f"Contact idx: {idx}; Affordance value: {afford:e}")
        return results, afford_vec

if __name__ == "__main__":
    # 创建一个mask
    H, W = 512, 512
    center = (W // 2, H // 2)
    radius = 150
    
    mask = np.zeros((H, W), dtype=np.uint8)
    # cv2.circle(mask, center, radius, 255, -1) # 255表示目标, -1表示填充
    cv2.rectangle(mask, (100, 100), (400, 200), 1, -1)

    fixed_points = np.array([[150, 100],
                             [150, 125],
                             [150, 150],
                             [150, 175],
                             [150, 200]])
    
    mesher = MaskTo2DMesh(
        boundary_resolution=100,   # 边界采样点数
        smooth_factor=1.0,         # 平滑度 (可调)
        mesh_max_area=50.0,        # 网格密度 (越小越密)
        mesh_min_angle=25.0        # 网格质量 (建议 20-34)
    )
    
    try:
        V, F = mesher.generate_mesh(mask)
        fixed_nodes_idx = mesher.find_nearest_nodes(fixed_points, k=3)
        fixed_nodes_idx_flat = np.unique(fixed_nodes_idx.flatten())
        fixed_nodes_pos = V[fixed_nodes_idx_flat]

        print(f"Nodes num: {V.shape[0]}; Faces num: {F.shape[0]}")

        write_mshv2_triangular(f"{MESH_DIR}/output_mesh.msh", np.array(V), np.array(F))
        print(f"网格已保存为 '{MESH_DIR}/output_mesh.msh'。")

        # 处理缝合线输入
        suture_start = np.array([300.0, 140.0])
        suture_end = np.array([285.0, 160.0])
        probe_points = mesher.define_probes(suture_start, suture_end, probe_length_factor=2.0)
        print(f"定义的探针端点:\n{probe_points}")

        probe_points_mapping = mesher.map_points_to_mesh(probe_points, k_neighbors=10) # 映射到网格的坐标
        # print(f"探针点映射结果:\n{probe_points_mapping}")

        mapping_points_indices = [item["v_indices"] for item in probe_points_mapping]
        mapping_points_indices_flat = [item for sublist in mapping_points_indices for item in sublist]  # 展平
        mapping_points = V[mapping_points_indices_flat]

        points_j = mesher.build_point_jacobian(probe_points_mapping)

        boundary_node_indices = find_boundary_node_indices(F)
        # print(f"Boundary node indices: {boundary_node_indices}")

        # 构建变形模型，计算candidate contact的affordance
        mesh_file = MESH_DIR / "output_mesh.msh"
        soft_model = Soft2D(shape=mesh_file, fix=fixed_nodes_idx_flat, candi_contact=boundary_node_indices,
                            E=1.e6, nu=0.3, dt=1.e-2, density=1.e-2)    # 注意长度单位改变后的杨氏模量尺度

        contacts_afford, contacts_jacobian = soft_model.compute_task_afford(probe_points, points_j)
        output_file = OUTPUT_DIR / "affordance_results.txt"

        with open(output_file, 'w') as f:
            for idx, afford in contacts_afford.items():
                f.write(f"Contact idx: {idx}; Affordance value: {afford:.8f}\n")
        print(f"Affordance结果已保存到 '{output_file}'。")

        # =========== 可视化代码 =========
        unique_fixed_nodes = np.unique(fixed_nodes_pos)     # 去重逻辑有点问题，不过不重要
        boundary_nodes_pos = V[boundary_node_indices]
        boundary_node_afford = list(contacts_afford.values())
        scale = 1./ max(boundary_node_afford)
        boundary_node_size = 20 * scale * np.array(boundary_node_afford)

        plt.figure(figsize=(10, 5))
        
        # 子图1: 原始Mask
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title('Input: Origin Mask')
        
        # 子图2: 生成的网格
        plt.subplot(1, 2, 2)
        plt.triplot(V[:, 0], V[:, 1], F, 'bo-', lw=0.5, markersize=1)
        plt.plot(fixed_points[:, 0], fixed_points[:, 1], 
                 'r*', markersize=6, label='Fixed Points (Input)')
        # plt.plot(fixed_nodes_pos[:, 0], fixed_nodes_pos[:, 1], 
        #          'g.', markersize=6, label='Fixed Nodes (Found)')
        # plt.scatter(mesher.smoothed_contour[:, 0], mesher.smoothed_contour[:, 1], 
        #             color='red', s=5, label='Smoothed Contour')
        plt.plot([suture_start[0], suture_end[0]], 
                 [suture_start[1], suture_end[1]], 
                 color='orange', linewidth=2, label='Suture Line')
        plt.plot(probe_points[:, 0], probe_points[:, 1], 
                 'y*', markersize=6, label='Probe Points')
        # plt.plot(mapping_points[:, 0], mapping_points[:, 1], 
        #          'g.', markersize=6, label='Mapped Points')
        plt.scatter(boundary_nodes_pos[:, 0], boundary_nodes_pos[:, 1], 
                    s=boundary_node_size, color='k', zorder=2, label='Candidate Contact Nodes')
        plt.quiver(boundary_nodes_pos[:, 0], boundary_nodes_pos[:, 1], 
                   contacts_jacobian[:, 0], contacts_jacobian[:, 1], color='k', 
                   scale_units='xy', # 关键参数：使箭头长度与数据单位一致
                   scale=0.2,          # 关键参数：调整此值以控制箭头显示长度 (例如，如果所有向量长度都在1左右，scale=1会使它们显得很大，可以调大到20, 50来缩小它们)
                   angles='xy', 
                   width=0.005,       # 箭头的线宽，相对于图形大小
                   headwidth=5,      # 箭头头部的宽度
                   headlength=8,     # 箭头头部的长度
                   zorder=3, label='Affordance Vectors')
        plt.title(f'Output: 2D Triangle Mesh\n(V={V.shape[0]}, F={F.shape[0]})')
        # plt.legend()
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()  # 匹配图像坐标系 (Y轴向下)
        
        plt.margins(x=0.05, y=0.2)
        plt.tight_layout()
        plt.savefig("mask&mesh.svg", dpi=300)

    except ValueError as e:
        print(f"处理失败: {e}")
