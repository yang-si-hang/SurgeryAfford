""" 
created on 2025-11-4
"""
import numpy as np
import cv2
import triangle
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import cKDTree

from utilize.mesh_io import write_mshv2_triangular
from utilize.mesh_util import extract_edge_from_face

class MaskTo2DMesh:
    """
    将2D二进制Mask转换为高质量2D三角形网格的类。

    工作流程 (模块化):
    1. [M1] _extract_raw_contour: 提取像素轮廓
    2. [M2] _smooth_and_resample_contour: 平滑并重采样轮廓
    3. [M3] _generate_triangle_mesh: 生成2D网格
    4. [M4] generate_mesh: 按顺序调用1-3，并返回V和F
    """
    def __init__(
        self,
        boundary_resolution: int = 100,
        smooth_factor: Optional[float] = 1.0,
        mesh_max_area: float = 100.0,
        mesh_min_angle: float = 30.0
    ):
        """
        初始化网格生成器。

        Args:
        boundary_resolution (int): [M2] 在平滑边界上重新采样的顶点数量。
        smooth_factor (Optional[float]): [M2] B样条插值的平滑因子 's'。
                                        值越大，曲线越平滑但可能偏离原始轮廓。
                                        'None' 将让scipy自动选择 (m - sqrt(2*m))。
                                        通常 0.1 - 5.0 是一个合理的调参范围。
        mesh_max_area (float): [M3] 'triangle'库的 'a' 参数，控制最大三角形面积（网格密度）。
                               值越小，网格越精细。
        mesh_min_angle (float): [M3] 'triangle'库的 'q' 参数，控制最小三角形角度（网格质量）。
        """
        self.boundary_resolution = boundary_resolution
        self.smooth_factor = smooth_factor
        self.mesh_max_area = mesh_max_area
        self.mesh_min_angle = mesh_min_angle

        self.V: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.E: Optional[np.ndarray] = None
        self.kdtree: Optional[cKDTree] = None

        self.L_avg: Optional[float] = None
        self.F_centroids: Optional[np.ndarray] = None
        self.centroid_kdtree: Optional[cKDTree] = None

    # --- [模块 1] 轮廓提取 ---
    def _extract_raw_contour(self, mask_image: np.ndarray) -> np.ndarray:
        """
        从Mask图像中提取最外层的原始像素轮廓。
        """
        # 确保输入是 uint8
        if mask_image.dtype != np.uint8:
            mask_image = mask_image.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_image, 
            cv2.RETR_EXTERNAL,      # 仅提取最外层轮廓
            cv2.CHAIN_APPROX_NONE   # 获取轮廓上所有点
        )

        if not contours:
            raise ValueError("在Mask中未找到轮廓。")

        # 找到最大的轮廓
        raw_contour = max(contours, key=len)
        
        # 将形状从 (N, 1, 2) 转换为 (N, 2)
        raw_points = raw_contour.reshape(-1, 2).astype(np.float32)
        
        if len(raw_points) < 4:
            raise ValueError("轮廓点太少(少于4个)，无法进行样条插值。")
            
        return raw_points

    # --- [模块 2] 轮廓平滑与重采样 ---
    def _smooth_and_resample_contour(self, raw_points: np.ndarray) -> np.ndarray:
        """
        使用B样条插值平滑轮廓，并按指定分辨率重新采样。
        """
        # Scipy splprep 需要 (dim, N) 格式，所以我们转置
        points_t = raw_points.T
        
        # k=3: 三次样条
        # per=True: 周期性（闭合曲线）
        # s=smooth_factor: 平滑因子
        tck, u = splprep(
            [points_t[0], points_t[1]], 
            s=self.smooth_factor, 
            k=3, 
            per=True
        )
        
        # 在 0 到 1 之间均匀采样 'boundary_resolution' 个点
        u_new = np.linspace(u.min(), u.max(), self.boundary_resolution, endpoint=False)
        
        # 在新采样点上评估样条，得到平滑后的坐标
        x_new, y_new = splev(u_new, tck)
        
        # 将平滑后的点组合成 (M, 2) 数组
        smoothed_contour = np.vstack((x_new, y_new)).T
        return smoothed_contour

    # --- [模块 3] 2D网格生成 ---
    def _generate_triangle_mesh(self, smoothed_contour: np.ndarray) -> Dict[str, Any]:
        """
        使用 'triangle' 库从平滑轮廓生成2D网格。
        """
        # 为 'triangle' 准备输入字典 (PSLG - 平面直线图)
        mesh_input: Dict[str, Any] = {}
        
        # 1. 定义顶点
        mesh_input['vertices'] = smoothed_contour
        
        # 2. 定义边界 (首尾相连的线段)
        # segments = [[0, 1], [1, 2], ..., [M-1, 0]]
        M = len(smoothed_contour)
        segments = np.array([[i, (i + 1) % M] for i in range(M)])
        mesh_input['segments'] = segments
        
        # 3. 定义网格化选项
        # p: 输入是 PSLG
        # q: 质量约束 (最小角度)
        # a: 面积约束 (最大面积)
        opts = f"pq{self.mesh_min_angle}a{self.mesh_max_area}"
        
        try:
            mesh_data = triangle.triangulate(mesh_input, opts)
            return mesh_data
        except Exception as e:
            print(f"Triangle 网格化失败: {e}")
            print("这通常发生在边界自相交或 'mesh_max_area' 相对于边界过大时。")
            raise

    # --- [模块 4] 生成网格的流程 ---
    def generate_mesh(self, mask_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行从Mask到Mesh的完整流程。

        Args:
            mask_image (np.ndarray): 输入的2D二进制Mask (H, W)，dtype=np.uint8。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                V (np.ndarray): 顶点坐标数组 (V_num, 2)。
                F (np.ndarray): 三角形面片索引数组 (F_num, 3)。
        """
        # 模块 1: 提取
        print("[M1] 正在提取轮廓...")
        raw_points = self._extract_raw_contour(mask_image)
        print(f"     ... 提取了 {len(raw_points)} 个原始轮廓点。")
        
        # 模块 2: 平滑
        print(f"[M2] 正在平滑并重采样到 {self.boundary_resolution} 个点...")
        smoothed_contour = self._smooth_and_resample_contour(raw_points)
        self.smoothed_contour = smoothed_contour
        
        # 模块 3: 网格化
        print(f"[M3] 正在生成网格 (Max Area={self.mesh_max_area}, Min Angle={self.mesh_min_angle})...")
        mesh_data = self._generate_triangle_mesh(smoothed_contour)
        
        # 模块 4: 数据提取
        V = mesh_data['vertices']   # 形状 (V_num, 2)
        F = mesh_data['triangles']  # 形状 (F_num, 3)
        E = extract_edge_from_face(F)  # 形状 (E_num, 2)

        self.V = V
        self.F = F
        self.E = E

        self.L_avg = self._calculate_avg_edge_length()

        return V, F

    def generate_mesh_tree(self):
        """ 为mesh数据生成KD-tree """
        print("[M4] 正在为节点 V 构建 KD-Tree...")
        if self.V is not None and len(self.V) > 0:
            self.kdtree = cKDTree(self.V)
            print(f"     ... KD-Tree 构建完毕 (基于 {len(self.V)} 个节点)。")
        else:
            print("     ... 警告: 未生成有效节点，KD-Tree 未构建。")

    def build_centroid_kdtree(self) -> cKDTree:
        """
        计算所有三角形的质心，并构建一个基于质心的KD-tree
        """
        if self.V is None or self.F is None:
            raise RuntimeError("必须先调用 generate_mesh()。")
        
        # 1. 获取所有三角形的顶点 (M, 3, 2)
        tris = self.V[self.F]
        
        # 2. 计算质心 (M, 2)
        self.F_centroids = np.mean(tris, axis=1)
        
        # 3. 构建 KD-Tree
        self.centroid_kdtree = cKDTree(self.F_centroids)
        print(f"     ... 质心 KD-Tree 构建完毕 (基于 {len(self.F)} 个三角形)。")
        return self.centroid_kdtree

    def find_nearest_nodes(self, query_coords: np.ndarray, k: int = 1) -> Optional[np.ndarray]:
        """
        在已生成的网格中，为给定的 'query_coords' 查找 k 个最近的节点索引。

        Args:
            query_coords (np.ndarray): (Num_Points, 2) 形状的数组，包含查询点的(x, y)坐标。
                                    也可以是 (2,) 形状的单个坐标。
            k (int): 每个查询点要查找的最近邻居的数量。

        Returns:
            np.ndarray:
                如果 k=1, 返回 (Num_Points,) 形状的索引数组。(如果输入是单个坐标，Num_Points=1)
                如果 k>1, 返回 (Num_Points, k) 形状的索引数组。
            Optional[None]:
                如果网格或KD-Tree尚未生成 (必须先调用 'generate_mesh()')。
        """
        self.generate_mesh_tree()   # 生成KD-tree

        if self.kdtree is None or self.V is None:
            print("错误: 必须先调用 'generate_mesh()' 来生成网格和KD-Tree。")
            return None
        print(f"\n[QUERY] 正在为 {len(query_coords)} 个查询坐标查找 {k} 个最近邻...")
        
        # 确保 query_coords 至少是 2D 的，以便 cKDTree.query 能一致处理
        coords_2d = np.atleast_2d(query_coords)
        
        # 2. 查询 KD-Tree
        # query() 会返回距离 (distances) 和索引 (indices)
        distances, indices = self.kdtree.query(coords_2d, k=k)
        
        # 3. 优化返回格式
        # 如果原始输入是 1D (单个坐标)，我们只返回该坐标的结果
        if query_coords.ndim == 1:
            return indices[0] # 返回 (k,) 数组 (如果k=1, 则是单个数字)
        else:
            return indices # 返回 (Num_Points, k) 数组

    def map_points_to_mesh(self, query_points: np.ndarray, k_neighbors: int = 10) -> List[Optional[Dict[str, Any]]]:
        """
        将 M 个查询点映射到网格上，找到它们的barycentric coordinates in the triangular mesh

        Args:
            query_points (np.ndarray): (M, 2) 数组的查询点。
            k_neighbors (int): 查找的候选三角形数量 (安全余量)。

        Returns:
            List[Optional[Dict[str, Any]]]: 长度为 M 的列表。
                每一项是: {'tri_index': int, 'v_indices': (3,) ndarray, 'b_coords': (3,) ndarray}
                或 None (如果未找到)。
        """
        self.build_centroid_kdtree() # 确保质心KD-tree已构建

        if self.centroid_kdtree is None:
            raise RuntimeError("质心 KD-Tree 未构建。请先运行 generate_mesh()。")
        
        query_points_2d = np.atleast_2d(query_points)
        M = len(query_points_2d)
        
        # 1. 查询 KD-Tree
        distances, candidate_tri_indices = self.centroid_kdtree.query(
            query_points_2d, k=k_neighbors
        )
        
        # 确保 candidate_tri_indices 总是 2D (处理 k=1 或 M=1 的边缘情况)
        if M == 1 and k_neighbors == 1:
             candidate_tri_indices = np.array([[candidate_tri_indices]])
        elif k_neighbors == 1:
            candidate_tri_indices = candidate_tri_indices.reshape(-1, 1)
        
        mappings = []
        
        # 2. 遍历 M 个查询点
        for i in range(M):
            P = query_points_2d[i]
            found = False
            
            # 3. 遍历 k 个候选三角形
            for tri_index in candidate_tri_indices[i]:
                v_indices = self.F[tri_index]
                A, B, C = self.V[v_indices]
                
                b_coords = self._calculate_barycentric(P, A, B, C)
                
                # 4. 检查是否在三角形内 (允许数值误差)
                if np.all((b_coords >= -1e-6) & (b_coords <= 1.0 + 1e-6)):
                    mappings.append({
                        'tri_index': tri_index,
                        'v_indices': v_indices,
                        'b_coords': b_coords
                    })
                    found = True
                    break # 找到了，停止搜索 k 邻居
            
            if not found:
                print(f"警告: 点 {i} 坐标 {P} 在 {k_neighbors} 个候选内未找到...")
                mappings.append(None)
        
        return mappings

    def build_point_jacobian(self, mappings: List[Optional[Dict[str, Any]]]) -> np.ndarray:
        """
        构建 M 个特征点关于 N 个网格节点坐标的雅可比矩阵。
        
        雅可比 J = d(P_deformed) / d(V_deformed)

        Args:
            mappings (List): 来自 map_points_to_mesh() 的输出。

        Returns:
            np.ndarray: 稠密雅可比矩阵 J，形状为 (2*M, 2*N)。
        """
        if self.V is None:
            raise RuntimeError("V is None. 请先运行 generate_mesh()。")
            
        N = len(self.V) # N = 网格节点数
        M = len(mappings) # M = 特征点数
        
        J = np.zeros((2 * M, 2 * N))
        
        # 2. 遍历 M 个特征点
        for i in range(M):
            mapping = mappings[i]
            
            # 3. 如果点无效 (未找到)，则该 2 行保持为 0
            if mapping is None:
                continue
                
            v_indices = mapping['v_indices'] # (vA, vB, vC)
            b_coords = mapping['b_coords']   # (wA, wB, wC)
            
            # 4. 确定 J 中的行 (2*i) 和 (2*i + 1)
            row_start = 2 * i
            
            # 5. 遍历该点依赖的 3 个节点
            for j in range(3):
                v_idx = v_indices[j]  # e.g., vA
                w = b_coords[j]       # e.g., wA
                
                # 确定 J 中的列
                col_start = 2 * v_idx
                
                # 6. 填充 2x2 块: d(P_i) / d(V_v_idx) = w * I
                J[row_start : row_start+2, col_start : col_start+2] = w * np.eye(2)
                # print(f"填充 J 的位置: ({row_start}:{row_start+2}, {col_start}:{col_start+2}); 权重: {w:.4f}")
        return J

    def define_probes(self, p_start: np.ndarray, p_end: np.ndarray, probe_length_factor: float = 2.0) -> np.ndarray:
        """
        与任务相关的函数（最好还是剥离转换为继承关系），根据缝合线和探针长度因子，计算三个探针（共6个）的端点。
        probe超过网格边界的情况未考虑。
        Args:
            p_start (np.ndarray): 缝合线起点 (2,)
            p_end (np.ndarray): 缝合线终点 (2,)
            probe_length_factor (float): 探针总长度 = L_avg * probe_length_factor

        Returns:
            np.ndarray: (6, 2) 数组。
                        [0:3] 是3个 "start" 点 (C1-eps*n, C2-eps*n, C3-eps*n)
                        [3:6] 是3个 "end" 点   (C1+eps*n, C2+eps*n, C3+eps*n)
        """
        if self.L_avg is None:
            raise RuntimeError("L_avg 未计算。是否已运行 generate_mesh()?")
        
        probe_length = self.L_avg * probe_length_factor
        epsilon = probe_length / 2.0 # 探针半长
        
        p_start = np.asarray(p_start)
        p_end = np.asarray(p_end)
        
        # 1. 中心点 (3, 2)
        centers = np.vstack([
            p_start,
            (p_start + p_end) / 2.0,
            p_end
        ])
        
        # 2. 法向量
        s_vec = p_end - p_start
        t_vec_norm = np.linalg.norm(s_vec)
        if t_vec_norm < 1e-9:
            raise ValueError("缝合线起点和终点重合。")
        
        t_vec = s_vec / t_vec_norm # 单位切向量
        n_vec = np.array([-t_vec[1], t_vec[0]]) # 单位法向量
        
        # 3. 广播计算6个点
        probe_starts = centers - epsilon * n_vec
        probe_ends = centers + epsilon * n_vec
        
        # 4. 依次堆叠 (6, 2)
        all_probe_points = np.empty((probe_starts.shape[0]+probe_ends.shape[0], 2),
                                     dtype=probe_starts.dtype)
        all_probe_points[0::2] = probe_starts
        all_probe_points[1::2] = probe_ends
        
        return all_probe_points

    def _calculate_avg_edge_length(self) -> float:
        """
        根据 self.E 计算平均边长并存储到 self.L_avg
        """
        if self.V is None or self.E is None:
            raise RuntimeError("必须先调用 generate_mesh() 且成功生成 V 和 F。")

        P1 = self.V[self.E[:, 0]]
        P2 = self.V[self.E[:, 1]]
        
        edge_lengths = np.linalg.norm(P1 - P2, axis=1)
        
        L_avg = np.mean(edge_lengths)
        print(f"     ... 平均边长 L_avg: {L_avg:.4f}")
        return L_avg

    def _calculate_barycentric(self, P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        计算点 P 相对于三角形 ABC 的重心坐标
        (P, A, B, C 都是 (2,) 数组)
        """
        v0 = B - A
        v1 = C - A
        v2 = P - A
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        
        if np.abs(denom) < 1e-10: # 三角形退化
            return np.array([-1.0, -1.0, -1.0]) # 返回无效值
            
        wB = (d11 * d20 - d01 * d21) / denom
        wC = (d00 * d21 - d01 * d20) / denom
        wA = 1.0 - wB - wC
        
        return np.array([wA, wB, wC])


if __name__ == "__main__":
    # 1. 创建一个简单的圆形Mask作为示例输入
    H, W = 512, 512
    center = (W // 2, H // 2)
    radius = 150
    
    mask = np.zeros((H, W), dtype=np.uint8)
    # cv2.circle(mask, center, radius, 255, -1) # 255表示目标, -1表示填充
    cv2.rectangle(mask, (100, 100), (400, 150), 1, -1) # 在圆内挖个洞

    fixed_points = np.array([[150, 100],
                             [150, 125],
                             [150, 150]], dtype=np.float64)

    # 2. 实例化网格生成器并配置参数
    mesher = MaskTo2DMesh(
        boundary_resolution=100,   # 边界采样点数
        smooth_factor=1.0,         # 平滑度 (可调)
        mesh_max_area=50.0,        # 网格密度 (越小越密)
        mesh_min_angle=25.0        # 网格质量 (建议 20-34)
    )
    
    # 3. 调用主函数
    try:
        V, F = mesher.generate_mesh(mask)
        fixed_nodes_idx = mesher.find_nearest_nodes(fixed_points, k=3)
        fixed_nodes_pos = V[fixed_nodes_idx.flatten()]

        print(f"成功生成网格！")
        print(f"  顶点 (V) 数量: {V.shape[0]}")
        print(f"  面片 (F) 数量: {F.shape[0]}")
        print(f"  V 的形状: {V.shape}")
        print(f"  F 的形状: {F.shape}")

        write_mshv2_triangular("output_mesh.msh", np.array(V), np.array(F))
        print("网格已保存为 'output_mesh.msh'。")

        # 处理缝合线输入
        suture_start = np.array([300.0, 120.0])
        suture_end = np.array([280.0, 140.0])
        probe_points = mesher.define_probes(suture_start, suture_end, probe_length_factor=2.0)
        print(f"定义的探针端点:\n{probe_points}")

        probe_points_mapping = mesher.map_points_to_mesh(probe_points, k_neighbors=10) # 映射到网格的坐标
        # print(f"映射到网格的探针端点信息:\n{probe_points_mapping}")

        mapping_points_indices = [item["v_indices"] for item in probe_points_mapping]
        mapping_points_indices_flatten = [item for sublist in mapping_points_indices for item in sublist]  # 展平
        mapping_points = V[mapping_points_indices_flatten]
        # print(f"映射到网格的探针端点索引:\n{mapping_points_indices}")

        mapping_coordinates = [item["b_coords"] for item in probe_points_mapping]
        # print(f"映射到网格的探针端点重心坐标:\n{mapping_coordinates}")

        # 4. (可选) 可视化结果
        try:
            import matplotlib.pyplot as plt

            unique_fixed_nodes = np.unique(fixed_nodes_pos)     # 去重逻辑有点问题，不过不重要

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
            plt.plot(fixed_nodes_pos[:, 0], fixed_nodes_pos[:, 1], 
                     'g.', markersize=6, label='Fixed Nodes (Found)')
            # plt.scatter(mesher.smoothed_contour[:, 0], mesher.smoothed_contour[:, 1], 
            #             color='red', s=5, label='Smoothed Contour')
            plt.plot([suture_start[0], suture_end[0]], 
                     [suture_start[1], suture_end[1]], 
                     color='orange', linewidth=2, label='Suture Line')
            plt.plot(probe_points[:, 0], probe_points[:, 1], 
                     'y*', markersize=6, label='Probe Points')
            plt.plot(mapping_points[:, 0], mapping_points[:, 1], 
                     'g.', markersize=6, label='Mapped Points')
            plt.title(f'Output: 2D Triangle Mesh\n(V={V.shape[0]}, F={F.shape[0]})')
            # plt.legend()
            plt.gca().set_aspect('equal')
            plt.gca().invert_yaxis()  # 匹配图像坐标系 (Y轴向下)
            
            plt.tight_layout()
            plt.savefig("mesh_visualization.svg", dpi=300)
            # plt.show()

        except ImportError:
            print("\n(可选) 请安装 'matplotlib' 来可视化结果: pip install matplotlib")

    except ValueError as e:
        print(f"处理失败: {e}")