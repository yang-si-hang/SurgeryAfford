"""
estimate stiffness field by using EKF
Changelog:
- 2026-01-14: Batch update implementation for EKF

Date: 2026-1-7
"""
import numpy as np
import taichi as ti
import torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
from pathlib import Path
from typing import List, Optional, Dict

from const import ROOT_DIR, MESH_DIR, OUTPUT_DIR, DATA_DIR, VISUALIZATION_DIR
from deformation_model.diffpd_2d import Soft2D
from deformation_model.pd_data_loader import HDF5PdDataset


@ti.data_oriented
class Soft2DForce(Soft2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dforce_dw = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.ELEMENT_N))  # gradient of internal force wrt. stretch weight
        self.dforce_dq = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.PARTICLE_N*2))  # tangent stiffness matrix

    @ti.kernel
    def cal_deformation_gradient(self):
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos[idx1], self.node_pos[idx2], self.node_pos[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = F_i

            U, sig, V = ti.svd(F_i, ti.f64)
            self.ele_u[f_i] = U
            self.ele_v[f_i] = V
            self.stretch_stress[f_i] = ti.Vector([sig[0,0], sig[1,1]], dt=ti.f64)
            self.Bp_shear[f_i] = U @ ti.Matrix([[1., 0], [0., 1]], ti.f64) @ V.transpose()

    @ti.kernel
    def cal_internal_force_gradient(self):
        """ compute the gradient of internal force wrt. stretch weights """
        self.dforce_dw.fill(0.)
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]

            self.dforce_dw[2*idx1, f_i]   += self.dforce_dw_mat[f_i][0, 0] * self.ele_volume[f_i]
            self.dforce_dw[2*idx1+1, f_i] += self.dforce_dw_mat[f_i][0, 1] * self.ele_volume[f_i]
            self.dforce_dw[2*idx2, f_i]   += self.dforce_dw_mat[f_i][1, 0] * self.ele_volume[f_i]
            self.dforce_dw[2*idx2+1, f_i] += self.dforce_dw_mat[f_i][1, 1] * self.ele_volume[f_i]
            self.dforce_dw[2*idx3, f_i]   += self.dforce_dw_mat[f_i][2, 0] * self.ele_volume[f_i]
            self.dforce_dw[2*idx3+1, f_i] += self.dforce_dw_mat[f_i][2, 1] * self.ele_volume[f_i]

    @ti.kernel
    def cal_internal_force_gradient_pos(self):
        """ compute the tangent stiffness matrix: gradient of internal force wrt. node positions """
        self.dforce_dq.fill(0.)

        # \partial force / \partial q = - \nabla W(q) 
        for i, j in self.dforce_dq:
            r = i // 2
            c = j // 2

            is_diag = i % 2 == j % 2
            # 将布尔值转为浮点数 (1.0 或 0.0)
            self.dforce_dq[i, j] = -(self.lhs_stretch[r, c] * ti.cast(is_diag, ti.f64) - self.dBp_stretch[i, j])

    @ti.kernel
    def reconstruct_stretch_weight(self, stretch_weight:ti.types.ndarray()):
        for f_i in range(self.ELEMENT_N):
            self.stretch_weight[f_i] = stretch_weight[f_i] * self.ele_volume[f_i]

class StiffnessEKF:
    def __init__(self, 
                 num_elements: int, 
                 initial_stiffness: torch.tensor, 
                 observe_nodes: List[int],
                 device: str="cuda",
                 p_init_var: float = 1.0, 
                 q_process_noise: float = 1e-4, 
                 q_inflation_factor: float = 100.0,
                 sigma_q: float = 1e-4, 
                 sigma_f: float = 1e-2):
        """
        基于扩展卡尔曼滤波(EKF)的刚度场辨识器

        Args:
            num_elements (int): 单元(Element)数量 m
            initial_stiffness (torch.tensor): 初始刚度猜测向量, shape (m,)
            observe_dofs (List[int]): 观测节点的自由度索引列表
            p_init_var (float): 初始估计协方差 P 的对角线方差值
            q_process_noise (float): 基础过程噪声 Q 的方差 (静态阶段)
            q_inflation_factor (float): 切割发生时，局部过程噪声的膨胀系数
            sigma_q (float): 视觉/位置观测噪声方差 (用于计算 R)
            sigma_f (float): 力传感器观测噪声方差 (用于计算 R)
        """
        self.device = device
        self.ELE_NUM = num_elements
        self.obs_nodes = torch.tensor(observe_nodes, device=device, dtype=torch.long)
        self.obs_dofs = torch.stack([
            torch.tensor(observe_nodes, device=device, dtype=torch.long) * 2,
            torch.tensor(observe_nodes, device=device, dtype=torch.long) * 2 + 1
        ], dim=-1).flatten()
        self.OBS_NUM = len(observe_nodes)

        # 1. State Vector: \hat{K} (Current Belief)
        self.k_hat = initial_stiffness.to(device, dtype=torch.float64)

        # 2. Covariance Matrix: P
        self.P = torch.eye(self.ELE_NUM, device=device, dtype=torch.float64) * p_init_var

        # 3. Noise Parameters
        self.q_base_val = q_process_noise
        self.q_inflation = q_inflation_factor
        
        # 观测噪声协方差矩阵的基底
        # Sigma_q: 视觉位置噪声 covariance matrix (3n x 3n -> simplified to diagonal)
        self.sigma_q_val = sigma_q
        # Sigma_f: 力传感器噪声 covariance matrix
        self.sigma_f_val = sigma_f
        self.Sigma_f_mat = torch.zeros(self.OBS_NUM*2, self.OBS_NUM*2, device=device, dtype=torch.float64)

    def predict(self, cut_element_indices: List[int] = []):
        """
        EKF 预测步 (Prediction Step)
        对应论文 Eq (1) 和 Eq (4)
        
        Args:
            cut_element_indices: 当前时间步发生切割/拓扑改变的单元索引列表
        """
        # --- Eq (1): Motion Model & Eq (4): Prediction ---
        
        # 1. 状态预测: \hat{K}_{k|k-1}
        # 如果有切割，模拟物理上的刚度下降到0
        if len(cut_element_indices) > 0:
            self.k_hat[cut_element_indices] = 1.e-4  # 刚度不能为负，设为极小值

        # 2. 协方差预测: P_{k|k-1} = P_{k-1|k-1} + Q(k)
        # 构建自适应 Q 矩阵
        Q_k_diag = torch.full((self.ELE_NUM,), self.q_base_val, device=self.device)
        np.eye(self.ELE_NUM) * self.q_base_val
        
        if len(cut_element_indices) > 0:
            # 论文策略: "noise covariance in the local modified area is inflated"
            Q_k_diag[cut_element_indices] *= self.q_inflation
        
        # P = P + diag(Q_k)
        self.P.diagonal().add_(Q_k_diag)

    def update(self, dforce_dw:torch.Tensor, dforce_dq:torch.Tensor, 
               internal_force:torch.Tensor, measured_f_ext:torch.Tensor,
               contact_idx:int):
        """
        EKF 更新步 (Update Step)
        对应论文 Eq (2), Eq (5), Eq (6)

        Args:
            dforce_dw: 内力对刚度的雅可比矩阵 A(k), shape (2*n, m)
            dforce_dq: 内力对节点位置的雅可比矩阵 J_f, shape (2*n, 2*n)
            internal_force: 预测的内力向量 f_int, shape (2*n,)
            measured_f_ext: 外部测量的力向量 (来自传感器), shape (n_obs,)
        """
        # --- 1. 获取雅可比矩阵 ---
        
        # A_k = d(f_int) / d(K)
        # full_dforce_dw = soft_model.dforce_dw.to_numpy() # Shape: [Total_DOFs, m]
        A_k = dforce_dw[self.obs_dofs, :]           # Shape: [n_obs, m]

        # J_f = d(f_int) / d(q) (Tangent Stiffness Matrix)
        J_f_obs = dforce_dq[self.obs_dofs][:, self.obs_dofs]  # Shape: [n_obs, n_obs]

        # --- 2. 计算等效观测噪声协方差 R(k) (Eq 6) ---
        
        # R = J_f * Sigma_q * J_f^T + Sigma_f
        # 实际工程中常简化 R 为常数矩阵，即 R \approx Sigma_f
        
        # 完整写法 (假设 J_f_obs 近似代表主要影响):
        Sigma_q_mat = self.sigma_q_val * torch.eye(self.OBS_NUM*2, device=self.device, dtype=torch.float64)
        sigma_f_full = torch.zeros(NODE_NUM*2, device=self.device, dtype=torch.float64)
        sigma_f_full[contact_idx] = self.sigma_f_val  # 仅在接触点有力传感器噪声
        self.Sigma_f_mat = torch.diag(sigma_f_full[self.obs_dofs])

        term1 = J_f_obs @ Sigma_q_mat @ J_f_obs.T
        R_k = term1 + self.Sigma_f_mat
        # np.savetxt(Path(OUTPUT_DIR) / "R_k_matrix.csv", R_k.cpu().numpy(), delimiter=",")

        # --- 3. 计算卡尔曼增益 K (Eq 6) ---
        
        # S = A P A^T + R
        # Shape: [n_obs, m] @ [m, m] @ [m, n_obs] -> [n_obs, n_obs]
        P_At = self.P @ A_k.T  # Shape: [m, n_obs]
        S = A_k @ P_At + R_k
        
        # K = P A^T S^{-1}
        # 为了数值稳定性，使用 solve 代替 inv: K = (S^{-1} (P A^T)^T)^T
        # K_gain = P_At @ np.linalg.inv(S) 
        K_gain = torch.linalg.solve(S.T, P_At.T).T  # Shape: [m, n_obs]

        # --- 4. 状态更新 (Eq 5) ---
        
        # 计算 Force Residual (Innovation)        
        f_int_pred = internal_force.flatten()[self.obs_dofs]  # 预测的内力 (观测点)
        innovation = measured_f_ext.flatten() - f_int_pred # (z - h(x))
        print(f"residual: {torch.norm(innovation)}")  # Debug: 打印创新项
        print(f"Kalman gained residual norm: {torch.norm(K_gain @ innovation)}")  # Debug: 打印增益后的创新项
        
        self.k_hat = self.k_hat + K_gain @ innovation

        # 物理约束：刚度必须非负
        self.k_hat = torch.maximum(self.k_hat, torch.tensor(1e-6, device=self.device))

        # # P = (I - K A) P
        # # 推荐使用 Joseph form 以保持对称性和正定性: P = (I-KA)P(I-KA)^T + KRK^T，但在论文中是简化形式
        # I = torch.eye(self.ELE_NUM, device=self.device)
        # self.P = (I - K_gain @ A_k) @ self.P

        # --- 5. 协方差更新 (Joseph Form) ---
        
        # 预计算 (I - K_gain @ A_k)
        I = torch.eye(self.ELE_NUM, device=self.device)
        ImKA = I - K_gain @ A_k
        
        # 第一部分: (I - KA) @ P @ (I - KA).T
        # 第二部分: K @ R_k @ K.T
        self.P = ImKA @ self.P @ ImKA.T + K_gain @ R_k @ K_gain.T
        
        # 进一步确保对称性 (数值技巧)
        # 即使使用 Joseph Form，极小的浮点误差也可能导致 A != A.T
        self.P = 0.5 * (self.P + self.P.T)

    def batch_update(self, 
                     dforce_dw_list: List[torch.Tensor], 
                     dforce_dq_list: List[torch.Tensor], 
                     internal_force_list: List[torch.Tensor], 
                     measured_f_ext_list: List[torch.Tensor],
                     contact_idx_list: List[int]):
        """
        批量 EKF 更新 (基于信息滤波逻辑)
        
        Args:
            dforce_dw_list: 长度为 N 的列表，每个元素 shape (2*n, m)
            dforce_dq_list: 长度为 N 的列表，每个元素 shape (2*n, 2*n)
            internal_force_list: 长度为 N 的列表，每个元素 shape (2*n,)
            measured_f_ext_list: 长度为 N 的列表，每个元素 shape (n_obs,)
        """
        # 1. 初始化信息增量 (Hessian 和 Gradient 贡献)
        m = self.ELE_NUM
        delta_Y = torch.zeros((m, m), device=self.device, dtype=torch.float64)
        delta_b = torch.zeros((m,), device=self.device, dtype=torch.float64)

        accumulated_strain_metric = torch.zeros(m, device=self.device, dtype=torch.float64)

        # 2. 遍历 Batch，累加每个观测点的贡献
        for i in range(len(dforce_dw_list)):
            # --- 提取当前样本数据 ---
            A_i = dforce_dw_list[i][self.obs_dofs, :] * self.k_hat.repeat(len(self.obs_dofs), 1)  # (n_obs, m)
            J_f_i = dforce_dq_list[i][self.obs_dofs][:, self.obs_dofs] # (n_obs, n_obs)
            f_int_pred = internal_force_list[i].flatten()[self.obs_dofs]
            f_meas = measured_f_ext_list[i].flatten()
            contact_idx = contact_idx_list[i]

            col_norms = torch.linalg.norm(dforce_dw_list[i][self.obs_dofs, :], dim=0) 
            accumulated_strain_metric += col_norms
            
            # --- 计算当前样本的等效观测噪声 R_i ---
            Sigma_q_mat = self.sigma_q_val * torch.eye(self.OBS_NUM*2, device=self.device, dtype=torch.float64)
            
            # 构造力传感器噪声 (仅在接触点)
            sigma_f_full = torch.zeros(NODE_NUM*2, device=self.device, dtype=torch.float64)
            sigma_f_full[contact_idx] = self.sigma_f_val  # 仅在接触点有力传感器噪声
            Sigma_f_mat = torch.diag(sigma_f_full[self.obs_dofs])

            R_k_i = J_f_i @ Sigma_q_mat @ J_f_i.T + Sigma_f_mat


            # --- 计算信息增量 ---
            # R_inv = torch.inverse(R_k_i) # n_obs 较小时直接求逆
            # 为了稳定性，使用 solve: A^T * R^-1 * A -> A^T * solve(R, A)
            R_inv_Ai = torch.linalg.solve(R_k_i, A_i) # (n_obs, m)
            
            # 更新 Hessian 贡献: A^T * R^-1 * A
            delta_Y += A_i.T @ R_inv_Ai
            
            # 更新 Gradient 贡献: A^T * R^-1 * (z - h)
            innovation = f_meas - f_int_pred
            delta_b += A_i.T @ torch.linalg.solve(R_k_i, innovation)
            # delta_b += A_i.T @ innovation  # 修正项

        # 3. 融合先验信息与观测信息 (Information Fusion)
        # P_post = (P_pre^-1 + delta_Y)^-1
        # 这种写法避免了多次 K 增益计算，统一进行一次大矩阵求逆
        P_inv_pre = torch.inverse(self.P)

        # 空间平滑增量
        lambda_s = 0.e-8 # 调节平滑强度的超参数
        delta_Y_reg = lambda_s * self.L
        delta_b_reg = -lambda_s * (self.L @ self.k_hat)

        P_inv_post = P_inv_pre + delta_Y + delta_Y_reg

        self.delta_Y = delta_Y
        
        # 更新协方差
        self.P = torch.inverse(P_inv_post)
        
        # 4. 状态更新: k_post = k_pre + P_post * delta_b
        # self.k_hat = self.k_hat + self.P @ (delta_b + delta_b_reg)
        self.k_hat = self.k_hat * torch.exp(self.P @ delta_b)

        # 5. 物理约束与数值对称化
        # self.k_hat = torch.maximum(self.k_hat, torch.tensor(1e-6, device=self.device))
        self.P = 0.5 * (self.P + self.P.T)

        avg_strain_factor = accumulated_strain_metric / len(dforce_dw_list)
        avg_internal_force_modulus = torch.abs(self.k_hat) * avg_strain_factor

        self.stress_metric = avg_internal_force_modulus

    def get_stiffness(self):
        return self.k_hat

def construct_laplacian_matrix(neighbors: np.ndarray) -> np.ndarray:
    num_faces = neighbors.shape[0]
    A = np.zeros((num_faces, num_faces))

    for i, row in enumerate(neighbors):
        for neighbor_idx in row:
            if neighbor_idx != -1:
                # i 和 neighbor_idx 共享一条边
                A[i, neighbor_idx] = 1

    # 2. 计算度矩阵 D (每个单元格相邻的单元格数量)
    # 沿着行求和得到每个单元的邻居数
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # 3. 构建图拉普拉斯矩阵 L
    L = D - A
    return L

if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True)

    hard_area = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
    free_area = [201, 203, 223, 227, 240, 241] + [133, 190, 197, 199, 206, 222, 259]
    
    # load dataset #
    demo_dir = DATA_DIR / "demo" / "pd_stretch_data_hete" / "20260122_094705"
    dataset = HDF5PdDataset(data_directory=str(demo_dir))
    print(f"数据集加载完成，共包含 {len(dataset)} 个样本。")

    neighbor_info = np.loadtxt(f"{MESH_DIR}/pd_stretch_demo_mesh_neighbors.csv", delimiter=",", dtype=np.int32)
    laplacian_mat = construct_laplacian_matrix(neighbor_info)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data directory.")

    MESH_DATA:dict = dataset.mesh_data
    FIXED_NODES = dataset.static_data['fix_nodes'].tolist()
    REAL_W = dataset.static_data['stiffness_truth']
    hard_ele_list = dataset.static_data['hard_ele_idx'].tolist()
    free_ele_list = dataset.static_data['free_ele_idx'].tolist()

    NODE_NUM = MESH_DATA['V'].shape[0]
    FACE_NUM = MESH_DATA['F'].shape[0]
    OBSERVE_NODES = list(set(range(NODE_NUM)) - set(FIXED_NODES))
    OBSERVE_DOFS = np.stack([np.array(OBSERVE_NODES) * 2, np.array(OBSERVE_NODES) * 2 + 1], axis=-1).flatten().tolist()

    # 使用 Dict[KeyType, ValueType]
    model_cache: Dict[int, Soft2DForce] = {}
    
    init_k_guess = torch.ones(FACE_NUM, device="cuda") * 400.0 # 初始猜测
    # init_k_guess[hard_ele_list] *= 100
    # init_k_guess[free_ele_list] *= 0.01
    
    ekf = StiffnessEKF(
        num_elements=FACE_NUM,
        initial_stiffness=init_k_guess,
        observe_nodes=OBSERVE_NODES,
        p_init_var=1.e20,
        q_process_noise=1.e3,
        sigma_q=1.e-5,
        sigma_f=1e-1,
    )

    ekf.L = torch.tensor(laplacian_mat, device="cuda", dtype=torch.float64)

    # 模拟加权最小二乘 #
    dforce_dw_list = []
    dforce_dq_list = []
    internal_force_list = []
    measured_f_ext_list = []
    contact_idx_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        contact_idx = int(sample['contact_idx'])
        measure_node_force = sample['force'].to("cuda")[OBSERVE_NODES,:]
        measure_q = sample['post_x'][:, :2].to("cuda")

        if contact_idx not in model_cache:
            # construct soft body model #
            new_model = Soft2DForce(
                shape=MESH_DATA, fix=FIXED_NODES, 
                contact=contact_idx, E=1.e1, nu=0.3, dt=1.e-2, density=1.e1, device="cuda"
            )
            model_cache[contact_idx] = new_model
            print(f"Construct new model for contact idx: {contact_idx}")

        soft_model = model_cache[contact_idx]

        cut_indices = []

        soft_model.reconstruct_stretch_weight(init_k_guess)
        soft_model.precomputation()

        # 运行物理步 (Forward & Backward)
        soft_model.node_pos.from_torch(measure_q) # 使用观测位置

        soft_model.cal_deformation_gradient()
        soft_model.update_internal_force()
        soft_model.cal_internal_force_gradient() # 计算 dforce_dw

        soft_model.hessian_stretch()
        soft_model.cal_internal_force_gradient_pos()    # 计算 dforce_dq
        
        # 4. EKF 更新
        dforce_dw_torch = soft_model.dforce_dw.to_torch(device="cuda").double()
        dforce_dq_torch = soft_model.dforce_dq.to_torch(device="cuda").double()
        internal_force_torch = soft_model.force.to_torch(device="cuda").double()

        dforce_dw_list.append(dforce_dw_torch)
        dforce_dq_list.append(dforce_dq_torch)
        internal_force_list.append(internal_force_torch)
        measured_f_ext_list.append(measure_node_force.double())
        contact_idx_list.append(contact_idx)

    ekf.batch_update(
        dforce_dw_list,
        dforce_dq_list,
        internal_force_list,
        measured_f_ext_list,
        contact_idx_list
    )

    current_k = ekf.get_stiffness()

    ### 模拟时间循环 ###
    # for i in range(len(dataset)):
    #     sample = dataset[i]
        
    #     # 获取观测数据
    #     contact_idx = int(sample['contact_idx'])
    #     measure_node_force = sample['force'].to("cuda")[OBSERVE_NODES,:]
    #     measure_q = sample['post_x'][:, :2].to("cuda")

    #     if contact_idx not in model_cache:
    #         # construct soft body model #
    #         new_model = Soft2DForce(
    #             shape=MESH_DATA, fix=FIXED_NODES, 
    #             contact=contact_idx, E=1.e1, nu=0.3, dt=1.e-2, density=1.e1, device="cuda"
    #         )
    #         # new_model.reconstruct_stretch_weight(init_w)
    #         # new_model.precomputation()

    #         model_cache[contact_idx] = new_model

    #     soft_model = model_cache[contact_idx]

    #     # 1. 检测是否发生切割 (根据 sample info 或外部逻辑)
    #     # is_cutting = check_if_cutting(sample)
    #     # cut_indices = get_cut_indices(sample) if is_cutting else []
    #     cut_indices = [] # 示例：暂无切割
        
    #     # 2. EKF 预测
    #     # 默认拓扑改变只有切割
    #     ekf.predict(cut_element_indices=cut_indices)
        
    #     # 3. 将预测的刚度注入物理模型，计算雅可比
    #     current_k_belief = ekf.get_stiffness()
    #     soft_model.reconstruct_stretch_weight(current_k_belief)
    #     soft_model.precomputation() # 理论上只用计算stretch部分，因为没有正向模拟
        
    #     # 运行物理步 (Forward & Backward)
    #     # [IMPORTANT] 确保这里的节点位置 q 是当前的观测位置，或者基于当前刚度平衡后的位置
    #     # Static Equilibrium Identification，这里需要 solve equilibrium
    #     soft_model.node_pos.from_torch(measure_q) # 使用观测位置

    #     soft_model.cal_deformation_gradient()
    #     soft_model.update_internal_force()
    #     soft_model.cal_internal_force_gradient() # 计算 dforce_dw

    #     soft_model.hessian_stretch()
    #     soft_model.cal_internal_force_gradient_pos()    # 计算 dforce_dq
        
    #     # 4. EKF 更新
    #     dforce_dw_torch = soft_model.dforce_dw.to_torch(device="cuda").double()
    #     dforce_dq_torch = soft_model.dforce_dq.to_torch(device="cuda").double()
    #     internal_force_torch = soft_model.force.to_torch(device="cuda").double()
    #     ekf.update(dforce_dw_torch, dforce_dq_torch, internal_force_torch, measure_node_force, contact_idx)
        
    #     # 5. 结果分析
    #     current_k = ekf.get_stiffness()
    #     print(f"Step {i}, Mean Stiffness: {torch.mean(current_k):.2f}")

    # 可视化最终结果 #
    # 可视化三角形网格每个单元的刚度值
    P_mat = ekf.P.cpu().numpy()
    raw_k = current_k.cpu().numpy().copy()
    P_diag = ekf.P.diagonal().cpu().numpy().copy()
    sigma_k = np.sqrt(P_diag)
    R_mat = P_mat / np.outer(sigma_k, sigma_k)
    delta_Y = ekf.delta_Y.cpu().numpy()

    np.savetxt(Path(OUTPUT_DIR) / "estimated_stiffness_ekf.csv", raw_k, delimiter=",")
    np.savetxt(Path(OUTPUT_DIR) / "P_matrix_ekf.csv", ekf.P.cpu().numpy(), delimiter=",")
    # np.savetxt(Path(OUTPUT_DIR) / "P_inv_matrix_ekf.csv", torch.linalg.inv(ekf.P).cpu().numpy(), delimiter=",")
    np.savetxt(Path(OUTPUT_DIR) / "P_diag_ekf.csv", P_diag, delimiter=",")   

    cells = [("triangle", soft_model.ele.to_numpy().astype(np.int32))]

    mesh = meshio.Mesh(
    soft_model.node_pos_init.to_numpy()[:, :2],
    cells,
    cell_data={
        "Stiffness": [raw_k],
        "Uncertainty": [P_diag],
    }
    )
    mesh.write(f"{OUTPUT_DIR}/mesh_with_stiffness.vtu")

    plt.figure(figsize=(6, 6))
    node_pos_np = soft_model.node_pos_init.to_numpy()[:, :2]  # Use post positions for deformed mesh
    triangles = soft_model.ele.to_numpy()
    triang = mtri.Triangulation(node_pos_np[:, 0], node_pos_np[:, 1], triangles)
    # for i in range(current_k.shape[0]):
    #     if current_k[i] <= 0:# or current_k[i] > 1.e3:
    #         raw_k[i] = -10

    epsilon = 1e-6
    # updated_w_show = np.log10(np.maximum(raw_k, epsilon))

    plt.tripcolor(triang, facecolors=np.log10(raw_k), cmap='RdBu_r', edgecolors='k')
    plt.colorbar(label='Stiffness')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Stiffness per Element')

    out_path_stiff = Path(VISUALIZATION_DIR) / f"stiffness_ekf.svg"
    plt.savefig(out_path_stiff, dpi=200)
    plt.close()
    print(f"Saved stiffness visualization to {out_path_stiff}")

    plt.figure(figsize=(6, 6))
    plt.tripcolor(triang, facecolors=np.log10(sigma_k), cmap='RdBu_r', edgecolors='k')  # 
    plt.colorbar(label='Stiffness Variance')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Stiffness Variance per Element')

    out_path_var = Path(VISUALIZATION_DIR) / f"stiffness_variance_ekf.svg"
    plt.savefig(out_path_var, dpi=200)
    plt.close()
    print(f"Saved stiffness variance visualization to {out_path_var}")

    # plt.hist(np.log10(raw_k), bins=50)
    # plt.xlabel("Log10 Stiffness")
    # plt.savefig(Path(VISUALIZATION_DIR) / "stiffness_log_histogram.svg", dpi=300)
    # plt.close()

    # plt.hist(np.log10(sigma_k), bins=50)
    # plt.xlabel("Log10 Stiffness Variance")
    # plt.savefig(Path(VISUALIZATION_DIR) / "variance_log_histogram.svg", dpi=300)
    # plt.close()

    # plt.figure(figsize=(6, 6))
    # plt.tripcolor(triang, facecolors=np.log10(metric_a), cmap='RdBu_r', edgecolors='k')  # 
    # plt.colorbar(label='Stiffness * Variance')
    # plt.gca().set_aspect('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'Stiffness * Variance per Element')

    # out_path_var = Path(VISUALIZATION_DIR) / f"stiffness*variance_ekf.svg"
    # plt.savefig(out_path_var, dpi=200)
    # plt.close()
    # print(f"Saved stiffness variance visualization to {out_path_var}")

    # plt.figure(figsize=(6, 6))
    # plt.tripcolor(triang, facecolors=np.log10(metric_b), cmap='RdBu_r', edgecolors='k')  #
    # plt.colorbar(label='Stiffness / Variance')
    # plt.gca().set_aspect('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'Stiffness / Variance per Element')

    # out_path_var = Path(VISUALIZATION_DIR) / f"stiffness_variance_ratio_ekf.svg"
    # plt.savefig(out_path_var, dpi=200)
    # plt.close()
    # print(f"Saved stiffness/variance ratio visualization to {out_path_var}")

    # plt.figure(figsize=(6, 6))
    # plt.tripcolor(triang, facecolors=np.diag(delta_Y), cmap='RdBu_r', edgecolors='k')
    # plt.colorbar(label='Delta Y Diagonal')
    # plt.gca().set_aspect('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'Delta Y Diagonal per Element')

    # out_path_deltay = Path(VISUALIZATION_DIR) / f"delta_Y_diagonal_ekf.svg"
    # plt.savefig(out_path_deltay, dpi=200)
    # plt.close()
    # print(f"Saved delta Y diagonal visualization to {out_path_deltay}")