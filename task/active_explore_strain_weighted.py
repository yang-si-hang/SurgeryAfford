"""
基于协方差矩阵加权的应变最大化主动探索规划器, 用划分好的刚度区域 (或者边界) 进行加权
loss = -|| S(t+1) ∑(t) ||_F = -sqrt( sum_k [ P_row_sum_k * (sigma0_k^2 + sigma1_k^2) ] )
优先选用"SLSQP"方法进行优化, 速度更快
Date: 2026-02-06
"""
import time
import taichi as ti
import meshio
import torch
torch.set_printoptions(linewidth=120)
import numpy as np
np.set_printoptions(linewidth=120)
from typing import Dict, List, Tuple
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from stiffoptim_force_ekf import Soft2DForce, StiffnessEKF
from deformation_model.pd_data_loader import HDF5PdDataset
from utilize.mesh_io import write_mshv2_triangular
from const import DATA_DIR, OUTPUT_DIR, VISUALIZATION_DIR


def compute_strain(u_vec:torch.Tensor, dF_du:torch.Tensor, device:str="cuda"):
    """
    计算约束值和 Jacobian
    Args:
        u_action: (N_act, ) 当前动作
        dF_du: (N_elem, 2, 2, N_act) 变形梯度对动作的导数张量
    """
    # 预测变形梯度 F (N_elem, 2, 2)
    F_base = torch.eye(2).repeat(dF_du.shape[0], 1, 1).to(device, dtype=torch.float64)  # 基准 F 矩阵 (单位矩阵)
    delta_F = torch.einsum('bija, a -> bij', dF_du, u_vec)
    F_pred = F_base + delta_F
    
    U, S, Vh = torch.linalg.svd(F_pred)  # S: (N_elem, 2), sigma_0 >= sigma_1

    # d(sigma)/dF = u_i * v_i^T            
    # u0 = U[:, :, 0], v0 = Vh[:, 0, :]
    dSig0_dF = torch.einsum('bi, bj -> bij', U[:, :, 0], Vh[:, 0, :])
    dSig0_du = torch.einsum('bij, bija -> ba', dSig0_dF, dF_du)
    
    dSig1_dF = torch.einsum('bi, bj -> bij', U[:, :, 1], Vh[:, 1, :])
    dSig1_du = torch.einsum('bij, bija -> ba', dSig1_dF, dF_du)
    
    return S, dSig0_du, dSig1_du


@ti.data_oriented
class Soft2DForceExtended(Soft2DForce):
    """
    扩展后的物理引擎，适配基于视觉反馈的主动感知任务
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dJ_dH_e = ti.Matrix.field(1, 6, dtype=ti.f64, shape=self.ELEMENT_N)  # per-element dJ/dH
        self.dJ_dq_tensor = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N * 2, ))  # dJ/dq tensor

        self.node2obs_map = ti.field(dtype=ti.i32, shape=self.PARTICLE_N)
        self.dof2obs_map = ti.field(dtype=ti.i32, shape=self.PARTICLE_N * 2)
        self.OBS_NUM = self._init_obs_map()

    def update_state_from_vision(self, q_vision: torch.Tensor, current_k: torch.Tensor):
        """
        从视觉传感器和当前的刚度估计更新物理系统的内部状态。
        
        Args:
            q_vision: shape (N_nodes, 2), 当前视觉观测到的节点位置
            current_k: shape (N_elements, ), 当前估计的刚度值
        """
        self.reconstruct_stretch_weight(current_k)
        self.precomputation()

        self.node_pos.from_torch(q_vision)

        # 基于当前位置，计算所有导数项
        self.cal_deformation_gradient()
        self.update_internal_force() 
        self.cal_internal_force_gradient()  # 计算 H 矩阵 (dforce_dw)

        # 计算 Hessian, 切线刚度 K_T (dforce_dq), 变形雅可比 (dq_du)
        self.construct_E_hessian()
        self.cal_internal_force_gradient_pos()

    def get_gradients_at_state(self, obs_dofs: List[int]):
        """
        获取当前状态下的雅可比矩阵, 用于计算梯度。
        
        Returns:
            H_obs: (N_obs, N_elem) 观测矩阵
            K_T: (N_total, N_total) 全局切线刚度矩阵
        """
        H_full = self.dforce_dw.to_torch(device=self.device).double() # (2*N_nodes, N_elem)
        K_T_full = self.dforce_dq.to_torch(device=self.device).double() # (2*N_nodes, 2*N_nodes)
        
        # 提取观测行
        H_obs = H_full[obs_dofs, :]
        
        return H_obs, K_T_full
    
    def compute_dJ_dq(self, dJ_dH: torch.Tensor):
        """
        计算 dJ/dq, 通过链式法则连接 dJ/dH 和 dH/dq
        Args:
            dJ_dH: shape (N_obs, N_elem) 的梯度张量
        """
        # 获取 dH/dq
        # 已经除了 stretch weight
        dBp_stretch_e = self.dBp_stretch_e.to_torch(device=self.device) # (N_elem, 6, 6)
        lhs_stretch_e = self.lhs_stretch_e.to_torch(device=self.device) # (N_elem, 3, 3)
        lhs_stretch_e_kron = torch.kron(lhs_stretch_e, torch.eye(2, device=self.device, dtype=torch.float64))
        dH_dq_e = -(lhs_stretch_e_kron - dBp_stretch_e)  # (N_elem, 6, 6)

        self.compute_ele_dJ_dH(dJ_dH)  # 计算每个单元的 dJ/dH
        dJ_dH_e = self.dJ_dH_e.to_torch(device=self.device)  # (N_elem, 1, 6)

        dJ_dq_e = dJ_dH_e @ dH_dq_e  # (N_elem, 1, 6) @ (N_elem, 6, 6) -> (N_elem, 1, 6)

        self.assemble_dJ_dq(dJ_dq_e)

        return self.dJ_dq_tensor.to_torch(device=self.device)

    @ti.kernel
    def compute_ele_dJ_dH(self, dJ_dH: ti.types.ndarray()):
        self.dJ_dH_e.fill(0.)

        for e_i in range(self.ELEMENT_N):
            nodes = self.ele[e_i]
            for k in ti.static(range(3)):
                obs_idx = self.node_to_obs_map[nodes[k]]
                if obs_idx != -1:
                    row_x = obs_idx * 2
                    row_y = obs_idx * 2 + 1

                    self.dJ_dH_e[e_i][0, k*2] = dJ_dH[row_x, e_i]
                    self.dJ_dH_e[e_i][0, k*2+1] = dJ_dH[row_y, e_i]
        
    @ti.kernel
    def assemble_dJ_dq(self, dJ_dq_e: ti.types.ndarray()):
        self.dJ_dq_tensor.fill(0.)

        for e_i in range(self.ELEMENT_N):
            for k in ti.static(range(3)):
                idx = self.ele[e_i][k]
                dof_x = idx * 2
                dof_y = idx * 2 + 1
                self.dJ_dq_tensor[dof_x] += dJ_dq_e[e_i, 0, k*2]
                self.dJ_dq_tensor[dof_y] += dJ_dq_e[e_i, 0, k*2+1]

    def compute_dH_du(self, dq_du:torch.Tensor, stiffness:torch.Tensor)->torch.Tensor:
        """ 在element-wise计算并组装, 计算 dH/du 矩阵"""
        N_act = len(self.contact_particle_list) * 2

        # 构建 (N_elem, 6) 的自由度索引矩阵
        ele_nodes = self.ele.to_torch(device=self.device) # (N_elem, 3)
        ele_dofs = torch.stack([ele_nodes * 2, ele_nodes * 2 + 1], dim=2).reshape(self.ELEMENT_N, 6).cpu()  # (N_elem, 6)

        dq_du_e = dq_du[ele_dofs]   # (N_elem, 6, N_contact_dofs), 提取每个单元对应的 dq/du 子矩阵
        # 已经除了 stretch weight
        dBp_stretch_e = self.dBp_stretch_e.to_torch(device=self.device) # (N_elem, 6, 6)
        lhs_stretch_e = self.lhs_stretch_e.to_torch(device=self.device) # (N_elem, 3, 3)
        lhs_stretch_e_kron = torch.kron(lhs_stretch_e, torch.eye(2, device=self.device, dtype=torch.float64))
        dH_dq_e = -(lhs_stretch_e_kron - dBp_stretch_e) * stiffness.unsqueeze(-1).unsqueeze(-1)  # (N_elem, 6, 6), unit siffness matrix

        dH_du_e = torch.bmm(dH_dq_e, dq_du_e)   # (N_elem, 6, N_contact_dofs)
        np.savetxt(OUTPUT_DIR / "dH_du_e.csv", dH_du_e.cpu().numpy().reshape(self.ELEMENT_N, 6 * len(self.contact_particle_list) * 2), delimiter=",")

        # === 将dH_du_e 映射到观测空间 === 
        dof2obs_map = self.dof2obs_map.to_torch(device=self.device) # (N_dofs, ), obs_dofs[k] -> k
        
        # 4.2 将单元的自由度索引转换为观测索引
        # 里面的值是 0 ~ N_obs-1，或者是 -1 (未被观测)
        ele_obs_indices = dof2obs_map[ele_dofs]     # (N_elem, 6), 以ele_dofs为索引, 查找对应的观测索引
        
        # 4.3 构建掩码 (Masking)
        valid_mask = ele_obs_indices >= 0  # (N_elem, 6)
        
        # 4.4 提取有效的索引和数值 (Flattening for advanced indexing)
        # valid_obs_idx: 属于哪个观测通道 (0 ~ N_obs-1)
        valid_obs_idx = ele_obs_indices[valid_mask]     # 被mask索引后会变成一维 (N_valid, )
        
        # valid_elem_idx: 属于哪个单元 (0 ~ N_elem-1)
        # 构造单元索引矩阵 (N_elem, 6)，值是行号
        elem_idx_grid = torch.arange(self.ELEMENT_N, device=self.device).unsqueeze(1).expand(-1, 6)
        valid_elem_idx = elem_idx_grid[valid_mask]
        
        # valid_values: 对应的数值 (N_valid, N_act)
        # dH_du_e: (N_elem, 6, N_act) -> mask -> (N_valid, N_act)
        valid_values = dH_du_e[valid_mask] 

        # 4.5 构造最终张量并赋值
        # 目标 shape: (N_obs, N_elem, N_act)
        dH_du = torch.zeros((2*self.OBS_NUM, self.ELEMENT_N, N_act), device=self.device, dtype=torch.float64)
        
        # 使用高级索引直接赋值
        # dH_du[obs_idx, elem_idx, :] = values
        # 由于 (obs_idx, elem_idx) 这一对坐标是唯一的 (一个单元的一个节点只对应一个观测点)
        # 所以不存在冲突，直接赋值即可 (不需要 scatter_add)
        dH_du[valid_obs_idx, valid_elem_idx, :] = valid_values

        return dH_du, dH_dq_e

    def compute_H(self, q_vec: torch.Tensor, k_guess: torch.Tensor) -> torch.Tensor:
        """ 根据观测的节点位置和输入的刚度, 计算观测矩阵 H = dJ/dw """
        self.reconstruct_stretch_weight(k_guess)
        self.precomputation()

        self.node_pos.from_torch(q_vec) # 使用观测位置

        self.cal_deformation_gradient()
        self.update_internal_force()
        self.cal_internal_force_gradient() # 计算 dforce_dw

        return self.dforce_dw.to_torch(device=self.device).double()
    
    def compute_dF_du(self, dq_du:torch.Tensor)->torch.Tensor:
        """ 计算 dF/du 矩阵 (变形梯度对动作的导数) """
        F_A_tensor = self.F_A.to_torch(device=self.device)  # (N_elem, 2, 3)

        ele = self.ele.to_torch(device=self.device)  # (N_elem, 3)
        dq_du_view = dq_du.view(self.PARTICLE_N, 2, -1)  # (N_nodes, 2, N_contact_dofs)
        dq_du_local = dq_du_view[ele]

        # 注意此处对F的转置, b: N_ele; j:2; n:2; a:N_contact_dofs. (i, j) 对应 F_ij
        dF_du_tensor = torch.einsum('bjn,bnia->bija', F_A_tensor, dq_du_local)  # (N_elem, 2, 2, N_contact_dofs)
        return dF_du_tensor

    def _init_obs_map(self)->int:
        """ 初始化观测节点和自由度的映射表 """
        self.node2obs_map.fill(-1)   # -1 represents unobserved
        observe_nodes = list(set(range(self.PARTICLE_N)) - set(self.fix_particle_list))
        observe_nodes.sort()
        obs_num = len(observe_nodes)

        # 准备 Numpy 数组用于构建映射
        node_map_np = torch.full((self.PARTICLE_N,), -1, dtype=torch.int32, device=self.device)
        dof_map_np = torch.full((self.PARTICLE_N * 2,), -1, dtype=torch.int32, device=self.device)
    
        # === 3. 同时填充 Node 映射和 DOF 映射 ===
        for obs_node_idx, global_node_idx in enumerate(observe_nodes):
            node_map_np[global_node_idx] = obs_node_idx
            
            global_dof_x = global_node_idx * 2
            global_dof_y = global_node_idx * 2 + 1
            
            obs_dof_x = obs_node_idx * 2
            obs_dof_y = obs_node_idx * 2 + 1
            
            dof_map_np[global_dof_x] = obs_dof_x
            dof_map_np[global_dof_y] = obs_dof_y
        
        self.node2obs_map.from_torch(node_map_np)
        self.dof2obs_map.from_torch(dof_map_np)

        return obs_num

    def _verify_dH_dq(self, dH_dq_e, ele_idx=0, epsilon=1e-5):
        """ 验证单元级别的 dH/dq 计算是否正确 """
        dH_dq_e_analytic = dH_dq_e[ele_idx].cpu() # (6, 6)
        
        # 获取该单元的节点索引
        node_indices = self.ele[ele_idx] # (3,)
        dofs_indices = torch.tensor([node_indices[0]*2, node_indices[0]*2+1,
                                     node_indices[1]*2, node_indices[1]*2+1,
                                     node_indices[2]*2, node_indices[2]*2+1], dtype=torch.long)
        
        q_current = self.node_pos_init.to_torch()
        
        # 定义一个辅助函数：计算该单元的内力
        def compute_H(q_vec):
            self.reconstruct_stretch_weight(torch.ones(self.ELEMENT_N, device="cuda") * 5000.0)
            self.precomputation()

            # 运行物理步 (Forward & Backward)
            self.node_pos.from_torch(q_vec) # 使用观测位置

            self.cal_deformation_gradient()
            self.update_internal_force()
            self.cal_internal_force_gradient() # 计算 dforce_dw

            return self.dforce_dw.to_torch(device=self.device).double()[dofs_indices, ele_idx].reshape(-1).cpu() # (6,)

        # 2. 计算数值梯度
        H0 = compute_H(q_current)  # 基准力 (6,)
        dH_dq_numeric = torch.zeros((6, 6), dtype=torch.float64)
        
        print(f"--- Verifying Element {ele_idx} ---")
        for i in range(6): # 遍历 6 个自由度
            node_idx = node_indices[i // 2]
            dim_idx = i % 2

            q_plus = q_current.clone()
            q_plus[node_idx, dim_idx] += epsilon
            
            H_plus = compute_H(q_plus)
            
            col_i = (H_plus - H0) / (epsilon)
            dH_dq_numeric[:, i] = col_i

        # 3. 对比
        diff = torch.abs(dH_dq_e_analytic - dH_dq_numeric)
        max_diff = diff.max().item()
        rel_diff = max_diff / (torch.abs(dH_dq_numeric).max().item() + 1e-8)
        
        print(f"Analytic vs Numeric Max Diff: {max_diff:.2e}")
        print(f"Relative Error: {rel_diff:.2%}")
        
        if rel_diff < 1e-3: # 通常精度要在 0.1% 以内
            print("✅ Jacobian Check Passed!")
        else:
            print("❌ Jacobian Check Failed!")
            print("Top-left block (Analytic):\n", dH_dq_e_analytic[:, :])
            print("Top-left block (Numeric):\n", dH_dq_numeric[:, :])
        

class ActiveStiffnessPlanner:
    def __init__(self, 
                 soft_model: Soft2DForceExtended, 
                 estimator: StiffnessEKF,
                 device="cuda"):
        self.soft_model = soft_model
        self.estimator = estimator
        self.device = device

    def visualize_loss_landscape(self, dF_du, P_prior, max_mag=0.03):
        """
        可视化 D-Optimality Loss 在 2D 动作空间的地形图
        """
        import matplotlib.pyplot as plt

        print("--- Visualizing Loss Landscape ---")

        P_row_sum = torch.sum(P_prior**2, dim=1)
        
        # 1. 定义网格范围 (比约束稍微大一点，看清边界外的情况)
        range_lim = max_mag * 1.5
        grid_size = 30
        x = np.linspace(-range_lim, range_lim, grid_size)
        y = np.linspace(-range_lim, range_lim, grid_size)
        X, Y = np.meshgrid(x, y)
        
        Z_loss = np.zeros_like(X)
        U_grad = np.zeros_like(X) # x方向梯度
        V_grad = np.zeros_like(X) # y方向梯度
        
        # 2. 遍历网格计算 Loss 和 梯度
        # 为了不破坏计算图，我们在 no_grad 模式下计算 Loss，临时开启 grad 计算梯度
        
        for i in range(grid_size):
            for j in range(grid_size):
                u_val = np.array([X[i, j], Y[i, j]])
                
                # 转换为 Tensor 并开启梯度追踪
                u_tensor = torch.from_numpy(u_val).to(self.device, dtype=torch.float64).requires_grad_(True)
                
                S, dSig0_du, dSig1_du = compute_strain(u_tensor, dF_du)

                sig0 = S[:, 0] # (N_elem, )
                sig1 = S[:, 1] # (N_elem, )

                # 计算目标函数值 J (Frobenius Norm)
                # J^2 = sum_k [ P_row_sum_k * (sig0_k^2 + sig1_k^2) ]
                strain_energy_density = sig0**2 + sig1**2
                weighted_sq_norm = torch.dot(P_row_sum, strain_energy_density)

                J = torch.sqrt(weighted_sq_norm)

                if J.item() < 1e-9:     # 避免 J 为 0 导致除零错误
                    return 0.0, np.zeros_like(u_tensor)

                loss = -J.item()
                
                # 4. 计算解析梯度
                # Grad = - (1/J) * sum_k [ P_row_sum_k * (sig0 * dSig0_du + sig1 * dSig1_du) ]
                
                term0 = sig0 * P_row_sum
                term1 = sig1 * P_row_sum
                
                # 利用广播机制计算加权梯度和: (N_elem, 1) * (N_elem, N_act) -> (N_elem, N_act) -> sum -> (N_act, )
                # grad_sum = sum( term0[:, None] * dSig0_du + term1[:, None] * dSig1_du, dim=0 )
                
                # 'n, na -> a': 对 n (element) 维度求和
                grad_sig0 = torch.einsum('n, na -> a', term0, dSig0_du)
                grad_sig1 = torch.einsum('n, na -> a', term1, dSig1_du)
                
                grad_u_tensor = -(grad_sig0 + grad_sig1) / J

                Z_loss[i, j] = loss

                # 注意：如果梯度过大，画图会乱，可以归一化
                U_grad[i, j] = grad_u_tensor[0].item()
                V_grad[i, j] = grad_u_tensor[1].item()

        # 3. 绘图
        plt.figure(figsize=(10, 8))
        
        # A. 绘制等高线 (Loss 地形)
        # 使用 levels 多一点可以看到细节
        cp = plt.contourf(X, Y, Z_loss, levels=50, cmap='viridis')
        plt.colorbar(cp, label='-LogDet Loss (Lower is Better)')
        
        # B. 绘制梯度场 (Quiver)
        # 稀疏采样一点，不然箭头太密看不清
        skip = 2
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                -U_grad[::skip, ::skip], -V_grad[::skip, ::skip], #以此方向为负梯度方向(下降方向)
                color='white', alpha=0.6, scale=None, label='Descent Direction')
        
        # C. 绘制约束边界 (圆)
        theta = np.linspace(0, 2*np.pi, 100)
        r = max_mag
        x_circ = r * np.cos(theta)
        y_circ = r * np.sin(theta)
        plt.plot(x_circ, y_circ, 'r--', linewidth=2, label='Max Action Constraint')
        
        # D. 标记原点
        plt.plot(0, 0, 'rx', markersize=10, label='Origin (u=0)')
        
        plt.title(f"D-Optimality Loss Landscape\n(Based on Linearized H)")
        plt.xlabel("Action X")
        plt.ylabel("Action Y")
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.savefig(VISUALIZATION_DIR / "loss_landscape.svg", dpi=150)
        print(f"Visualization saved to '{VISUALIZATION_DIR / 'loss_landscape.svg'}'")
        plt.close()

    def optimize_action(self, 
                        contact_node_idx: int, 
                        current_q_vision: torch.Tensor,
                        current_stiffness_est: torch.Tensor,
                        weighted_cells: List[int],
                        max_action_mag: float = 0.03,
                        optimize_options: Dict = None,):
        """
        基于当前视觉观测和刚度估计，优化下一步的施力动作
        Args:
            contact_node_idx: 施力点索引
            current_q_vision (torch.Tensor): 视觉测量的当前位置
            current_stiffness_est (torch.Tensor): 当前刚度估计
            weighted_cells: 参与加权的单元索引列表
            max_force_mag: 动作幅值的截断上限
            iter_num: 优化迭代次数
            method: 优化方法选择 ('SLSQP' or 'IPOPT')        
        Returns:
            optimal_action: shape (2,) 优化后的动作向量 (fx, fy)
        """
        
        # 1. 更新物理模型到当前观测状态
        self.soft_model.update_state_from_vision(current_q_vision, current_stiffness_est)
        dq_du = self.soft_model.contact_jacobian()  # (N_dofs, N_contact_dofs)
        dq_du = torch.from_numpy(dq_du).to(self.device, dtype=torch.float64)
        # np.savetxt(OUTPUT_DIR / "dq_du.csv", dq_du.cpu().numpy(), delimiter=",")
        
        # dH_du, _ = self.soft_model.compute_dH_du(dq_du, current_stiffness_est)  # 计算 dH/du, 虽然不依赖于H, 但保险起见放在H计算之后
        dF_du_tensor = self.soft_model.compute_dF_du(dq_du)  # 计算 dF/du 矩阵 (N_elem, 2, 2, N_contact_dofs)
        # np.savetxt(OUTPUT_DIR / "dH_du_act0.csv", dH_du[:, :, 0].cpu().numpy(), delimiter=",")
        # np.savetxt(OUTPUT_DIR / "dH_du_act1.csv", dH_du[:, :, 1].cpu().numpy(), delimiter=",")
        
        P_prior = self.estimator.P
        P_row_sum = torch.sum(P_prior**2, dim=1)
        weight = torch.ones((self.soft_model.ELEMENT_N,), device=self.device, dtype=torch.float64)
        weight[weighted_cells] = 5.0  # 加大这些单元的权重
        P_row_sum = P_row_sum * weight
        # np.savetxt(OUTPUT_DIR / "Y_prior.csv", Y_prior.cpu().numpy(), delimiter=",")
        # np.savetxt(OUTPUT_DIR / "P_prior.csv", P_prior.cpu().numpy(), delimiter=",")
      
        # self.visualize_loss_landscape(dF_du_tensor, P_prior, max_mag=max_action_mag)
        # exit()

        sig_sum = torch.ones((self.soft_model.ELEMENT_N, ), device=self.device, dtype=torch.float64) * 2.0  # 初始应变值 (无动作时)
        loss_current = torch.sqrt(torch.dot(P_row_sum, sig_sum)).item()
        print(f"Current loss (|S_(t+1) Σ_t|): {loss_current:.4f}")

        print(f"--- Optimizing Action for Node {contact_node_idx} ---")

        # ======================================================================
        # Maximize the uncertainty-weighted strain energy norm (encourage stretching)
        # ======================================================================
        def objective_func(u_vec: np.ndarray):
            u_tensor = torch.from_numpy(u_vec).to(self.device, dtype=torch.float64)

            # # 预测运动学状态 q_next
            # delta_q = (dq_du @ u_tensor).view(current_q_vision.shape)
            # q_next = current_q_vision + delta_q

            # 计算应变及其梯度 (前向传播)
            S, dSig0_du, dSig1_du = compute_strain(u_tensor, dF_du_tensor)

            sig0 = S[:, 0] # (N_elem, )
            sig1 = S[:, 1] # (N_elem, )

            # 计算目标函数值 J (Frobenius Norm)
            # J^2 = sum_k [ P_row_sum_k * (sig0_k^2 + sig1_k^2) ]
            strain_energy_density = sig0**2 + sig1**2
            weighted_sq_norm = torch.dot(P_row_sum, strain_energy_density)

            J = torch.sqrt(weighted_sq_norm)

            if J.item() < 1e-9:     # 避免 J 为 0 导致除零错误
                return 0.0, np.zeros_like(u_vec)

            loss = -J.item()
            
            # 4. 计算解析梯度
            # Grad = - (1/J) * sum_k [ P_row_sum_k * (sig0 * dSig0_du + sig1 * dSig1_du) ]
            term0 = sig0 * P_row_sum
            term1 = sig1 * P_row_sum
            
            # 利用广播机制计算加权梯度和: (N_elem, 1) * (N_elem, N_act) -> (N_elem, N_act) -> sum -> (N_act, )
            # grad_sum = sum( term0[:, None] * dSig0_du + term1[:, None] * dSig1_du, dim=0 )
            
            # 'n, na -> a': 对 n (element) 维度求和
            grad_sig0 = torch.einsum('n, na -> a', term0, dSig0_du)
            grad_sig1 = torch.einsum('n, na -> a', term1, dSig1_du)
            
            grad_u_tensor = -(grad_sig0 + grad_sig1) / J

            return loss / loss_current, grad_u_tensor.cpu().numpy() / loss_current
        
        # ===== 约束动作的大小 =====
        def compute_act_cons(u):
            val = np.dot(u, u) / (max_action_mag**2 + 1e-12)
            # if val > 1.00: print(f"  [Action Violation] |u|={np.linalg.norm(u):.4f}>{max_action_mag:.4f}")
            return val   
            
        cons_act = NonlinearConstraint(
            fun=compute_act_cons,
            lb=-np.inf,
            ub=1.0,
            jac=lambda u: 2*u / (max_action_mag**2 + 1e-12), # 解析雅可比
            keep_feasible=True # 关键参数: 试图让中间迭代始终在圆内
        )

        # 组装约束列表
        constraints = [cons_act]
        bounds = [(-max_action_mag, max_action_mag), (-max_action_mag, max_action_mag)]
        
        # u0_np = np.ones(len(self.soft_model.contact_particle_list) * 2) * 0.03
        u0_np = np.array([0.0, 0.0])  # 初始动作猜测

        if optimize_options.get("method") is not None:
            method = optimize_options["method"]
        else:
            raise ValueError("Optimization method must be specified in optimize_options['method']")
        iter_num = optimize_options.get("iter_num", 100)

        if method == 'SLSQP':   # method='SLSQP': 序列最小二乘规划，适合处理含约束的平滑非线性优化
            options = {
                'ftol': 1e-4, 
                'maxiter': iter_num, 
                'disp': True}
        elif method == 'trust-constr':
            options = {
                'gtol': 1e-4,
                'maxiter': iter_num,
                'disp': True}
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        start_time = time.time()
        res = minimize(
            fun=objective_func,
            x0=u0_np,
            method=method,  # 'trust-constr', 'SLSQP'
            jac=True,       # 明确告知 objective_func 会返回梯度
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        
        print(f"Optimization completed in {time.time() - start_time:.4f} seconds.")

        if not res.success:
            print(f"Optimization Warning: {res.message}")

        node_pos_new = current_q_vision + (dq_du @ torch.from_numpy(res.x).to(self.device, dtype=torch.float64)).view(current_q_vision.shape)
        # write_mshv2_triangular(OUTPUT_DIR / "optimized_mesh.msh", node_pos_new.cpu().numpy(), self.soft_model.ele.to_numpy())
        cells = [("triangle", soft_model.ele.to_numpy().astype(np.int32))]
        mesh = meshio.Mesh(
            np.hstack([node_pos_new.cpu().numpy(), np.zeros((node_pos_new.shape[0], 1))]),
            cells,
        )
        mesh.write(f"{OUTPUT_DIR}/optimized_mesh.vtu")

        return res.x, -res.fun * loss_current


if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True)

    demo_dir = DATA_DIR / "demo" / "pd_stretch_data_hete" / "20260206_101721"
    dataset = HDF5PdDataset(data_directory=str(demo_dir))
    print(f"数据集加载完成，共包含 {len(dataset)} 个样本。")

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
    OBSERVE_DICT = {node_idx: i for i, node_idx in enumerate(OBSERVE_NODES)}

    model_cache: Dict[int, Soft2DForceExtended] = {}
    
    init_k_guess = torch.ones(FACE_NUM, device="cuda") * 400.0 # 初始猜测

    ekf = StiffnessEKF(
        mesh_info={"PARTICLE_N": NODE_NUM, "ELEMENT_N": FACE_NUM},
        initial_stiffness=init_k_guess,
        observe_nodes=OBSERVE_NODES,
        p_init_var=1.e20,
        q_process_noise=1.e3,
        sigma_q=1.e-4,
        sigma_f=1e-1,
    )

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
                contact=contact_idx, E=1.e1, nu=0.3, dt=1.e-2, density=1.e1, device="cuda",
            )
            model_cache[contact_idx] = new_model
            print(f"-> Construct new model for contact idx: {contact_idx}")

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

    np.savetxt(OUTPUT_DIR / "P_matrix_ekf.csv", ekf.P.cpu().numpy(), delimiter=",")
    
    edge_celles = [44, 58, 74, 85, 87, 90, 91, 130, 138, 142, 319]

    soft_init_model = Soft2DForceExtended(
                shape=MESH_DATA, fix=FIXED_NODES, 
                contact=22, E=1.e1, nu=0.3, dt=1.e-2, density=1.e1, device="cuda",
            )

    planner = ActiveStiffnessPlanner(soft_init_model, ekf)

    vision_q_tensor = soft_init_model.node_pos_init.to_torch(device="cuda").double()

    time_start = time.time()
    optimal_action, optimal_fun = planner.optimize_action(
        contact_node_idx=22,
        current_q_vision=vision_q_tensor,
        current_stiffness_est=torch.ones(FACE_NUM, device="cuda") * 400.0,
        weighted_cells = edge_celles,
        max_action_mag=0.03,
        optimize_options = {
            "method": 'SLSQP',  # 'SLSQP' or 'trust-constr'
            "iter_num": 100,
        },
    )
    spend_time = time.time() - time_start
    print(f"Optimization Time: {spend_time:.4f} seconds")
    print(f"Optimal Action: {optimal_action}; Magnitude: {np.linalg.norm(optimal_action):.4f}")
    print(f"Optimal Objective Function Value: {optimal_fun:.6f}")