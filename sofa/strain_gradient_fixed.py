""" 使用diffpd计算节点位置相对于应变权重的梯度矩阵；
并且fixed constraint等边界条件不变

Date: 2025-11-12
"""
import time
from collections import defaultdict
import numpy as np
from typing import Dict
import torch
import taichi as ti

from const import MESH_DIR, OUTPUT_DIR, ROOT_DIR
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular
from deformation_model.diffpd_2d import Soft2D
from sofa.stretch_dataloader import SofaMSHDataset

@ti.data_oriented
class Soft2DStrainGradient(Soft2D):
    def __init__(self, shape, fix, contact, dt, density, device, **kwargs):
        super().__init__(shape, fix, contact, E=1.e3, nu=0.3, dt=dt, density=density, damp=3.e0, **kwargs)

        print(f"Stretch weight of element 1: {self.stretch_weight[1]}")

        self.device = device
        self.stretch_weight.fill(0.)
        self.stretch_grad = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.ELEMENT_N))
        self.mass_torch = self.node_mass.to_torch(device=self.device)
        self.lhs_torch:torch.Tensor = None  # 用于缓存 lhs 矩阵的 torch 版本
        self.z_torch:torch.Tensor = None

        # 完成预计算
        self.construct_lhs_mass()
        self.construct_lhs_positional()
        self.construct_lhs_damp()
        self.construct_lhs_stretch()

    @ti.kernel
    def construct_stretch_lhs(self, w_input:ti.types.ndarray()):
        """ 根据输入的权重张量构建stretch weight字段
        Args:
            w_input (torch.Tensor): 形状为 (num_elements,) 的张量，表示每个单元的stretch weight (与体积无关)
        """
        self.lhs_stretch.fill(0.)
        for e_i in range(self.ELEMENT_N):
            self.stretch_weight[e_i] = w_input[e_i] * self.ele_volume[e_i]
            ATA = self.ATA[e_i]
            stretch_w = self.stretch_weight[e_i]
            idx1, idx2, idx3 = self.ele[e_i]
            q_idx_vec = ti.Vector([idx1, idx2, idx3])
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs_stretch[lhs_row_idx, lhs_col_idx] += stretch_w * ATA[A_row_idx, A_col_idx]

    @ti.kernel
    def gradient_stretch(self):
        """ 计算stretch constraint的一阶梯度
        """
        self.stretch_grad.fill(0.)
        for f_i in range(self.ELEMENT_N):
            F_AT = self.F_A[f_i].transpose()
            F_i = self.F[f_i]
            Bp_stretch = self.Bp_shear[f_i]

            F_ATstrain = F_AT @ (F_i - Bp_stretch).transpose()  # shape: (3, 2), strain=F-Bp

            for q_i, dim_idx in ti.ndrange(3, 2):
                q_idx = self.ele[f_i][q_i]
                self.stretch_grad[q_idx*2+dim_idx, f_i] += F_ATstrain[q_i, dim_idx] * self.ele_volume[f_i]

    def backward_z(self, dL_dxew: torch.Tensor):
        """ 计算diffpd中的z向量 
        Args:
            dL_dxew (torch.Tensor): 支持 (num_nodes*2,m) 的张量，表示损失对节点位置的梯度
        """
        b = dL_dxew
        # mass_torch = self.node_mass.to_torch(device=self.device) / self.dt**2
        # mass_matrix = torch.diag(torch.repeat_interleave(mass_torch, 2))
        eye2 = torch.eye(2, device=self.device, dtype=torch.float64)
        # A = mass_matrix + torch.kron(self.lhs_stretch.to_torch(device=self.device), eye2) \
        #      - self.dBp_stretch.to_torch(device=self.device) \
        #      + torch.kron(self.lhs_positional.to_torch(device=self.device), eye2)
        A = torch.kron(self.lhs_torch, eye2) - self.dBp_stretch.to_torch(device=self.device) \

        z = torch.linalg.solve(A, b)
        return z

    def backward_stretch_w(self):
        """ 计算loss对于stretch weight的梯度，可以处理rhs为向量或矩阵的情况 """
        rhs = - self.stretch_grad.to_torch(device=self.device)
        # rhs = torch.sum(rhs, dim=1)  # Shape: (num_nodes*2,)
        z = self.z
        return z @ rhs

    @ti.kernel
    def reconstruct_lhs(self):
        self.lhs.fill(0.)
        for i, j in ti.ndrange(self.PARTICLE_N, self.PARTICLE_N):
            self.lhs[i, j] = self.lhs_mass[i, j] + self.lhs_stretch[i, j] \
                             + self.lhs_damp[i, j] + self.lhs_positional[i, j]

    # @ti.kernel
    # def construct_sn(self, action:ti.types.ndarray()):
    #     for q_i in range(self.PARTICLE_N):
    #         self.sn[q_i*2  ] = self.node_pos[q_i].x
    #         self.sn[q_i*2+1] = self.node_pos[q_i].y
        
    #     q_i = self.contact_particle_ti[0]
    #     self.sn[q_i*2  ] = self.node_pos[q_i].x + action[0]
    #     self.sn[q_i*2+1] = self.node_pos[q_i].y + action[1]

    # 重载原方法
    def substep(self):
        self.construct_sn()
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            self.local_solve()
            rhs_ts = self.rhs.to_torch(device=self.device)
            node_pos_ts_x = torch.linalg.solve(self.lhs_torch, rhs_ts[0::2])
            node_pos_ts_y = torch.linalg.solve(self.lhs_torch, rhs_ts[1::2])

            self.update_pos_new(node_pos_ts_x, node_pos_ts_y)

        self.update_vel_pos()

    def forward_step(self, x_prev_ref: torch.Tensor, stretch_w: torch.Tensor, u_t: torch.Tensor):
        """ 基于当前参数前向一步模拟，返回节点位置 """
        self.construct_stretch_lhs(stretch_w)
        self.reconstruct_lhs()
        self.lhs_torch = self.lhs.to_torch(device=self.device)
        np.savetxt("lhs_stretch_mat.csv", self.lhs_stretch.to_numpy(), fmt='%.6f', delimiter=',')
        # np.savetxt("lhs_matrix.csv", self.lhs.to_numpy(), fmt='%.6f', delimiter=',')

        self.node_pos.from_torch(x_prev_ref[:, 0:2])
        self.node_vel.fill(0.)
        self.contact_vel.from_torch(u_t[0:2].unsqueeze(0) / self.dt)
        self.substep()

        self.contact_vel.fill(0.)
        for i in range(200):
            self.substep()
            node_vel_avg = torch.mean(torch.linalg.norm(self.node_vel.to_torch(device=self.device), dim=1))
            # print(f"Iteration {i}, average node velocity: {node_vel_avg.item():.6e}")
            # if node_vel_avg.item() < 1.e-6:
            #     break

        return self.node_pos.to_torch(device=self.device)

    def backward_step(self, dL_dxnew:torch.Tensor):
        """ 计算损失对各种输入的梯度（位置，权重参数等）
        Args:
            dL_dxnew (torch.Tensor) : 后向传播传入的损失对节点位置的梯度，形状为 (num_nodes*2,)
        """
        self.construct_E_hessian()
        self.gradient_stretch()

        self.z = self.backward_z(dL_dxnew)
        dL_dw_torch = self.backward_stretch_w()
        return dL_dw_torch

    
MODEL_CACHE: Dict[int, Soft2DStrainGradient] = {}   # 缓存模型实例的全局字典
contact_list = [77, 115, 120]
mesh_file = MESH_DIR / "rectangle.msh"

def get_model_for_batch(contact_idx: int) -> Soft2DStrainGradient:
    """
    一个辅助函数，用于从缓存中获取（或创建）模型。
    """    
    if contact_idx not in MODEL_CACHE:
        # 实例化一个新模型 (昂贵的操作，但只发生一次)
        contact_node = contact_idx
        # triangles = batch['triangles']
        # device = triangles.device
        
        MODEL_CACHE[contact_idx] = Soft2DStrainGradient(
            shape=[0.1, 0.1], fix=list(range(0, 11)),   # 先假定已知，实际上应该从batch中读取
            contact=contact_node, dt=1.e-2, density=1.e1, device="cuda"
        )
    return MODEL_CACHE[contact_idx]


if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True)

    dataset = SofaMSHDataset(data_directory=str(OUTPUT_DIR), rule=r"pd_contact(\d+)_step(\d{3})\.msh")
    
    target_step:int = 19

    target_index = -1
    for i, pair in enumerate(dataset.sample_pairs):
        # pair 是一个字典: {'pre_step': 19, 'contact_id': ..., ...}
        if pair['pre_step'] == target_step:
            target_index = i
            print(f"找到目标数据！索引: {i}")
            print(f"文件对: {pair['pre_file'].name} -> {pair['post_file'].name}")
            break

    if target_index == -1:
        raise ValueError(f"未找到 step {target_step} 开头的数据对。")
    
    sample = dataset[target_index]
    contact_node = sample['contact_idx']
    pre_node_pos = sample['pre_x'].to('cuda')
    post_node_pos = sample['post_x'].to('cuda')
    action = sample['action'].to('cuda')

    contact_node = 77
    pre_node, _ = read_mshv2_triangular("gt_pre.msh")
    post_node, _ = read_mshv2_triangular("gt_post.msh")
    pre_node_pos = torch.tensor(pre_node, device="cuda", dtype=torch.float64)
    post_node_pos = torch.tensor(post_node, device="cuda", dtype=torch.float64)
    action = post_node_pos[contact_node, :] - pre_node_pos[contact_node, :]

    # print(f"pre nodes pos: {pre_node_pos}")
    # print(f"post nodes pos: {post_node_pos}")
    print(f"action: {action}")
    # print(f"node 77 post pos: {post_node_pos[77]}")

    soft_model = Soft2DStrainGradient(
        shape=[0.1, 0.1], fix=list(range(0, 11)), 
        contact=contact_node, dt=1.e-2, density=1.e1, device="cuda"
    )

    # init_w = 0.9e3 / (1+0.3) * torch.ones(soft_model.ELEMENT_N, device="cuda", dtype=torch.float64)
    init_w = 980 * torch.ones(soft_model.ELEMENT_N, device="cuda", dtype=torch.float64)

    post_node_sim_pos = soft_model.forward_step(pre_node_pos, init_w, action)
    np.savetxt("strain_post_sim_pos.csv", post_node_sim_pos.cpu().numpy(), fmt="%.6f", delimiter=',')

    # j = soft_model.backward_step()
    # print(f"dx_const: {soft_model.dx_const.to_numpy()}")

    error = (post_node_sim_pos - post_node_pos[:, 0:2]) / 1.e-2 * 1.e3  # 转为mm单位下的误差
    mass_torch = soft_model.mass_torch
    # loss = (error.flatten() * error.flatten() * mass_torch.repeat_interleave(2)).sum()
    loss = torch.sum(error**2)
    print(f"Loss: {loss.item():.6e}")

    loss_w_grad = soft_model.backward_step(2*error.flatten())
    print(f"loss regarding w: {loss_w_grad}")

    # w_grad = error.flatten() @ j   # (N_nodes*DIM,) = (N_nodes*DIM,) @ (N_nodes*DIM, N_params)
    # print(f"loss to w gradient: {w_grad}")

    # np.savetxt("error_e2.csv", error.cpu().numpy(), fmt="%.6f", delimiter=',')
    # np.savetxt("w_grad.csv", w_grad.cpu().numpy(), fmt="%e", delimiter=',')
    # np.savetxt("strain_gradient.csv", j.cpu().numpy(), delimiter=',')

    write_mshv2_triangular(f"{ROOT_DIR}/pd_gt_post.msh", post_node_pos[:, 0:2].cpu().numpy(), soft_model.ele.to_numpy())
    write_mshv2_triangular(f"{ROOT_DIR}/pd_sim_post.msh", post_node_sim_pos.cpu().numpy(), soft_model.ele.to_numpy())
