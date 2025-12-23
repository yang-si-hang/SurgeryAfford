"""
Input a deformation field and optimize the strain parameters to minimize the force residual.
* Use Newton-Raphson method for optimization. w -= (H + lambda*I)^{-1} @ g
    lambda: regularization coefficient to ensure H + lambda*I is invertible
* Uncertainty estimation: Resolution Matrix R = (H + lambda*I)^{-1} @ H
    R[i] << 1: parameter i is ill-conditioned, R[i] ~ 1: parameter i is well-conditioned
* Covariance matrix of the estimated parameters: Cov = sigma^2 * (H + lambda*I)^{-1}, where sigma^2 = loss / dof
    Cov[i, i] is the variance of parameter i, < 0.1 means high confidence
* Sensitivity matrix: S = dR / dw, the gradient of internal force w.r.t. strain parameters
    coupled elements have similiar sensitivity vector
    to optimize the deformation field for decoupling elements parameter estimation, the sensitivity matrix should be as diagonal as possible (Gram matrix)
Date: 2025-12-17
"""
import time
from pathlib import Path
from typing import List
import numpy as np
from scipy import sparse
import torch
import taichi as ti
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from const import ROOT_DIR, MESH_DIR, OUTPUT_DIR, DATA_DIR
from deformation_model.diffpd_2d import Soft2D
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular
from utilize.mesh_util import extract_edge_from_face, mesh_obj_tri
from deformation_model.pd_data_loader import HDF5PdDataset


@ti.data_oriented
class Soft2DForce(Soft2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dforce_dw = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.ELEMENT_N))  # gradient of internal force wrt. stretch weight

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

            self.dforce_dw[2*idx1, f_i]   += self.dforce_dw_mat[f_i][0, 0]
            self.dforce_dw[2*idx1+1, f_i] += self.dforce_dw_mat[f_i][0, 1]
            self.dforce_dw[2*idx2, f_i]   += self.dforce_dw_mat[f_i][1, 0]
            self.dforce_dw[2*idx2+1, f_i] += self.dforce_dw_mat[f_i][1, 1]
            self.dforce_dw[2*idx3, f_i]   += self.dforce_dw_mat[f_i][2, 0]
            self.dforce_dw[2*idx3+1, f_i] += self.dforce_dw_mat[f_i][2, 1]

    @ti.kernel
    def reconstruct_stretch_weight(self, stretch_weight:ti.types.ndarray()):
        for f_i in range(self.ELEMENT_N):
            self.stretch_weight[f_i] = stretch_weight[f_i] * self.ele_volume[f_i]


if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True)

    real_w, init_w = None, None  # 可以预设真实值和初始值
    lam = 1.e-6  # Hessian 正则化系数

    # load dataset #
    demo_dir = DATA_DIR / "demo" / "pd_stretch_data_hete"
    dataset = HDF5PdDataset(data_directory=str(demo_dir))

    target_step:int = 19
    print(f"正在数据集中查找 Step {target_step} ...")

    target_idx_list = []
    for i, sample_meta in enumerate(dataset.samples):
        # sample_meta 是加载到内存的原始字典 (numpy array格式)
        
        # 检查 step 是否匹配
        # (如果你的 HDF5 文件不止一个，这里可能需要增加对 contact_idx 或 source_file 的判断，
        #  否则它会返回第一个文件中的第 19 步)
        if sample_meta['step_idx'] == target_step:
            target_idx_list.append(i)
            print(f"找到目标数据！索引: {i}")
            print(f"来源文件: {sample_meta['source_file']}")
            print(f"时间步: {sample_meta['step_idx']}")
            print(f"Contact ID: {sample_meta['contact_idx']}")
            # break

    if not target_idx_list:
        raise IndexError(f"Target step {target_step} not found in dataset.")

    partial_g_list = []
    hessian_g_list = []

    for target_idx in target_idx_list:
        sample = dataset[target_idx]      # (Dataset 会在这里把 numpy 转为 tensor)

        contact_node = sample['contact_idx']  # 标量 Tensor 或 int
        pre_node_pos = sample['pre_x'].to('cuda')
        post_node_pos = sample['post_x'].to('cuda')
        action = sample['action'].to('cuda')
        node_force = sample['force'].to('cuda')

        print("\n数据加载成功:")
        print(f"  Pre Pos Shape: {pre_node_pos.shape}")
        print(f"  Action: {action}")

        # construct soft body model #
        soft_model = Soft2DForce(
            shape=[0.1, 0.1], fix=list(range(0, 11)), 
            contact=contact_node, E=1.e1, nu=0.3, dt=1.e-2, density=1.e1, device="cuda"
        )

        real_w = 1000 / 2 / 1.3 * 1.e-4 * torch.ones(soft_model.ELEMENT_N, device="cuda", dtype=torch.float64) if real_w is None else real_w
        init_w = 450 * torch.ones(soft_model.ELEMENT_N, device="cuda", dtype=torch.float64) if init_w is None else init_w

        soft_model.reconstruct_stretch_weight(init_w)
        soft_model.precomputation()

        soft_model.node_pos.from_numpy(pre_node_pos[:, 0:2].cpu().numpy())
        soft_model.cal_deformation_gradient()
        soft_model.update_internal_force()
        soft_model.cal_internal_force_gradient()

        internal_force = soft_model.force.to_numpy()
        dforce_dw = soft_model.dforce_dw.to_numpy()  # shape: [N*2, E]
        dforce_dw_observe = dforce_dw[11*2:,:]    # 去掉固定节点的部分

        force_residual = internal_force[11:,:] - node_force.cpu().numpy()[11:,:]  # shape: [N*2 - fixed_N*2, 2]
        loss = np.sum(force_residual**2)

        partial_g = 2 * force_residual.flatten() @ dforce_dw_observe  # shape: [E,]
        hessian_g = 2 * dforce_dw_observe.T @ dforce_dw_observe  # shape: [E, E]
    
        partial_g_list.append(partial_g)
        hessian_g_list.append(hessian_g)

    # aggregate gradients from multiple contacts (if any) #
    partial_g = np.mean(np.stack(partial_g_list, axis=0), axis=0)
    hessian_g = np.mean(np.stack(hessian_g_list, axis=0), axis=0)
    hessian_g_new = hessian_g + np.eye(soft_model.ELEMENT_N) * lam

    delta_w = - np.linalg.solve(hessian_g, partial_g)
    updated_w = soft_model.stretch_weight.to_numpy() + delta_w
    update_w_mean = np.mean(updated_w)
    resolution_mat = np.linalg.inv(hessian_g_new) @ hessian_g

    problematic_indices = np.where(np.diag(resolution_mat)<0.6)[0]

    np.savetxt(f"stretch_weight_update.csv", updated_w, fmt="%.6f", delimiter=',')
    np.savetxt(f"dforce_dw.csv", dforce_dw, fmt="%.6f", delimiter=',')
    np.savetxt(f"hessian_g.csv", hessian_g, fmt="%.6f", delimiter=',')
    np.savetxt(f"hessian_g_inv.csv", np.linalg.inv(hessian_g), fmt="%.6f", delimiter=',')
    np.savetxt(f"resolution_mat.csv", np.diag(resolution_mat), fmt="%.6f", delimiter=',')
    print(f"Loss: {loss:.6e}")
    print(f"Upated stretch weights (first 10): {updated_w[:10]}")
    print(f"Updated weight mean: {update_w_mean:.6f}")
    print(f"Problematic indices (resolution < 0.6): {problematic_indices}")
    print(f"Problematic indices value: {updated_w[problematic_indices]}")

    # 可视化 internal_force（每个力向量绘制在对应的 post_node_pos 位置） #
    pos_np = post_node_pos.detach().cpu().numpy()
    force_np = - internal_force  # shape: [N, 2]
    print(f"Force sum: {np.sum(force_np, axis=0)}")
    np.savetxt("internal_force.csv", force_np, fmt="%.6f", delimiter=',')

    plt.figure(figsize=(6, 6))
    plt.quiver(
        pos_np[:, 0], pos_np[:, 1],
        force_np[:, 0], force_np[:, 1],
        angles="xy", scale_units="xy", scale=10.0, width=0.002, color="C1", label="Internal Force"
    )
    plt.quiver(
        pos_np[:, 0], pos_np[:, 1],
        -node_force.cpu().numpy()[:, 0], -node_force.cpu().numpy()[:, 1],
        angles="xy", scale_units="xy", scale=10.0, width=0.002, color="C0", label="Target Force"
        )
    plt.xlim(np.min(pos_np[:,0]) - 0.05, np.max(pos_np[:,0]) + 0.05)
    plt.ylim(np.min(pos_np[:,1]) - 0.05, np.max(pos_np[:,1]) + 0.05)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Internal Force Field (step {target_step})")
    plt.legend()
    plt.tight_layout()
    
    out_path = Path(OUTPUT_DIR) / f"internal_force_step{target_step:03d}.svg"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved internal force visualization to {out_path}")

    # 可视化三角形网格每个单元的刚度值 #
    plt.figure(figsize=(6, 6))
    node_pos_np = soft_model.node_pos_init.to_numpy()[:, :2]  # Use post positions for deformed mesh
    triangles = soft_model.ele.to_numpy()
    triang = mtri.Triangulation(node_pos_np[:, 0], node_pos_np[:, 1], triangles)
    
    updated_w_show = updated_w.copy()
    for i in range(updated_w.shape[0]):
        if updated_w[i] <= 0 or updated_w[i] > 1.e3:
            updated_w_show[i] = -10

    plt.tripcolor(triang, facecolors=updated_w_show, cmap='RdBu_r', edgecolors='k')
    plt.colorbar(label='Stiffness')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Stiffness per Element (step {target_step})')

    out_path_stiff = Path(OUTPUT_DIR) / f"stiffness_step{target_step:03d}.svg"
    plt.savefig(out_path_stiff, dpi=200)
    plt.close()
    print(f"Saved stiffness visualization to {out_path_stiff}")

    # resimulate with updated stretch weights and calculate uncertainty #
    soft_model.stretch_weight.from_numpy(updated_w)
    soft_model.precomputation()

    soft_model.node_pos.from_numpy(pre_node_pos[:, 0:2].cpu().numpy())
    soft_model.cal_deformation_gradient()
    soft_model.update_internal_force()

    internal_force_update = soft_model.force.to_numpy()
    loss_new = np.sum((internal_force_update[11:,:] - node_force.cpu().numpy()[11:,:])**2)
    cov_mat = loss_new/(110*2-200)*np.linalg.inv(hessian_g)
    dev = np.diag(cov_mat)
    uncertainty_abs = np.sqrt(np.diag(cov_mat))
    uncertainty_rel = uncertainty_abs / updated_w
    uncertainty_rel_real = (updated_w - real_w.cpu().numpy()) / real_w.cpu().numpy()

    print(f"New Loss after update: {loss_new:.6e}")
    np.savetxt(f"dev.csv", dev, fmt="%.6f", delimiter=',')
    np.savetxt(f"cov_mat.csv", cov_mat, fmt="%.6f", delimiter=',')
    np.savetxt(f"uncertainty_abs.csv", uncertainty_abs, fmt="%.6f", delimiter=',')
    np.savetxt(f"uncertainty_rel.csv", uncertainty_rel, fmt="%.6f", delimiter=',')
    np.savetxt(f"uncertainty_rel_real.csv", uncertainty_rel_real, fmt="%.6f", delimiter=',')
