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
from typing import List, Dict
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
import torch
import taichi as ti
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from const import ROOT_DIR, MESH_DIR, OUTPUT_DIR, DATA_DIR, VISUALIZATION_DIR, ARTICLE_VIS_DIR
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

            self.dforce_dw[2*idx1, f_i]   += self.dforce_dw_mat[f_i][0, 0] * self.ele_volume[f_i]
            self.dforce_dw[2*idx1+1, f_i] += self.dforce_dw_mat[f_i][0, 1] * self.ele_volume[f_i]
            self.dforce_dw[2*idx2, f_i]   += self.dforce_dw_mat[f_i][1, 0] * self.ele_volume[f_i]
            self.dforce_dw[2*idx2+1, f_i] += self.dforce_dw_mat[f_i][1, 1] * self.ele_volume[f_i]
            self.dforce_dw[2*idx3, f_i]   += self.dforce_dw_mat[f_i][2, 0] * self.ele_volume[f_i]
            self.dforce_dw[2*idx3+1, f_i] += self.dforce_dw_mat[f_i][2, 1] * self.ele_volume[f_i]

    @ti.kernel
    def reconstruct_stretch_weight(self, stretch_weight:ti.types.ndarray()):
        for f_i in range(self.ELEMENT_N):
            self.stretch_weight[f_i] = stretch_weight[f_i] * self.ele_volume[f_i]


if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True)

    lam = 1.e-6  # Hessian 正则化系数

    # load dataset #
    demo_dir = DATA_DIR / "demo" / "pd_stretch_data_hete" / "20260215_hard[2]"
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
    OBSERVE_NODES_INIT = list(set(range(NODE_NUM)) - set(FIXED_NODES))
    # OBSERVE_DOFS = np.stack([np.array(OBSERVE_NODES) * 2, np.array(OBSERVE_NODES) * 2 + 1], axis=-1).flatten().tolist()

    total_partial_g = np.zeros((FACE_NUM,))
    total_hessian_g = np.zeros((FACE_NUM, FACE_NUM))
    count_samples = 0
    total_loss = 0.0

    init_w_value = 450.0
    init_w = init_w_value * torch.ones(FACE_NUM, device="cuda", dtype=torch.float64)
    print(f"初始化刚度值: {init_w_value}")

    model_cache: Dict[int, Soft2DForce] = {}

    target_idx_list = []
    for i in range(len(dataset)):
        sample = dataset[i]

        # if sample['step_idx'] != 19:
        #     continue  # 只使用每个轨迹的最后一个时间步数据

        contact_idx = int(sample['contact_idx'])
        # pre_node_pos = sample['pre_x'].to('cuda')
        post_node_pos = sample['post_x'][:, :2].to('cuda')
        # action = sample['action'].to('cuda')
        node_force = sample['force'].to('cuda')

        OBSERVE_NODES = list(set(OBSERVE_NODES_INIT))
        OBSERVE_DOFS = np.stack([np.array(OBSERVE_NODES) * 2, np.array(OBSERVE_NODES) * 2 + 1], axis=-1).flatten().tolist()

        if contact_idx not in model_cache:
            # construct soft body model #
            new_model = Soft2DForce(
                shape=MESH_DATA, fix=FIXED_NODES, 
                contact=contact_idx, E=1.e1, nu=0.3, dt=1.e-2, density=1.e1, device="cuda"
            )
            new_model.reconstruct_stretch_weight(init_w)
            new_model.precomputation()

            model_cache[contact_idx] = new_model

        soft_model = model_cache[contact_idx]

        # soft_model.node_pos.from_numpy(pre_node_pos[:, 0:2].cpu().numpy())
        soft_model.node_pos.from_torch(post_node_pos)

        # forward: Deformation gradient and internal force #
        soft_model.cal_deformation_gradient()
        soft_model.update_internal_force()

        # backward: gradient of internal force wrt. stretch weight #
        soft_model.cal_internal_force_gradient()

        internal_force = soft_model.force.to_numpy()
        dforce_dw = soft_model.dforce_dw.to_numpy()  # shape: [N*2, E]

        dforce_dw_observe = dforce_dw[OBSERVE_DOFS,:]    # 去掉固定节点的部分
        force_residual = internal_force[OBSERVE_NODES,:] - node_force.cpu().numpy()[OBSERVE_NODES,:]  # shape: [N - fixed_N, 2]
        residual_flat = force_residual.flatten()

        current_loss = np.sum(force_residual**2)
        # print(f"Sample {i} (Contact {contact_idx}): Loss = {current_loss:.4e}")
        
        current_partial_g = 2 * residual_flat @ dforce_dw_observe  # shape: [E,]
        current_hessian_g = 2 * dforce_dw_observe.T @ dforce_dw_observe  # shape: [E, E]

        total_partial_g += current_partial_g
        total_hessian_g += current_hessian_g
        total_loss += current_loss
        count_samples += 1

    print(f"\n数据处理完毕，共聚合 {count_samples} 个样本。开始求解线性方程组...")

    avg_partial_g = total_partial_g / count_samples
    avg_hessian_g = total_hessian_g / count_samples

    # H_new = H + lambda * I
    hessian_reg = avg_hessian_g + np.eye(FACE_NUM) * lam

    # scipy, 带约束求解
    H = avg_hessian_g + np.eye(FACE_NUM) * lam
    g0 = avg_partial_g
    w_init_val = init_w.cpu().numpy() # 你的初始猜测值
    def objective(delta_w):
        # 0.5 * dw' * H * dw + g0' * dw
        return 0.5 * delta_w.dot(H @ delta_w) + g0.dot(delta_w)

    def jacobian(delta_w):
        # H * dw + g0
        return H @ delta_w + g0

    bounds = [(1e-6 - w_val, None) for w_val in w_init_val]

    delta_w_0 = np.zeros_like(w_init_val)
    print("-> 开始 Scipy 优化...")
    res = minimize(objective, delta_w_0, method='L-BFGS-B', jac=jacobian, 
                   bounds=bounds, )

    if res.success:
        final_delta_w = res.x
        final_w = w_init_val + final_delta_w
        print("优化成功！")
    else:
        print("优化失败")

    updated_w = init_w.cpu().numpy() + res.x

    # # Solve: (H + lam*I) * delta_w = -g
    # delta_w = - np.linalg.solve(avg_hessian_g, avg_partial_g)
    # updated_w = init_w.cpu().numpy() + delta_w

    # print hard and free element stiffness #
    print("\n刚度值更新完毕，部分单元刚度值如下：")
    print(f"hard elements: {updated_w[hard_ele_list]}")
    print(f"free elements: {updated_w[free_ele_list]}")
    
    # analyse results #
    update_w_mean = np.mean(updated_w)
    resolution_mat = np.linalg.inv(hessian_reg) @ total_hessian_g
    problematic_indices = np.where(np.diag(resolution_mat)<0.6)[0]
    rel_error = np.abs(REAL_W - updated_w) / REAL_W

    np.savetxt(f"{OUTPUT_DIR}/stretch_weight_update.csv", updated_w, fmt="%.6f", delimiter=',')
    # np.savetxt(f"dforce_dw.csv", dforce_dw, fmt="%.6f", delimiter=',')
    # np.savetxt(f"hessian_g.csv", total_hessian_g, fmt="%.6f", delimiter=',')
    # np.savetxt(f"hessian_g_inv.csv", np.linalg.inv(total_hessian_g), fmt="%.6f", delimiter=',')
    np.savetxt(f"{OUTPUT_DIR}/resolution_mat.csv", np.diag(resolution_mat), fmt="%.6f", delimiter=',')
    print(f"Loss: {total_loss:.6e}")
    print(f"Upated stretch weights (first 10): {updated_w[:10]}")
    print(f"Updated weight mean: {update_w_mean:.6f}")
    print(f"Problematic indices (resolution < 0.6): {problematic_indices}")
    print(f"Problematic indices value: {updated_w[problematic_indices]}")

    # 可视化 internal_force（每个力向量绘制在对应的 post_node_pos 位置） #
    pos_np = post_node_pos.detach().cpu().numpy()
    force_np = - internal_force  # shape: [N, 2]
    print(f"Force sum: {np.sum(force_np, axis=0)}")
    np.savetxt(f"{OUTPUT_DIR}/internal_force.csv", force_np, fmt="%.6f", delimiter=',')

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
    plt.title(f"Internal Force Field")
    plt.legend()
    plt.tight_layout()
    
    out_path = Path(VISUALIZATION_DIR) / f"internal_force.svg"
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
    plt.title(f'Stiffness per Element')

    out_path_stiff = Path(VISUALIZATION_DIR) / f"stiffness_gn.svg"
    plt.savefig(out_path_stiff, dpi=200)
    plt.close()
    print(f"Saved stiffness visualization to {out_path_stiff}")

    # resimulate with updated stretch weights and calculate uncertainty #
    soft_model.stretch_weight.from_numpy(updated_w)
    soft_model.precomputation()

    soft_model.node_pos.from_numpy(post_node_pos[:, 0:2].cpu().numpy())
    soft_model.cal_deformation_gradient()
    soft_model.update_internal_force()

    internal_force_update = soft_model.force.to_numpy()
    loss_new = np.sum((internal_force_update[OBSERVE_NODES,:] - node_force.cpu().numpy()[OBSERVE_NODES,:])**2)
    cov_mat = loss_new/(len(OBSERVE_NODES)*2-FACE_NUM)*np.linalg.inv(total_hessian_g)
    dev = np.diag(cov_mat)
    uncertainty_abs = np.sqrt(np.diag(cov_mat))
    uncertainty_rel = uncertainty_abs / updated_w
    uncertainty_rel_real = (updated_w - REAL_W) / REAL_W

    print(f"New Loss after update: {loss_new:.6e}")
    np.savetxt(f"dev.csv", dev, fmt="%.6f", delimiter=',')
    np.savetxt(f"cov_mat.csv", cov_mat, fmt="%.6f", delimiter=',')
    np.savetxt(f"uncertainty_abs.csv", uncertainty_abs, fmt="%.6f", delimiter=',')
    np.savetxt(f"uncertainty_rel.csv", uncertainty_rel, fmt="%.6f", delimiter=',')
    np.savetxt(f"uncertainty_rel_real.csv", uncertainty_rel_real, fmt="%.6f", delimiter=',')


    # exit()
    # ==========================================
    # 论文结果美化代码
    # ==========================================
    from matplotlib.colors import ListedColormap
    import seaborn as sns

    plt.style.use('default') # 重置
    params = {
        'font.family': 'serif',        # 衬线体 (Times New Roman 等)
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], # 显式指定
        'mathtext.fontset': 'stix',    # 数学公式字体更像 LaTeX
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.spines.top': False,      # 去掉顶部边框 (更简洁)
        'axes.spines.right': False,    # 去掉右侧边框
    }
    plt.rcParams.update(params)

    cmap_custom = sns.color_palette("RdYlBu_r", as_cmap=True)

    # 可视化刚度值
    raw_k = updated_w.copy()
    v_min, v_max = raw_k.min(), raw_k.max()
    tick_locs = np.arange(np.ceil((v_min)), np.floor((v_max)))
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    im2 = ax1.tripcolor(node_pos_np[:, 0], node_pos_np[:, 1], triangles,
                        facecolors=raw_k, shading='flat', 
                        edgecolors='#333333', linewidth=1., alpha=0.9, cmap=cmap_custom)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # 自定义 Colorbar 的刻度
    cbar1 = plt.colorbar(im2, ax=ax1, pad=0.01, shrink=0.8, aspect=20, fraction=0.046)
    cbar1.ax.tick_params(length=0, labelsize=14)

    # cbar1.set_ticks(tick_locs)
    # cbar1.set_ticklabels([f"$10^{{ {int(loc):d} }}$" for loc in tick_locs])

    plt.savefig(f"{ARTICLE_VIS_DIR}/stiffness_value_gn.svg", transparent=True, format='svg', dpi=300, bbox_inches='tight')
