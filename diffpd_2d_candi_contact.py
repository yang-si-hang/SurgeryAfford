""" 尝试在一起计算多个候选接触点的deformation gradient雅可比矩阵
再计算affordance值
created on 2025-11-5
"""
import time
import json
from pathlib import Path
from typing import List
import numpy as np
import cupy as cp
from scipy import sparse
import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

from const import ROOT_DIR, OUTPUT_DIR
from utilize.mesh_util import read_mshv2_triangular, mesh_obj_tri, write_mshv2_triangular
from diffpd_2d import read_msh

@ti.data_oriented
class Soft2DNocontact:
    # 只计算与接触点无关的部分
    def __init__(self, shape, fix:List[int], 
                 E:float, nu:float, dt:float, density:float, **kwargs):
        self.shape = shape
        if isinstance(self.shape, Path):
            node_np, edge_np, ele_np = read_msh(self.shape)
        elif isinstance(self.shape, List):
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.01)

            msh_file:str = "shape.msh"
            write_mshv2_triangular(msh_file, node_np, ele_np)
        elif isinstance(self.shape, dict):
            node_np = np.array(self.shape['V'])
            edge_np = np.array(self.shape['E'])
            ele_np = np.array(self.shape['F'])

        self.solve_itr:int = 10
        self.E, self.nu, self.dt, self.density = E, nu, dt, density
        self.dim = 2
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
       
        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.node_pos      = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new  = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_vel      = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_voronoi  = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass     = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass_sum = ti.field(dtype=ti.f64, shape=())
        self.node_pos_init.from_numpy(node_np[:,:2].astype(np.float64))
        self.node_pos.from_numpy(node_np[:,:2].astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge.from_numpy(edge_np.astype(np.int32))

        self.ele = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)
        self.ele_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele.from_numpy(ele_np.astype(np.int32))

        self.stretch_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.positional_weight = 0.         # define later

        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.ele_u = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)          # singular value decomposition
        self.ele_v = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.Bp_shear = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)       # stretch part
        self.stretch_stress = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)

        self.sn  = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_positional = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.rhs_stretch = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.pre_fact_lhs_solve = None

        self.dBp_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.PARTICLE_N*2))
        self.dA = None
        self.dx_const = ti.field(ti.f64, shape=self.dim*self.PARTICLE_N)          # dx_dy中的常数部分

        self.fix_particle_list = fix
        self.FIX_N:int = len(self.fix_particle_list)
        self.fix_particle_ti   = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))

        self.construct_mass()
        self.construct_Xg_inv()
        self.positional_weight = 1.e3 * self.node_mass_sum[None] / self.PARTICLE_N / self.dt**2
        self.construct_dx_const()

        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")
        print(f"Positional weight: {self.positional_weight:.3f}")

    @ti.kernel
    def construct_mass(self):
        for f_i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[f_i]
            qa, qb, qc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            ele_volume_tmp = 0.5 * ti.abs(((qb - qa).cross(qc - qa)))
            # print(f"Element {f_i}: {ele_volume_tmp}")

            self.node_voronoi[ia] += ele_volume_tmp / 3.
            self.node_voronoi[ib] += ele_volume_tmp / 3.
            self.node_voronoi[ic] += ele_volume_tmp / 3.
            self.ele_volume[f_i] = ele_volume_tmp
            self.stretch_weight[f_i] = 2 * self.mu * self.ele_volume[f_i]

        for q_i in range(self.PARTICLE_N):
            self.node_mass[q_i] = self.density * self.node_voronoi[q_i]
            self.node_mass_sum[None] += self.node_mass[q_i]
        print(f"Node mass sum: {self.node_mass_sum[None]:.3f}")

    @ti.kernel
    def construct_Xg_inv(self):
        for i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[i]
            a = ti.Vector([self.node_pos_init[ia].x, self.node_pos_init[ia].y])
            b = ti.Vector([self.node_pos_init[ib].x, self.node_pos_init[ib].y])
            c = ti.Vector([self.node_pos_init[ic].x, self.node_pos_init[ic].y])
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.Xg_inv[i] = B_i_inv.inverse()

    @ti.kernel
    def construct_lhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            self.lhs[q_i, q_i] += self.node_mass[q_i] / self.dt**2

    @ti.kernel
    def construct_lhs_stretch(self):
        # https://medium.com/@victorlouisdg/jax-cloth-tutorial-part-1-e7a0e285864f
        for f_i in range(self.ELEMENT_N):
            Xg_inv = self.Xg_inv[f_i]
            a, b, c, d = Xg_inv[0, 0], Xg_inv[0, 1], Xg_inv[1, 0], Xg_inv[1, 1]

            # F's dim=4*6，flatten(F)按照列优先
            self.F_A[f_i][0, 0] = -a - c
            self.F_A[f_i][0, 1] = a
            self.F_A[f_i][0, 2] = c
            self.F_A[f_i][1, 0] = -b - d
            self.F_A[f_i][1, 1] = b
            self.F_A[f_i][1, 2] = d

        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            q_idx_vec = ti.Vector([idx1, idx2, idx3])
            F_A = self.F_A[f_i]
            ATA = F_A.transpose() @ F_A

            stretch_weight = self.stretch_weight[f_i]
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]
                self.lhs_stretch[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]

    @ti.kernel
    def construct_lhs_positional(self):
        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight
            self.lhs_positional[q_i, q_i] += self.positional_weight

        # for i in range(self.CON_N):
        #     q_i = self.contact_particle_ti[i]
        #     self.lhs[q_i, q_i] += self.positional_weight
        #     self.lhs_positional[q_i, q_i] += self.positional_weight

    def precomputation(self):
        self.construct_lhs_mass()
        self.construct_lhs_stretch()
        self.construct_lhs_positional()

    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        # 默认节点速度为0
        for q_i in range(self.PARTICLE_N):
            idx1, idx2 = dim*q_i, dim*q_i+1
            self.sn[idx1] = self.node_pos[q_i].x + dt * 0.
            self.sn[idx2] = self.node_pos[q_i].y + dt * 0.

    @ti.kernel
    def warm_start(self):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = self.sn[q_i*2]
            self.node_pos_new[q_i].y = self.sn[q_i*2 + 1]

    @ti.kernel
    def construct_rhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            idx1, idx2 = q_i*self.dim, q_i*self.dim+1
            self.rhs[idx1] += self.node_mass[q_i] * self.sn[idx1] / self.dt**2
            self.rhs[idx2] += self.node_mass[q_i] * self.sn[idx2] / self.dt**2

    @ti.kernel
    def construct_rhs_stretch(self):
        # 计算的变形梯度和奇异值分解，是diffpd构建导数需要的
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos_new[idx1], self.node_pos_new[idx2], self.node_pos_new[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = F_i
            # print(f"F_i:{F_i:e}")

            U, sig, V = ti.svd(F_i, ti.f64)
            self.ele_u[f_i] = U
            self.ele_v[f_i] = V
            self.stretch_stress[f_i] = ti.Vector([sig[0,0], sig[1,1]], dt=ti.f64)
            # print(f"U:{U:e}; sig:{sig:e}; V:{V:e}")
            self.Bp_shear[f_i] = U @ ti.Matrix([[1., 0], [0., 1]], ti.f64) @ V.transpose()
            self.stretch_energy[f_i] = 0.5 * self.ele_volume[f_i] * ((sig[0,0]-1.)**2 + (sig[1,1]-1.)**2)
            # print(f"Bp_shear:{self.Bp_shear[f_i]:e}")

        for f_i in range(self.ELEMENT_N):
            Bp_shear_i = self.Bp_shear[f_i]
            F_AT = self.F_A[f_i].transpose()

            # Bp_shear_i做transpose，因为AT需要与Bp的x，y，z分别矩阵乘法
            F_ATBp = F_AT @ Bp_shear_i.transpose() * self.stretch_weight[f_i]

            for q_i, dim_idx in ti.ndrange(3, 2):
                q_idx = self.ele[f_i][q_i]
                self.rhs[q_idx*2+dim_idx] += F_ATBp[q_i, dim_idx]
                # self.rhs_stretch[q_idx*3+dim_idx] += F_ATBp_lim[q_i, dim_idx]

    @ti.kernel
    def construct_rhs_positional(self):
        for q_i in range(self.FIX_N):
            weight = self.positional_weight
            q_idx = self.fix_particle_ti[q_i]
            q_i_x, q_i_y = q_idx*self.dim, q_idx*self.dim+1
            self.rhs[q_i_x] += weight * self.node_pos_init[q_idx].x
            self.rhs[q_i_y] += weight * self.node_pos_init[q_idx].y

        # for i in range(self.CON_N):
        #     q_i = self.contact_particle_ti[i]
        #     self.rhs[q_i*2] += self.positional_weight * (self.node_pos[q_i].x + self.contact_vel[i].x * self.dt)
        #     self.rhs[q_i*2+1] += self.positional_weight * (self.node_pos[q_i].y + self.contact_vel[i].y * self.dt)

    def local_solve(self):
        self.rhs.fill(0.)
        self.rhs_stretch.fill(0.)
        self.construct_rhs_mass()
        self.construct_rhs_stretch()
        self.construct_rhs_positional()

    def substep(self, step_num:int):
        # 只计算，不更新位置（rhs没有被使用）
        self.construct_sn()
        self.warm_start()
        self.local_solve()

    # ========== difffpd ==========
    @ti.kernel
    def construct_dx_const(self):
        """解决Movement Constraint中dBp/dq为常数的情况
        """
        for q_i in range(self.PARTICLE_N):
            for d in ti.static(range(self.dim)):
                self.dx_const[q_i*self.dim+d] = self.node_mass[q_i] / self.dt**2

        # for q_i in ti.static(self.contact_particle_list):
        #     for d in ti.static(range(self.dim)):
        #         self.dx_const[q_i*self.dim+d] += self.positional_weight

    @ti.kernel
    def hessian_stretch(self):
        """ 计算diffpd中stretch constraint的二阶导数矩阵(Hessian)
        """
        self.dBp_stretch.fill(0.)
        dim = self.dim
        for f_i in range(self.ELEMENT_N):
            F_Ai = self.F_A[f_i]
            U, sig, V = self.ele_u[f_i], self.stretch_stress[f_i], self.ele_v[f_i]

            dBp_dF = ti.Matrix.zero(ti.f64, 4, 4)
            for m in range(2):
                for n in range(2):
                    Omega_uv = ti.Matrix.zero(ti.f64, 2, 2)
                    Omega_uv[0, 1] = (U[m,0]*V[n,1] - U[m,1]*V[n,0]) / (sig[0] + sig[1])
                    Omega_uv[1, 0] = -Omega_uv[0, 1]
                    dBp_df = U @ Omega_uv @ V.transpose()
                    dBp_dF[dim*m + n, :] = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[1, 0], dBp_df[1, 1]])

            idx1, idx2, idx3 = self.ele[f_i]
            for m, n in ti.ndrange(2, 2):                   # 2*2表示Bp*q的维数
                dBp_dF_i = ti.Matrix.zero(ti.f64, 2, 2)     # 2表示参数空间的维数
                for k, l in ti.ndrange(2, 2):
                    dBp_dF_i[k, l] = dBp_dF[2*m+k, 2*n+l]
                AT_dBp_dq_i = F_Ai.transpose() @ dBp_dF_i @ F_Ai    # A.T @ (dBp_x / dF_x) @ A
                AT_dBp_dq_i *= self.stretch_weight[f_i]

                row_idx_vec = ti.Vector([idx1*dim+m, idx2*dim+m, idx3*dim+m])
                col_idx_vec = ti.Vector([idx1*dim+n, idx2*dim+n, idx3*dim+n])
                for k, l in ti.ndrange(3, 3):               # AT_dBp_dq_i's dim
                    row_idx = row_idx_vec[k]
                    col_idx = col_idx_vec[l]
                    self.dBp_stretch[row_idx, col_idx] += AT_dBp_dq_i[k, l]

    def construct_g_hessian(self):
        """ 论文中g函数的Hessian矩阵dA = dBp/dq """
        self.hessian_stretch()
        self.dA = self.dBp_stretch.to_numpy()

    def model_gradient_const(self):
        """ 计算与接触点无关的部分A_pre和B_pre """
        mass_np = self.node_mass.to_numpy() / self.dt**2
        mass_matrix = np.diag(np.repeat(mass_np, 2))
        A_pre = mass_matrix + np.kron(self.lhs_stretch.to_numpy(), np.eye(2)) + \
                np.kron(self.lhs_positional.to_numpy(), np.eye(2)) - self.dA
        B_pre = np.diag(self.dx_const.to_numpy())

        return A_pre, B_pre


# 将候选接触点放在一个类里，统一计算deformation gradient
@ti.data_oriented
class Soft2D(Soft2DNocontact):
    def __init__(self, shape, fix:List[int], candi_contact:List[int], 
                 E:float, nu:float, dt:float, density:float, **kwargs):
        super().__init__(shape, fix, E, nu, dt, density, **kwargs)
        self.candi_contact = candi_contact
        self.CON_N:int = 1
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)

        self.marker_idx = [103, 104]     # 需要指定哪些节点
        self.marker_direction = [np.array([-1, 0]), np.array([1, 0])]

        self.precomputation()
        self.substep(0)
        self.construct_g_hessian()
        self.A_pre, self.B_pre = self.model_gradient_const()

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
    
    def compute_marker_jacobian(self, dq_dcontact):
        """ 计算marker点的雅可比矩阵
        Args:
            dq_dcontact (npt.NDArray[np.float64]): dim=2N*2，N为节点数量
        Returns:
            npt.NDArray[np.float64]: dim=2M*2，M为marker点数量
        """
        marker_idx = self.marker_idx
        marker_jacobian = np.zeros((len(marker_idx)*2, 2), dtype=np.float64)
        for i, idx in enumerate(marker_idx):
            marker_jacobian[i*2:i*2+2, :] = dq_dcontact[2*idx:2*idx+2, :]
        return marker_jacobian
    
    def construct_loss_gradient(self):
        idx1, idx2 = self.marker_idx[0], self.marker_idx[1]
        d1 = self.node_pos[idx2].to_numpy() - self.node_pos[idx1].to_numpy()
        loss = d1 @ d1

        # 以下均为手工推导的梯度表达式
        dd1_dmarker = np.kron(np.array([1, -1]), np.eye(2))
        grad = 2 * d1 @ dd1_dmarker
        return loss, grad
    
    def compute_afford(self, dq_dcontact):
        """ 通过变形雅可比矩阵计算当前接触条件的affordance值
        Returns:
            affordance (float): affordance value
        """
        loss, grad = self.construct_loss_gradient()
        dq_dmarker = self.compute_marker_jacobian(dq_dcontact)
        dloss_dcontact = grad @ dq_dmarker
        # 求解loss对action的梯度，取其范数作为affordance值（单接触点情况下直接简化）
        afford = np.linalg.norm(dloss_dcontact)
        return afford
    
    def compute_task_afford(self):
        """ 根据loss的构建和当前的接触条件计算affordance值 """
        results = {}
        candi_contact = self.candi_contact
        for idx in candi_contact:
            dq_dcontact = self.model_gradient(idx)
            afford = self.compute_afford(dq_dcontact)
            results[idx] = afford
            print(f"Contact idx: {idx}; Affordance value: {afford:.6f}")

        return results    


if __name__ == "__main__":
    # 测试Soft2D类
    mesh_file = OUTPUT_DIR / "mesh/output_mesh.msh"
    fix_list = list(range(0, 161, 16))
    candidate_contact_list = list(range(1, 15, 1)) + list(range(15, 176, 16)) + list(range(161, 175, 1))
    soft_model = Soft2D(shape=[0.15, 0.1], fix=fix_list, candi_contact=candidate_contact_list,
                        E=1.e3, nu=0.3, dt=1.e-2, density=1.e1)
    
    results = soft_model.compute_task_afford()
    output_file = ROOT_DIR / "affordance_results.json"

    try:
        with open(output_file, "w") as f:
            json.dump(results, f)
    except Exception as e:
        print(f"Error saving results: {e}")

    """
    不对结果进行可视化，如有需要在实验阶段补充：
    输入：Image -> Soft2D模型 -> affordance值
    1. 输入图像预处理：将图像转换为适合Soft2D模型的格式（如网格表示）。
    2. Soft2D模型计算：使用Soft2D类计算各候选接触点的affordance值。
    3. 结果可视化：将affordance值映射回图像空间，并使用heatmap进行可视化展示。
    """
