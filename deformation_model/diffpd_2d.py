""" 
输入三角形网格，使用projective dynamics创建变形模型，并得到deformation jacobian；
本文件中的Soft2D类作为diffpd的基础类，供其他模块调用

---------------------------------------------------------------------
Change Log:
    2025-11-24：重新修改了diffpd的求解方式（与理论推导一致）
    2025-11-25: 增加了damping项的支持（pd和diffpd）
---------------------------------------------------------------------
Date: 2025-11-4
"""
import os
# # --- 关键代码：在 import numpy 之前设置 ---
# # 限制 OpenBLAS (NumPy 在很多系统上的默认后端)
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# # 限制 MKL (Intel MKL 后端)
# os.environ["MKL_NUM_THREADS"] = "1"
# # 限制 OMP (OpenMP)
# os.environ["OMP_NUM_THREADS"] = "1"
# # 限制 NumPy 内部的线程池 (较新版本)
# os.environ["NUMPY_NUM_THREADS"] = "1"
# # ----------------------------------------

import time
from pathlib import Path
from typing import List
import numpy as np
from scipy import sparse
import taichi as ti

from const import ROOT_DIR, MESH_DIR
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular
from utilize.mesh_util import extract_edge_from_face, mesh_obj_tri

# def read_msh(file:str):
#     nodes, faces = read_mshv2_triangular(file)

#     # 构建边列表
#     edges = []
#     for face in faces:
#         edge1 = sorted([face[0], face[1]])
#         edge2 = sorted([face[1], face[2]])
#         edge3 = sorted([face[2], face[0]])
        
#         edges.append(edge1)
#         edges.append(edge2)
#         edges.append(edge3)
    
#     # Remove duplicate edges by converting to tuples, using a set, then back to list
#     unique_edges = []
#     edge_set = set()
#     for edge in edges:
#         edge_tuple = tuple(edge)
#         if edge_tuple not in edge_set:
#             edge_set.add(edge_tuple)
#             unique_edges.append(edge)
    
#     edges = np.array(unique_edges, dtype=int)

#     return nodes, edges, faces

@ti.data_oriented
class Soft2D:
    """ 基于triangular mesh，以及fix & contact nodes' indices构建的PD变形模型 """
    def __init__(self, shape, fix:List[int], contact:int, 
                 E:float, nu:float, dt:float, density:float, **kwargs):
        self.shape = shape

        if isinstance(self.shape, Path):
            node_np, ele_np = read_mshv2_triangular(self.shape)
            edge_np = extract_edge_from_face(ele_np)
        elif isinstance(self.shape, List):
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.01)
            # node_np_3d = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))

            msh_file:str = MESH_DIR / "shape.msh"
            write_mshv2_triangular(msh_file, node_np, ele_np)
        elif isinstance(self.shape, dict):
            node_np = np.array(self.shape['V'])
            edge_np = np.array(self.shape['E'])
            ele_np = np.array(self.shape['F'])

        damp:float = kwargs.get('damp', 5.e-3)
        self.solve_itr:int = 10
        self.dt = dt
        self.E, self.nu, self.damp, self.density = E, nu, damp*np.sqrt(E), density
        self.dim = 2
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
       
        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        # topology fields
        self.node_pos      = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new  = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_vel      = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.force         = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
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

        # material fields
        self.stretch_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.positional_weight = 0.         # define later

        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.ATA = ti.Matrix.field(3, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # F_A^T * F_A
        self.ele_u = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)          # singular value decomposition
        self.ele_v = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.Bp_shear = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)       # stretch part
        self.stretch_stress = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)

        self.sn  = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_mass = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_positional = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_damp = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.rhs_stretch = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.pre_fact_lhs_solve = None

        self.mass_matrix:np.ndarray = None
        self.g_hessian:np.ndarray = None
        self.dBp_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.PARTICLE_N*2))
        self.dA = None
        self.dforce_dw_mat = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)  # gradient of internal force wrt. stretch weight
        self.dfa_dy = ti.field(ti.f64, shape=self.PARTICLE_N)          # fa: attachment force
        self.dfd_dy = ti.field(ti.f64, shape=self.PARTICLE_N)          # fd: damping force

        self.fix_particle_list = fix
        self.contact_particle_list = [int(contact)]
        self.FIX_N = len(self.fix_particle_list)
        self.CON_N = 1  # 默认只有一个接触
        self.fix_particle_ti     = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))
        self.contact_particle_ti.from_numpy(np.array(self.contact_particle_list).astype(np.int32))
        self.contact_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.CON_N)
        self.contact_vel.fill(0.)

        self.construct_mass()
        self.construct_Xg_inv()
        self.positional_weight = 1.e5 * self.node_mass_sum[None] / self.PARTICLE_N / self.dt**2
        # self.construct_dx_const()

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
        self.lhs_mass.fill(0.)
        for q_i in range(self.PARTICLE_N):
            self.lhs[q_i, q_i] += self.node_mass[q_i] / self.dt**2
            self.lhs_mass[q_i, q_i] += self.node_mass[q_i] / self.dt**2

    @ti.kernel
    def construct_lhs_stretch(self):
        # https://medium.com/@victorlouisdg/jax-cloth-tutorial-part-1-e7a0e285864f
        self.lhs_stretch.fill(0.)
        for f_i in range(self.ELEMENT_N):
            Xg_inv = self.Xg_inv[f_i]
            a, b, c, d = Xg_inv[0, 0], Xg_inv[0, 1], Xg_inv[1, 0], Xg_inv[1, 1]

            # F_A's dim=4*6，F_A @ q = F.T，flatten的时候左侧行优先，而F列优先
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
            self.ATA[f_i] = ATA

            stretch_weight = self.stretch_weight[f_i]
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]
                self.lhs_stretch[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]

    @ti.kernel
    def construct_lhs_positional(self):
        self.lhs_positional.fill(0.)
        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight
            self.lhs_positional[q_i, q_i] += self.positional_weight

        for i in range(self.CON_N):
            q_i = self.contact_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight
            self.lhs_positional[q_i, q_i] += self.positional_weight

    @ti.kernel
    def construct_lhs_damp(self):
        """ 全局阻尼直接衰减速度 """
        self.lhs_damp.fill(0.)
        for q_i in range(self.PARTICLE_N):
            self.lhs[q_i, q_i] += self.damp / self.dt * self.node_mass[q_i]
            self.lhs_damp[q_i, q_i] += self.damp / self.dt * self.node_mass[q_i]

    def precomputation(self):
        self.lhs.fill(0.)
        self.construct_lhs_mass()
        self.construct_lhs_stretch()
        self.construct_lhs_positional()
        self.construct_lhs_damp()

    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for q_i in range(self.PARTICLE_N):
            idx1, idx2 = dim*q_i, dim*q_i+1
            self.sn[idx1] = self.node_pos[q_i].x + self.node_vel[q_i].x * dt
            self.sn[idx2] = self.node_pos[q_i].y + self.node_vel[q_i].y * dt

        # Contact particles update
        for idx in range(self.CON_N):
            q_i = self.contact_particle_ti[idx]
            self.sn[q_i*dim] = self.node_pos[q_i].x + self.contact_vel[idx].x * dt
            self.sn[q_i*dim + 1] = self.node_pos[q_i].y + self.contact_vel[idx].y * dt

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
        self.rhs_stretch.fill(0.)
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

        for i in range(self.CON_N):
            q_i = self.contact_particle_ti[i]
            self.rhs[q_i*2] += self.positional_weight * (self.node_pos[q_i].x + self.contact_vel[i].x * self.dt)
            self.rhs[q_i*2+1] += self.positional_weight * (self.node_pos[q_i].y + self.contact_vel[i].y * self.dt)

    @ti.kernel
    def construct_rhs_damp(self):
        for q_i in range(self.PARTICLE_N):
            idx1, idx2 = q_i*self.dim, q_i*self.dim+1
            self.rhs[idx1] += self.damp / self.dt * self.node_mass[q_i] * self.node_pos[q_i].x
            self.rhs[idx2] += self.damp / self.dt * self.node_mass[q_i] * self.node_pos[q_i].y

    @ti.kernel
    def update_internal_force(self):
        self.force.fill(0.)
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]

            F_AT = self.F_A[f_i].transpose()
            self.dforce_dw_mat[f_i] = - F_AT @ (self.F[f_i].transpose() - self.Bp_shear[f_i].transpose())

            force_mat = self.stretch_weight[f_i] * self.dforce_dw_mat[f_i]
            self.force[idx1] += force_mat[0,:]
            self.force[idx2] += force_mat[1,:]
            self.force[idx3] += force_mat[2,:]

    def local_solve(self):
        self.rhs.fill(0.)
        self.construct_rhs_mass()
        self.construct_rhs_stretch()
        self.construct_rhs_positional()
        self.construct_rhs_damp()
        self.update_internal_force()

    @ti.kernel
    def update_pos_new(self, sol_x:ti.types.ndarray(), sol_y:ti.types.ndarray()):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = sol_x[q_i]
            self.node_pos_new[q_i].y = sol_y[q_i]

    @ti.kernel
    def update_vel_pos(self):   
        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_pos_new[q_idx] = self.node_pos[q_idx] + self.contact_vel[idx] * self.dt

        for i in range(self.PARTICLE_N):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]

        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.node_pos[q_i] = self.node_pos_init[q_i]
            self.node_vel[q_i] = ti.Vector([0., 0.], dt=ti.f64)

        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_vel[q_idx] = ti.Vector([0., 0.], dt=ti.f64)

    def substep(self, step_num:int):
        self.construct_sn()
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            # print(f"Iteration: {itr} ------------------------------------")
            self.local_solve()
            rhs_np = self.rhs.to_numpy()
            # print(f"Rhs:\n{self.rhs_stretch.to_numpy().reshape(-1, 2)}")
            # Split rhs_np into x,y components
            rhs_np_x = rhs_np[0::2]
            rhs_np_y = rhs_np[1::2]

            node_pos_new_np_x = self.pre_fact_lhs_solve(rhs_np_x)
            node_pos_new_np_y = self.pre_fact_lhs_solve(rhs_np_y)

            self.update_pos_new(node_pos_new_np_x, node_pos_new_np_y)
            # print(f"Node pos new:\n", self.node_pos_new.to_numpy().reshape(-1, 2))
        
        self.update_vel_pos()

    # ========== difffpd ========== #
    def construct_mass_matrix(self):
        mass_np = self.node_mass.to_numpy() / self.dt**2
        mass_matrix = np.diag(np.repeat(mass_np, 2))
        self.mass_matrix = mass_matrix

    @ti.kernel
    def hessian_stretch(self):
        """计算diffpd中stretch constraint的二阶导数矩阵(Hessian)
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
            # 与方法construct_rhs_stretch中Bp_shear一致
            dBp_dF = dBp_dF.transpose()    # 转置以符合后续计算需求，转置似乎无变化（存疑，但无法理论证明）

            idx1, idx2, idx3 = self.ele[f_i]
            for m, n in ti.ndrange(2, 2):                   # 2*2表示Bp的维数*q的维数
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
        """ 构建diffpd中g函数的Hessian矩阵 """
        # A = self.mass_matrix + np.kron(self.lhs_stretch.to_numpy(), np.eye(2)) + \
        #     np.kron(self.lhs_positional.to_numpy(), np.eye(2)) - self.dA
        A = np.kron(self.lhs.to_numpy(), np.eye(2)) - self.dA
        self.g_hessian = A

    def construct_E_hessian(self):
        """ 论文中能量函数的Hessian矩阵，dA = dBp/dq """
        # self.construct_dx_const()
        self.construct_mass_matrix()
        self.hessian_stretch()
        self.dA = self.dBp_stretch.to_numpy()
        self.construct_g_hessian()

    def model_z(self, dloss:np.ndarray):
        """ 计算diffpd中的z向量：（\partial^2 g）@ z = dloss/dq_(n+1)  """
        A = self.g_hessian
        b = dloss
        
        z = np.linalg.solve(A, b)
        return z

    @ti.kernel
    def attach_gradient_pos(self):
        """ Movement Constraint中的df_int/dx（其他约束没有） """
        self.dfa_dy.fill(0.)
        for q_i in ti.static(self.contact_particle_list):
            self.dfa_dy[q_i] += self.positional_weight

    @ti.kernel
    def damp_gradient_pos(self):
        """ 阻尼项中的df_int/dx """
        self.dfd_dy.fill(0.)
        for q_i in range(self.PARTICLE_N):
            self.dfd_dy[q_i] += self.damp / self.dt * self.node_mass[q_i]
                
    def contact_jacobian(self):
        """ 计算变形雅可比矩阵dq(n+1)/da(n)中contact相关的列 """
        A = self.g_hessian
        self.attach_gradient_pos()
        self.damp_gradient_pos()
        B = self.mass_matrix + np.diag(np.repeat(self.dfa_dy.to_numpy(), 2)) \
             + np.diag(np.repeat(self.dfd_dy.to_numpy(), 2))
        contact_idx = self.contact_particle_list[0]
        dq_dy_np = np.linalg.solve(A, B[:,2*contact_idx:2*contact_idx+2])
        return dq_dy_np


@ti.data_oriented
class Soft2DAfford(Soft2D):
    """ 用于affordance计算的Soft2D子类 """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.marker_idx = [103, 104]     # 需要指定哪些节点

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
        self.construct_E_hessian()
        dq_dcontact = self.contact_jacobian()
        afford = self.compute_afford(dq_dcontact)
        return dq_dcontact, afford


if __name__ == "__main__":
    ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

    soft = Soft2D(shape=[0.1, 0.1], fix=list(range(11)), contact=120,
                  E=1.e3, nu=0.3, dt=1.e-2, density=1.e1, damp=5.e0)
    
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    np.savetxt(f"lhs_stretch_mat_gt.csv", soft.lhs_stretch.to_numpy(), fmt="%.6f", delimiter=",")
    exit()

    mean_avg_list = []
    soft.contact_vel[0] = 0.001 * ti.Vector([1.0, 1.0], dt=ti.f64) / soft.dt
    for step in range(10):
        soft.substep(step_num=step)
        print(f"Average velocity at step {step}: {np.mean(np.linalg.norm(soft.node_vel.to_numpy(), axis=1)):.6e}")
        mean_avg_list.append(np.mean(np.linalg.norm(soft.node_vel.to_numpy(), axis=1)))

    soft.contact_vel.fill(0.)
    for step in range(500):
        soft.substep(step_num=step)
        print(f"Average velocity at step {step}: {np.mean(np.linalg.norm(soft.node_vel.to_numpy(), axis=1)):.6e}")
        mean_avg_list.append(np.mean(np.linalg.norm(soft.node_vel.to_numpy(), axis=1)))

    # np.savetxt(f"node_pos_damp_0.csv", soft.node_pos.to_numpy(), fmt="%.6f", delimiter=",")
    # write_mshv2_triangular("deform.msh", soft.node_pos.to_numpy(), soft.ele.to_numpy())
    
    import matplotlib.pyplot as plt

    # 新增：使用非交互后端并绘制 mean_avg_list 曲线，保存为 SVG（不显示）
    try:
        import matplotlib
        matplotlib.use("Agg")  # 确保使用无显示后端
    except Exception:
        pass

    # mean_avg_list 在上面已被填充
    try:
        if 'mean_avg_list' in globals() and len(mean_avg_list) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(mean_avg_list)), mean_avg_list, marker='o', linestyle='-', color='tab:blue')
            ax.set_xlabel('Step')
            ax.set_ylabel('Mean velocity')
            ax.set_title('Mean average velocity over time')
            ax.grid(True, linestyle=':', alpha=0.6)
            # 保存到 MESH_DIR，确保目录存在
            try:
                out_dir = Path(MESH_DIR)
            except Exception:
                out_dir = Path('.')
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = f"mean_avg_list_{time.strftime('%Y%m%d-%H%M%S')}.svg"
            fig.savefig(str(out_file), format='svg')
            plt.close(fig)
            print(f"Saved mean_avg_list plot to: {out_file}")
        else:
            print("mean_avg_list is empty or not found; skip plotting.")
    except Exception as e:
        print(f"Failed to generate/save mean_avg_list plot: {e}")

    exit(0)

    mesh_file = MESH_DIR / "output_mesh.msh"
    fix_list = list(range(0, 161, 16))
    contact_idx = 15
    soft_model = Soft2DAfford(shape=[0.15, 0.1], fix=fix_list, contact=contact_idx,
                              E=1.e3, nu=0.3, dt=1.e-2, density=1.e1)
    # soft_model = Soft2DAfford(shape=[150, 100], fix=fix_list, contact=contact_idx,
    #                           E=1.e3, nu=0.3, dt=1.e-2, density=1.e1)

    soft_model.precomputation()
    lhs_np = soft_model.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_model.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    soft_model.substep(step_num=0)
    dq_dy, afford = soft_model.compute_task_afford()
    print(f"Contact idx: {contact_idx}; Affordance value: {afford:.6f}")
    # np.savetxt("deformation_jacobian.csv", dq_dy, fmt="%.6f", delimiter=",")