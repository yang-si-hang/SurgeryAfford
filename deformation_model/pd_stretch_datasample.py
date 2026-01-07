""" 使用pd采集不同的attachment节点的stretch数据；
增加静止时间步，确保稳态数据
Modified to save HDF5 data

Date: 2025-11-20
"""
import cv2
import numpy as np
from scipy import sparse
import h5py
from pathlib import Path
import taichi as ti

from const import DATA_DIR, OUTPUT_DIR, ROOT_DIR, MESH_DIR
from utilize.gen_mesh import MaskTo2DMesh
from deformation_model.diffpd_2d import Soft2D
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
Path(f"{OUTPUT_DIR}/{TIMESTAMP}").mkdir(parents=True, exist_ok=True)


def generate_mesh_and_exit():
    """ 生成mesh并退出，后续仿真直接使用msh文件 """
    FACTOR = 1.e-3

    # 使用确定的mask生成mesh
    H, W = 512, 512
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (200, 200), 1, -1)
    data_mesher = MaskTo2DMesh(boundary_resolution=60, mesh_max_area=50, mesh_min_angle=25)

    print(f"="*10+" Generating mesh from mask... "+"="*10)
    V, F = data_mesher.generate_mesh(mask)
    V = V * FACTOR  # 缩放到合适尺寸
    E = data_mesher.E # 边界点索引
    mesh_file = f"{MESH_DIR}/pd_stretch_demo_mesh_init.msh"
    write_mshv2_triangular(f"{mesh_file}", V, F)
    print(f"Mesh saved to {mesh_file} with {V.shape[0]} vertices and {F.shape[0]} faces.")
    mesh_dict = {"V": V, "F": F, "E": E}
    exit()

# generate_mesh_and_exit()

if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True, default_fp=ti.f64)

    # fix 和 contact的索引使用paraview可视化选择
    mesh_file = f"{MESH_DIR}/pd_stretch_demo_mesh_init.msh"
    fix_nodes = [0] + list(range(45, 60))
    contact_node = 15
    hard_ele_list = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
    free_ele_list = [201, 203, 223, 227, 240, 241]

    # 规则化配置，用于测试
    # mesh_file = [0.1, 0.1]
    # fix_nodes = list(range(0, 11))
    # contact_node = 110
    # hard_ele_list = []

    # 读取网格文件
    soft = Soft2D(shape=mesh_file, fix=fix_nodes, contact=contact_node,
                        E=1.e3, nu=0.3, dt=1.e-2, density=1.e1, damp=1.e-4)
    
    # 增加不同的硬度区域
    stretch_w_np = soft.stretch_weight.to_numpy()
    for e_i in hard_ele_list:
        stretch_w_np[e_i] *= 100
    for e_i in free_ele_list:
        stretch_w_np[e_i] *= 0.01
    soft.stretch_weight.from_numpy(stretch_w_np)

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    data_buffer = {
        "q_prev": [],
        "q_curr": [],
        "action_val": [],
        "action_idx": [],
        "forces_field": []
    }

    # 保存初始网格拓扑（只取一次即可，因为拓扑不变）
    mesh_faces = soft.ele.to_numpy()
    mesh_edges = soft.edge.to_numpy()
    mesh_rest_pos = soft.node_pos.to_numpy() # 初始位置

    print("Start Simulation & Data Collection...")

    for step in range(20):
        print(f"Action step {step}"+"-"*10)

        q_tm1 = soft.node_pos.to_numpy()
        action_value = np.array([-1., 1.0]) * 0.001 / soft.dt
        current_action_idx = soft.contact_particle_list[0]

        # 此处要注意action与contact之间的对应
        soft.contact_vel.from_numpy(np.array([[-1., 1.0]]) * 0.001 / soft.dt)
        soft.substep(step_num=0)

        soft.contact_vel.fill(0.)
        for sub_step in range(200):
            soft.substep(step_num=sub_step+1)
            if sub_step > 190:
                vel_avg = np.linalg.norm(soft.node_vel.to_numpy(), axis=1).mean()
                print(f"  Substep {sub_step+1}, average node velocity: {vel_avg:.6e}")
            
        q_t = soft.node_pos.to_numpy()
        nodes_force = soft.force.to_numpy()

        write_mshv2_triangular(f"{OUTPUT_DIR}/{TIMESTAMP}/pd_contact{soft.contact_particle_list[0]}_step{step+1:03d}.msh",
                               soft.node_pos.to_numpy(), soft.ele.to_numpy())

        # --- [存入 Buffer] ---
        data_buffer["q_prev"].append(q_tm1)
        data_buffer["q_curr"].append(q_t)
        data_buffer["action_val"].append(action_value) # 存原始动作值
        data_buffer["action_idx"].append(current_action_idx)
        data_buffer["forces_field"].append(nodes_force)

    # --- [保存为 HDF5] ---
    h5_path = Path(DATA_DIR) / "demo" / "pd_stretch_data_hete" / TIMESTAMP / "0.hdf5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path = str(h5_path)

    print(f"Saving data to {h5_path}...")
    
    with h5py.File(h5_path, 'w') as f:
        # 1. 保存通用的 Mesh 结构 (所有 step 共享)
        g_mesh = f.create_group('mesh_structure')
        g_mesh.create_dataset('faces', data=mesh_faces, compression="gzip")
        g_mesh.create_dataset('edges', data=mesh_edges, compression="gzip")
        g_mesh.create_dataset('rest_pos', data=mesh_rest_pos, compression="gzip")
        g_mesh.create_dataset('fix_nodes', data=np.array(fix_nodes), compression="gzip")
        g_mesh.create_dataset('contact_nodes', data=np.array([contact_node]), compression="gzip")
        g_mesh.create_dataset('hard_ele_idx', data=np.array(hard_ele_list), compression="gzip")
        g_mesh.create_dataset('free_ele_idx', data=np.array(free_ele_list), compression="gzip")
        g_mesh.create_dataset('stiffness_truth', data=soft.stretch_weight.to_numpy(), compression="gzip")

        # 2. 保存仿真参数 (Metadata)
        f.attrs['E'] = 1.e3
        f.attrs['nu'] = 0.3
        f.attrs['dt'] = 1.e-2
        f.attrs['total_steps'] = len(data_buffer["q_prev"])
        f.attrs['description'] = "Simulation of soft tissue stretching"
        
        # 3. 保存轨迹数据 (Converting lists to numpy arrays)
        # 最终 shape 示例: q_prev -> (20, N, 2)
        g_data = f.create_group('trajectories')
        
        g_data.create_dataset('q_prev', data=np.stack(data_buffer["q_prev"]), compression="gzip")
        g_data.create_dataset('q_curr', data=np.stack(data_buffer["q_curr"]), compression="gzip")
        g_data.create_dataset('action_val', data=np.stack(data_buffer["action_val"]), compression="gzip")
        g_data.create_dataset('action_idx', data=np.stack(data_buffer["action_idx"]), compression="gzip")
        g_data.create_dataset('forces_field', data=np.stack(data_buffer["forces_field"]), compression="gzip")

    print("Data saved successfully.")