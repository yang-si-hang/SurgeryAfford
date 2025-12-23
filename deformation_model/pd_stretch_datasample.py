""" 使用pd采集不同的attachment节点的stretch数据；
增加静止时间步，确保稳态数据
Modified to save HDF5 data

Date: 2025-11-20
"""
import numpy as np
from scipy import sparse
import h5py
from pathlib import Path
import taichi as ti

from const import DATA_DIR, OUTPUT_DIR, ROOT_DIR
from deformation_model.diffpd_2d import Soft2D
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular


if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True, default_fp=ti.f64)

    contact_node = 120 # 77, 115， 120

    soft = Soft2D(shape=[0.1, 0.1], fix=list(range(0, 11)), contact=contact_node,
                        E=1.e3, nu=0.3, dt=1.e-2, density=1.e1)
    
    write_mshv2_triangular(f"{OUTPUT_DIR}/pd_contact{soft.contact_particle_list[0]}_step{0:03d}.msh",
                           soft.node_pos.to_numpy(), soft.ele.to_numpy())
    
    # 增加不同的硬度区域
    hard_ele_list = [120, 121, 122, 123, 132, 133, 136, 137]
    stretch_w_np = soft.stretch_weight.to_numpy()
    for e_i in hard_ele_list:
        stretch_w_np[e_i] *= 100  # 初始拉伸权重调大一些，方便观察效果
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
    mesh_topology = soft.ele.to_numpy()
    mesh_rest_pos = soft.node_pos.to_numpy() # 初始位置

    print("Start Simulation & Data Collection...")

    for step in range(20):
        print(f"Action step {step} -----")

        q_tm1 = soft.node_pos.to_numpy()
        action_value = np.array([1., 1.0]) * 0.001 / soft.dt
        current_action_idx = soft.contact_particle_list[0]

        # 此处要注意action与contact之间的对应
        soft.contact_vel.from_numpy(np.array([[1., 1.0]]) * 0.001 / soft.dt)
        soft.substep(step_num=0)

        soft.contact_vel.fill(0.)
        for sub_step in range(200):
            soft.substep(step_num=sub_step+1)
            if sub_step > 190:
                vel_avg = np.linalg.norm(soft.node_vel.to_numpy(), axis=1).mean()
                print(f"  Substep {sub_step+1}, average node velocity: {vel_avg:.6e}")
            
        q_t = soft.node_pos.to_numpy()
        nodes_force = soft.force.to_numpy()

        write_mshv2_triangular(f"{OUTPUT_DIR}/pd_contact{soft.contact_particle_list[0]}_step{step+1:03d}.msh",
                               soft.node_pos.to_numpy(), soft.ele.to_numpy())

        # --- [存入 Buffer] ---
        data_buffer["q_prev"].append(q_tm1)
        data_buffer["q_curr"].append(q_t)
        data_buffer["action_val"].append(action_value) # 存原始动作值
        data_buffer["action_idx"].append(current_action_idx)
        data_buffer["forces_field"].append(nodes_force)

    # --- [保存为 HDF5] ---
    h5_path = Path(DATA_DIR) / "demo" / "pd_stretch_data_hete" / "0.hdf5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path = str(h5_path)

    print(f"Saving data to {h5_path}...")
    
    with h5py.File(h5_path, 'w') as f:
        # 1. 保存通用的 Mesh 结构 (所有 step 共享)
        g_mesh = f.create_group('mesh_structure')
        g_mesh.create_dataset('faces', data=mesh_topology, compression="gzip")
        g_mesh.create_dataset('rest_pos', data=mesh_rest_pos, compression="gzip")
        
        # 2. 保存仿真参数 (Metadata)
        f.attrs['E'] = 1.e3
        f.attrs['nu'] = 0.3
        f.attrs['dt'] = 1.e-2
        f.attrs['total_steps'] = len(data_buffer["q_prev"])
        
        # 3. 保存轨迹数据 (Converting lists to numpy arrays)
        # 最终 shape 示例: q_prev -> (20, N, 2)
        g_data = f.create_group('trajectories')
        
        g_data.create_dataset('q_prev', data=np.stack(data_buffer["q_prev"]), compression="gzip")
        g_data.create_dataset('q_curr', data=np.stack(data_buffer["q_curr"]), compression="gzip")
        g_data.create_dataset('action_val', data=np.stack(data_buffer["action_val"]), compression="gzip")
        g_data.create_dataset('action_idx', data=np.stack(data_buffer["action_idx"]), compression="gzip")
        g_data.create_dataset('forces_field', data=np.stack(data_buffer["forces_field"]), compression="gzip")

    print("Data saved successfully.")