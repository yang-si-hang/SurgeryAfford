""" 使用pd采集不同的attachment节点的stretch数据；
增加静止时间步，确保稳态数据
Modified to save HDF5 data
Changelog:
    - 2025-01-12: 增加噪声采集

Date: 2025-11-20
"""
import cv2
import numpy as np
import meshio
from scipy import sparse
import h5py
from pathlib import Path
import taichi as ti

from const import DATA_DIR, OUTPUT_DIR, ROOT_DIR, MESH_DIR
from utilize.gen_mesh import MaskTo2DMesh
from deformation_model.diffpd_2d import Soft2D
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular
from datetime import datetime


NOISE_CFG = {
    "pos_sigma": 3.e-4,    # 位置传感标准差
    "force_sigma": 1.e-1,  # 力传感标准差
}

def generate_mesh_and_exit():
    """ 生成mesh并退出, 后续仿真直接使用msh文件 """
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
    neighbors = data_mesher.neighbors
    mesh_file = f"{MESH_DIR}/pd_stretch_demo_mesh_init.msh"
    write_mshv2_triangular(f"{mesh_file}", V, F)
    print(f"Mesh saved to {mesh_file} with {V.shape[0]} vertices and {F.shape[0]} faces.")
    np.savetxt(f"{MESH_DIR}/pd_stretch_demo_mesh_neighbors.csv", neighbors, fmt="%d", delimiter=",")
    print(f"Neighbors info saved in {MESH_DIR}/pd_stretch_demo_mesh_neighbors.csv")
    mesh_dict = {"V": V, "F": F, "E": E, "neighbors": neighbors}
    exit()

# generate_mesh_and_exit()

def simulate_deformation(
    mesh_file: str,
    contact_idx: int,
    fix_nodes: list,
    hard_ele_list: list,
    action_value: np.ndarray,
    action_step: int,
    save_data: bool = False
) -> dict:
    """
    执行形变仿真并返回稳态数据。
    
    Args:
        mesh_file: msh网格文件路径
        contact_idx: 接触/施加动作的节点索引
        hard_ele_list: 硬质区域的单元索引列表
        action_value: 施加的动作值 (如位移或速度)，需为 numpy array
        action_step: 外部施加动作的循环次数
        save_data: 是否将结果保存为 HDF5 和 MSH 文件
        
    Return:
        dict: 包含 contact_idx, internal_force (N步受力场), post_x (N步节点位置)
    """
    
    free_ele_list = []
    
    # 1. 初始化仿真器
    soft = Soft2D(
        shape=mesh_file, 
        fix=fix_nodes, 
        contact=contact_idx,
        E=1.e3, nu=0.3, dt=1.e-2, density=1.e1, print_info=save_data, damp=1.e-5
    )
    
    # 2. 异质材料配置
    stretch_w_np = soft.stretch_weight.to_numpy()
    for e_i in hard_ele_list:
        stretch_w_np[e_i] *= 100
    for e_i in free_ele_list:
        stretch_w_np[e_i] *= 0.01
    soft.stretch_weight.from_numpy(stretch_w_np)

    # 3. 预计算与矩阵分解
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    # 数据缓存
    data_buffer = {
        "post_x": [],         # 对应原代码的 q_curr (加噪后位置)
        "internal_force": [], # 对应原代码的 forces_field (加噪后受力)
        "action_val": [],
        "action_idx": []
    }

    print("--- Start Simulation & Data Collection ---")

    # 4. 核心仿真循环 (外层循环由 action_step 控制)
    for step in range(action_step):
        if save_data:
            print(f"Action step {step} " + "-"*10)

        # 确保动作维度匹配 Taichi 的接收格式
        if action_value.ndim == 1:
            action_value = np.expand_dims(action_value, axis=0) / soft.dt  # 转为 (1, 2) 形状
        soft.contact_vel.from_numpy(action_value/soft.dt) # 转换为速度输入
        soft.substep(step_num=0)

        # 清空速度输入，进入稳态衰减
        soft.contact_vel.fill(0.)
        for sub_step in range(200): # 内部稳态步数保持 200 不变
            soft.substep(step_num=sub_step+1)
            if sub_step > 190 and save_data: # 最后10步打印平均速度，确认已接近稳态
                vel_avg = np.linalg.norm(soft.node_vel.to_numpy(), axis=1).mean()
                print(f"  Substep {sub_step+1}, average node velocity: {vel_avg:.6e}")
            
        # 提取稳态物理量
        q_t = soft.node_pos.to_numpy()
        nodes_force = soft.force.to_numpy()

        # 添加噪声
        pos_noise = np.random.randn(*q_t.shape) * NOISE_CFG["pos_sigma"]
        q_t_noise = q_t.copy() + pos_noise

        nodes_force_noise = nodes_force.copy()
        nodes_force_noise[contact_idx, :] += np.random.randn(2) * NOISE_CFG["force_sigma"]

        # 存入 Buffer
        data_buffer["post_x"].append(q_t_noise)
        data_buffer["internal_force"].append(nodes_force_noise)
        data_buffer["action_val"].append(action_value)
        data_buffer["action_idx"].append(soft.contact_particle_list)

    # 将列表转换为 numpy arrays 以方便返回和后续计算
    post_x_arr = np.stack(data_buffer["post_x"])
    internal_force_arr = np.stack(data_buffer["internal_force"])

    # 5. 文件保存逻辑 (由 save_data 控制)
    if save_data:
        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(OUTPUT_DIR) / TIMESTAMP
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最后一步的 mesh
        write_mshv2_triangular(
            f"{out_dir}/pd_contact{contact_idx}_step{action_step:03d}.msh",
            q_t_noise, soft.ele.to_numpy()
        )
        meshio.write(f"{OUTPUT_DIR}/{TIMESTAMP}/pd_contact{contact_idx}_step{step+1:03d}.vtu",
                 meshio.Mesh(points=q_t_noise, cells=[("triangle", soft.ele.to_numpy())]))

        # 保存 HDF5
        h5_path = Path(DATA_DIR) / "demo" / "pd_stretch_data_hete" / TIMESTAMP / "0.hdf5"
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving data to {h5_path}...")
        with h5py.File(str(h5_path), 'w') as f:
            g_mesh = f.create_group('mesh_structure')
            g_mesh.create_dataset('faces', data=soft.ele.to_numpy(), compression="gzip")
            g_mesh.create_dataset('edges', data=soft.edge.to_numpy(), compression="gzip")
            g_mesh.create_dataset('rest_pos', data=soft.node_pos_init.to_numpy(), compression="gzip")
            g_mesh.create_dataset('fix_nodes', data=np.array(fix_nodes), compression="gzip")
            g_mesh.create_dataset('contact_nodes', data=np.array([contact_idx]), compression="gzip")
            g_mesh.create_dataset('hard_ele_idx', data=np.array(hard_ele_list), compression="gzip")
            g_mesh.create_dataset('free_ele_idx', data=np.array(free_ele_list), compression="gzip")
            g_mesh.create_dataset('stiffness_truth', data=soft.stretch_weight.to_numpy(), compression="gzip")

            f.attrs['E'] = 1.e3
            f.attrs['nu'] = 0.3
            f.attrs['dt'] = 1.e-2
            f.attrs['total_steps'] = action_step
            f.attrs['noise_pos_sigma'] = NOISE_CFG["pos_sigma"]
            f.attrs['noise_force_sigma'] = NOISE_CFG["force_sigma"]
            
            g_data = f.create_group('trajectories')
            g_data.create_dataset('q_curr', data=post_x_arr, compression="gzip")
            g_data.create_dataset('action_val', data=np.stack(data_buffer["action_val"]), compression="gzip")
            g_data.create_dataset('action_idx', data=np.stack(data_buffer["action_idx"]), compression="gzip")
            g_data.create_dataset('forces_field', data=internal_force_arr, compression="gzip")
            
        print("Data saved successfully.")

    # 6. 构建并返回结果字典
    return {
        "contact_idx": contact_idx,
        "internal_force": internal_force_arr,
        "post_x": post_x_arr
    }

if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=True, default_fp=ti.f64)

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(f"{OUTPUT_DIR}/{TIMESTAMP}").mkdir(parents=True, exist_ok=True)

    # fix 和 contact的索引使用paraview可视化选择
    # mesh_file = f"{MESH_DIR}/pd_stretch_demo_mesh_init.msh"
    # fix_nodes = [0] + list(range(45, 60)) + []   # + 23, 38, 11, 8
    # contact_node = 21   # 22, 15, 8, 37, 30, 36
    # # action_value = np.array([1., 1.]) * 0.004 # 每步的动作量
    # action_value = np.array([-0.00403961, 0.03979567]) / 10.
    # hard_ele_list = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
    # # hard_ele_list = [5, 6, 31, 86, 94, 124, 125, 131, 136, 140, 145, 151, 177, 
    # #                  179, 185, 213, 219, 252, 255]
    # free_ele_list = []

    mesh_file = f"{MESH_DIR}/pd_stretch_tissue_mesh_init_2.msh"
    # fix_nodes = list(range(8, 17)) + list(range(28, 35))
    fix_nodes = list(range(13, 20)) + list(range(27, 34))
    # contact_node = [23, 115]   # [46, 67], [23, 115]
    contact_node = [40, 76] # [3, 61], [4, 96]
    # action_value = np.array([-0.05, 0.05]) / 10
    action_value = np.array([[0.05954685, -0.00736087]]) / 10 # [-0.01959656, -0.05670971]
    action_value = np.tile(action_value, (len(contact_node), 1))
    # hard_ele_list = [48, 55, 63, 91, 94, 126, 128, 138, 174, 201, 204, 247, 250, 254, 255, 
    #                  256, 297, 327, 330, 379, 390, 391, 393, 396, 397, 401, 403, 412] + \
    #                 [238, 278, 287, 288, 359, 365, 366, 367, 370, 374, 414]
    hard_ele_list = [66, 125, 138, 193, 197, 211, 219, 231, 234, 280, 284, 285, 290, 
                     298, 301, 344, 345, 370, 376]
    free_ele_list = []

    #=== 规则化配置，用于测试 === 
    # mesh_file = [0.1, 0.1]
    # fix_nodes = list(range(0, 11))
    # contact_node = 110
    # hard_ele_list = []

    simulate_deformation(
        mesh_file=mesh_file,
        contact_idx=contact_node,
        fix_nodes=fix_nodes,
        hard_ele_list=hard_ele_list,
        action_value=action_value,
        action_step=10,
        save_data=True
    )

    exit()

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
        # "q_prev": [],
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

    for step in range(10):
        print(f"Action step {step}"+"-"*10)

        q_tm1 = soft.node_pos.to_numpy()
        current_action_idx = soft.contact_particle_list[0]

        # 此处要注意action与contact之间的
        soft.contact_vel.from_numpy(np.expand_dims(action_value, axis=0) / soft.dt)
        soft.substep(step_num=0)

        soft.contact_vel.fill(0.)
        for sub_step in range(200):
            soft.substep(step_num=sub_step+1)
            if sub_step > 190:
                vel_avg = np.linalg.norm(soft.node_vel.to_numpy(), axis=1).mean()
                print(f"  Substep {sub_step+1}, average node velocity: {vel_avg:.6e}")
            
        q_t = soft.node_pos.to_numpy()
        nodes_force = soft.force.to_numpy()

        pos_noise = np.random.randn(*q_t.shape) * NOISE_CFG["pos_sigma"]
        q_t_noise = q_t.copy() + pos_noise

        nodes_force_noise = nodes_force.copy()
        nodes_force_noise[contact_node, :] += np.random.randn(2) * NOISE_CFG["force_sigma"]

        # --- [存入 Buffer] ---
        # data_buffer["q_prev"].append(q_tm1)
        data_buffer["q_curr"].append(q_t_noise)
        data_buffer["action_val"].append(action_value) # 存原始动作值
        data_buffer["action_idx"].append(current_action_idx)
        data_buffer["forces_field"].append(nodes_force_noise)

    write_mshv2_triangular(f"{OUTPUT_DIR}/{TIMESTAMP}/pd_contact{soft.contact_particle_list[0]}_step{step+1:03d}.msh",
                        q_t_noise, soft.ele.to_numpy())

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
        f.attrs['total_steps'] = len(data_buffer["q_curr"])
        f.attrs['description'] = "Simulation of soft tissue stretching"

        f.attrs['noise_pos_sigma'] = NOISE_CFG["pos_sigma"]
        f.attrs['noise_force_sigma'] = NOISE_CFG["force_sigma"]
        
        # 3. 保存轨迹数据 (Converting lists to numpy arrays)
        # 最终 shape 示例: q_prev -> (20, N, 2)
        g_data = f.create_group('trajectories')
        
        # g_data.create_dataset('q_prev', data=np.stack(data_buffer["q_prev"]), compression="gzip")
        g_data.create_dataset('q_curr', data=np.stack(data_buffer["q_curr"]), compression="gzip")
        g_data.create_dataset('action_val', data=np.stack(data_buffer["action_val"]), compression="gzip")
        g_data.create_dataset('action_idx', data=np.stack(data_buffer["action_idx"]), compression="gzip")
        g_data.create_dataset('forces_field', data=np.stack(data_buffer["forces_field"]), compression="gzip")

    print("Data saved successfully.")