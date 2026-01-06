"""
HDF5 数据加载器，用于pd仿真数据的读取和批处理
假设数据已经按照 deformation_model/pd_stretch_datasample.py 中的逻辑保存为 HDF5 格式
Data: 2025-12-17
"""
import torch
import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os

from const import DATA_DIR 


# ========== 核心 Dataset 类 ==========
class HDF5PdDataset(Dataset):
    """
    用于加载 HDF5 格式的仿真数据。
    每个 HDF5 文件可能包含多个时间步 (Trajectories)，
    该 Dataset 会将所有文件的所有时间步展平为单独的样本。
    """
    def __init__(self, data_directory: str, load_to_ram: bool = True):
        """
        Args:
            data_directory (str): 包含 .h5 文件的文件夹路径。
            load_to_ram (bool): 是否将所有数据预加载到内存。
                                针对数据量 < 100 对的情况，强烈建议 True，
                                可以显著加快训练速度并避免 HDF5 并发读取错误。
        """
        super().__init__()
        self.data_directory = Path(data_directory)
        self.samples = []
        self.mesh_data = {
            'F': None,
            'E': None,
            'V': None,
        }
        self.static_data = {
            'fix_nodes': None,
            'stiffness_truth': None
        }
        
        # 扫描所有 h5 文件
        h5_files = list(self.data_directory.rglob('*.hdf5'))
        
        if not h5_files:
            print(f"警告: 在 {self.data_directory} 下未找到 .hdf5 文件。")
            return

        print(f"正在扫描目录: {self.data_directory}，发现 {len(h5_files)} 个 HDF5 文件。")

        for file_path in h5_files:
            try:
                self._load_file(file_path, load_to_ram)
            except Exception as e:
                print(f"读取文件 {file_path.name} 失败: {e}")

        print(f"数据加载完成，共 {len(self.samples)} 个样本对。")

    def _load_file(self, file_path, load_to_ram):
        """读取单个 HDF5 文件并提取数据"""
        # 使用 'r' 模式读取
        with h5py.File(file_path, 'r') as f:
            # 1. 读取静态网格信息 (仅在第一次读取时加载)
            if self.mesh_data['F'] is None:
                g_mesh = f['mesh_structure']
                self.mesh_data['F'] = g_mesh['faces'][:]
                self.mesh_data['E'] = g_mesh['edges'][:]
                self.mesh_data['V'] = g_mesh['rest_pos'][:]

                self.static_data['fix_nodes'] = g_mesh['fix_nodes'][:]
                self.static_data['stiffness_truth'] = g_mesh['stiffness_truth'][:]
                
                # 读取全局属性
                self.E = f.attrs.get('E', None)
                self.nu = f.attrs.get('nu', None)
            
            current_contact_node = f['mesh_structure/contact_nodes'][:]

            # 2. 获取轨迹数据组
            grp = f['trajectories']
            num_steps = grp['q_prev'].shape[0]
            
            if load_to_ram:
                # 一次性读取整个数组到内存 (numpy array)
                # 这种方式比在循环里切片读取要快得多
                q_prev_all = grp['q_prev'][:]
                q_curr_all = grp['q_curr'][:]
                act_val_all = grp['action_val'][:]
                act_idx_all = grp['action_idx'][:]
                forces_field_all = grp['forces_field'][:]
                
                for t in range(num_steps):
                    self.samples.append({
                        'pre_x': q_prev_all[t],         # (N, 2)
                        'post_x': q_curr_all[t],        # (N, 2)
                        'action': act_val_all[t],       # (2,)
                        'contact_idx': act_idx_all[t],  # (1,)
                        'force': forces_field_all[t],    # (N, 2)
                        'step_idx': t,                  # 当前是第几步
                        'source_file': file_path.name   # 来源文件名
                    })
            else:
                # 如果数据极大，无法放入内存，则只存索引
                for t in range(num_steps):
                    self.samples.append({
                        'file_info': (str(file_path), t),
                        'contact_node': current_contact_node 
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        返回一个样本字典。
        """
        sample_data = self.samples[idx]

        # 如果 sample_data 是元组 (path, t)，说明没有预加载内存 (Lazy Loading)
        # 针对你的小数据集，通常走上面的 load_to_ram 逻辑，这里是防御性代码
        if isinstance(sample_data, tuple):
            file_path, t = sample_data
            with h5py.File(file_path, 'r') as f:
                grp = f['trajectories']
                item = {
                    'pre_x': torch.tensor(grp['q_prev'][t], dtype=torch.float32),
                    'post_x': torch.tensor(grp['q_curr'][t], dtype=torch.float32),
                    'action': torch.tensor(grp['action_val'][t], dtype=torch.float32),
                    'contact_idx': int(grp['action_idx'][t]), 
                    'force': torch.tensor(grp['action_force'][t], dtype=torch.float32),
                    'step_idx': torch.tensor(t, dtype=torch.long),
                }
                return item

        # 内存模式直接转换 Tensor
        # contact_idx 在 HDF5 中可能是数组形式 (1,)，需要转为 int 或 scalar tensor
        c_idx = sample_data['contact_idx']
        if isinstance(c_idx, (np.ndarray, list)):
            c_idx = c_idx.item() # 转为 python scalar
        
        return {
            'pre_x': torch.tensor(sample_data['pre_x'], dtype=torch.float32),
            'post_x': torch.tensor(sample_data['post_x'], dtype=torch.float32),
            'action': torch.tensor(sample_data['action'], dtype=torch.float32),
            'contact_idx': torch.tensor(c_idx, dtype=torch.long), # 索引通常用 long
            'force': torch.tensor(sample_data['force'], dtype=torch.float32),
            'step_idx': torch.tensor(sample_data['step_idx'], dtype=torch.long),
        }


# ========== DataLoader 创建函数 ==========
def get_dataloader(data_dir: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    创建 DataLoader。
    """
    dataset = HDF5PdDataset(data_directory=data_dir, load_to_ram=True)
    
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查路径。")
    
    # 检测 Full-Batch
    if batch_size >= len(dataset):
        print(f"检测到 Full-Batch (batch_size {batch_size} >= dataset {len(dataset)})")
        shuffle = False
        batch_size = len(dataset)

    print(f"创建 DataLoader: Batch Size={batch_size}, Shuffle={shuffle}")
    
    # 这里的 num_workers=0 是安全的，因为我们已经预加载数据到内存了，
    # 避免了 Windows/Mac 上 HDF5 跨进程读取的经典坑。
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True
    )
    
    return loader


# ========== 测试部分 ==========
if __name__ == "__main__":
    # 确保有一个可以测试的路径
    test_dir = Path(DATA_DIR) / "demo" / "pd_stretch_data"
    
    if not os.path.exists(test_dir):
        print(f"目录 {test_dir} 不存在，无法进行读取测试。请先运行保存数据的脚本。")
    else:
        try:
            # 1. 实例化 Dataset (以便获取 mesh info)
            ds = HDF5PdDataset(test_dir)
            
            if len(ds) > 0:
                # 获取一次静态拓扑
                rest_pos, faces = ds.get_mesh_topology()
                print(f"\n[Mesh Info]")
                print(f"  Nodes (Rest): {rest_pos.shape}")
                print(f"  Faces: {faces.shape} (用于构建 Edge Index)")

                # 2. 获取 Loader
                loader = get_dataloader(data_dir=test_dir, batch_size=4, shuffle=True)
                
                # 3. 模拟训练循环
                print("\n[Iterate Batch]")
                for i, batch in enumerate(loader):
                    print(f"Batch {i+1}:")
                    print(f"  pre_x:       {batch['pre_x'].shape}")       # (B, N, 2)
                    print(f"  post_x:      {batch['post_x'].shape}")      # (B, N, 2)
                    print(f"  action:      {batch['action'].shape}")      # (B, 2)
                    print(f"  contact_idx: {batch['contact_idx'].shape}") # (B) or (B, 1)
                    print(f"  force:       {batch['force'].shape}")       # (B, 2)
                    
                    # 打印第一个样本的 contact_idx 验证读取正确性
                    print(f"  -> Sample 0 contact_idx: {batch['contact_idx'][0]}")
                    break # 只测一个 batch
            else:
                print("数据集为空。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"测试出错: {e}")

    print("\n--- 测试完成 ---")