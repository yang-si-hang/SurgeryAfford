import torch
import meshio
import numpy as np
import re
import collections
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from const import MESH_DIR, OUTPUT_DIR
from utilize.mesh_io import read_mshv2_triangular, write_mshv2_triangular

# ========== 核心 Dataset 类 ==========
class SofaMSHDataset(Dataset):
    """
    一个自定义数据集，用于加载成对的 .msh 文件。

    它会扫描一个目录，根据文件名 f"sofa_contact{ID}_step{NUM}.msh" 来
    自动配对 (step, step+1) 的文件，并确保这些配对只发生在
    相同的 contact_id 内部。
    """
    def __init__(self, data_directory: str, rule: str):
        """
        初始化数据集，扫描并配对所有 msh 文件。

        Args:
            data_directory (str): 包含 .msh 文件的根目录。
            rule (str): 用于数据处理的规则。
        """
        super().__init__()
        self.data_directory = Path(data_directory)
        self.sample_pairs = []
        self.rule = rule

        # 1. 定义正则表达式来解析文件名
        # 匹配 "sofa_contact" + "数字" + "_step" + "3位数字" + ".msh"
        self.file_pattern = re.compile(self.rule)
        
        # 2. 扫描文件并按 contact_id 分组
        data_map = collections.defaultdict(list)
        
        print(f"正在扫描目录: {self.data_directory}")
        # .rglob('*') 会递归扫描所有子目录
        for filepath in self.data_directory.rglob('*.msh'):
            match = self.file_pattern.search(filepath.name)
            
            if match:
                contact_id = int(match.group(1))
                step = int(match.group(2))
                data_map[contact_id].append((step, filepath))

        # 3. 在每个组内创建有效的 (pre, post) 对
        for contact_id, files in data_map.items():
            # 按 step 排序，确保时间顺序
            files.sort(key=lambda x: x[0])
            
            # 遍历文件列表，查找连续的 (step, step+1) 对
            for i in range(len(files) - 1):
                step_pre, file_pre = files[i]
                step_post, file_post = files[i+1]
                
                # 检查 step 是否连续
                if step_post == step_pre + 1:
                    # 这是一个有效的样本对
                    self.sample_pairs.append({
                        'pre_file': file_pre,
                        'post_file': file_post,
                        'contact_id': contact_id,
                        'pre_step': step_pre
                    })
        
        if not self.sample_pairs:
            print("警告: 未找到任何有效的 (step, step+1) msh 文件对。")
        else:
            print(f"成功找到 {len(self.sample_pairs)} 个有效样本对。")

    def __len__(self) -> int:
        """返回找到的有效样本对的总数。"""
        return len(self.sample_pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        加载一个样本对 (pre_x, post_x, contact_idx, action)。
        """
        # 1. 获取文件路径信息
        pair_info = self.sample_pairs[idx]
        pre_file_path = pair_info['pre_file']
        post_file_path = pair_info['post_file']
        contact_idx = pair_info['contact_id']
        
        # 2. 加载 MSH 文件中的节点位置
        pre_x = self.load_msh_positions(pre_file_path)
        post_x = self.load_msh_positions(post_file_path)

        action = post_x[contact_idx] - pre_x[contact_idx]

        # 3. 转换为 Torch Tensors
        return {
            'pre_x': torch.tensor(pre_x, dtype=torch.float32),
            'post_x': torch.tensor(post_x, dtype=torch.float32),
            'contact_idx': int(contact_idx),
            'action': torch.tensor(action, dtype=torch.float32)
        }

    def load_msh_positions(self, filepath: Path) -> np.ndarray:
        nodes, faces = read_mshv2_triangular(filepath)
        return nodes

# ========== DataLoader 创建函数 ==========
def get_dataloader(data_dir: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    一个辅助函数，用于创建和返回 DataLoader。
    始终使用 Mini-Batch 并设置 `shuffle=True`。
    """
    dataset = SofaMSHDataset(data_directory=data_dir, rule=r"pd_contact(\d+)_step(\d{3})\.msh")
    
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查您的 data_directory 和文件命名。")
    
    # 确定是否是 Full-Batch
    if batch_size >= len(dataset):
        print("检测到 Full-Batch 模式 (batch_size >= dataset size)。")
        print("在这种模式下，'shuffle' 参数无效。")
        shuffle = False # 设为 False 以避免 DataLoader 警告
        batch_size = len(dataset) # 确保只出一个 batch

    print(f"创建 DataLoader: Batch Size={batch_size}, Shuffle={shuffle}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, # 在 Windows 和 macOS 上设为 0 通常更稳定
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":

    file_path = OUTPUT_DIR
    try:
        mini_batch_loader = get_dataloader(data_dir=file_path, batch_size=5, shuffle=True)
        
        for i, batch in enumerate(mini_batch_loader):
            print(f"Batch {i+1}:")
            print(f"  pre_x 形状:  {batch['pre_x'].shape}")
            print(f"  post_x 形状: {batch['post_x'].shape}")
            print(f"  action 形状: {batch['action'].shape}")
        
    except ValueError as e:
        print(f"测试失败: {e}")

    print("\n--- 测试完成 ---")