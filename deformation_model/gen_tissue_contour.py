"""
生成一个逼真的软组织二维 Mask 和轮廓坐标, 并基于此生成高质量的三角形网格
created on 2026-02-16
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import meshio
from scipy.interpolate import splprep, splev

from utilize.gen_mesh import MaskTo2DMesh
from utilize.mesh_io import write_mshv2_triangular
from const import MESH_DIR

def generate_tissue_mask(image_size=400, base_radius=120, noise=40, num_ctrl_points=7):
    """
    生成一个逼真的软组织二维 Mask 和轮廓坐标
    """
    # 1. 在极坐标下生成随机的控制点
    angles = np.linspace(0, 2 * np.pi, num_ctrl_points, endpoint=False)
    # 随机扰动半径，打造不规则形状
    radii = base_radius + np.random.uniform(-noise, noise, size=num_ctrl_points)
    
    # 闭合曲线：把首尾点连起来
    angles = np.append(angles, angles[0])
    radii = np.append(radii, radii[0])
    
    # 将极坐标转换为笛卡尔坐标，并平移到图像中心
    center = image_size / 2
    x = center + radii * np.cos(angles)
    y = center + radii * np.sin(angles)
    
    # 2. 使用 B 样条曲线进行平滑插值（这是让形状像“软组织”的关键）
    tck, _ = splprep([x, y], s=0, per=True)
    # 生成 200 个平滑的边缘点
    u_new = np.linspace(0, 1, 200)
    smooth_x, smooth_y = splev(u_new, tck)
    
    # 组合为 (N, 2) 的整数坐标矩阵，方便 OpenCV 和网格划分使用
    contour_points = np.vstack((smooth_x, smooth_y)).T.astype(np.int32)
    
    # 3. 在空白画布上绘制二值化 Mask
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], 255)
    
    return mask, contour_points

# ================= 测试运行 =================
if __name__ == "__main__":
    # 生成 Mask 和 轮廓点
    tissue_mask, contour = generate_tissue_mask()
    np.savetxt("contour_points.csv", contour, fmt="%d", delimiter=",")  # 保存轮廓点为 CSV 文件
    
    mesher = MaskTo2DMesh(
        boundary_resolution=50, 
        smooth_factor=2.0,       
        mesh_max_area=200.0,     
        mesh_min_angle=30.0
    )

    V, F = mesher.generate_mesh(tissue_mask)
    V = V * 1.e-3  # 缩放到合适尺寸

    mesh_file = f"{MESH_DIR}/pd_stretch_tissue_mesh_init_2.msh"
    write_mshv2_triangular(f"{mesh_file}", V, F)

    # 保存为 meshio 格式
    meshio.write(f"{mesh_file.replace('.msh', '.vtu')}", meshio.Mesh(points=V, cells=[("triangle", F)]))

    # 打印轮廓数据维度，这正是网格划分算法需要的输入格式 (N, 2)
    print(f"成功提取轮廓！点集维度: {contour.shape}") 

    # 可视化结果
    plt.figure(figsize=(8, 4))
    
    # 左图：展示生成的二值化 Mask
    plt.subplot(1, 2, 1)
    plt.title("Soft Tissue Mask")
    plt.imshow(tissue_mask, cmap='gray')
    plt.axis('off')
    
    # 右图：展示提取的二维边界点
    plt.subplot(1, 2, 2)
    plt.title("Extracted Contour Points")
    plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
    plt.gca().invert_yaxis() # 翻转 Y 轴以匹配图像坐标系
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig("tissue_mask_and_contour.svg", dpi=300)

    plt.figure(figsize=(6, 6))
    plt.title(f"2D Mesh (V: {len(V)}, F: {len(F)})")
    # 使用 matplotlib 内置的 triplot 画三角网格
    plt.triplot(V[:, 0], V[:, 1], F, color='blue', linewidth=0.5)
    plt.plot(V[:, 0], V[:, 1], 'ro', markersize=2) # 画出节点
    plt.gca().invert_yaxis() # OpenCV的 Y 轴是向下的，这里反转以匹配显示
    plt.axis('equal')

    # plt.tight_layout()
    plt.savefig("tissue_mesh.svg", dpi=300, bbox_inches='tight')