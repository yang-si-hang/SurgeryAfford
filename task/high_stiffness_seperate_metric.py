import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from scipy.stats import wasserstein_distance
from collections import defaultdict

from utilize.mesh_io import read_mshv2_triangular
from const import MESH_DIR, OUTPUT_DIR, VISUALIZATION_DIR

def generate_mock_data():
    """
    生成模拟数据：一个矩形区域内的三角形网格，
    中心有一个高刚度圆形区域。
    """
    # 1. 生成节点
    x = np.linspace(0, 10, 30)
    y = np.linspace(0, 10, 30)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # 2. 生成三角形拓扑
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    elements = tri.triangles
    
    # 3. 计算单元重心
    centers_x = points[elements, 0].mean(axis=1)
    centers_y = points[elements, 1].mean(axis=1)
    
    # 4. 定义真值 (Ground Truth): 中心半径为3的圆为高刚度区
    radius = 3.0
    center_pos = np.array([5.0, 5.0])
    dist = np.sqrt((centers_x - center_pos[0])**2 + (centers_y - center_pos[1])**2)
    gt_labels = (dist < radius).astype(int) # 1 for inclusion, 0 for background
    
    # 5. 生成估计刚度 (Estimated Stiffness)
    # 背景刚度 ~ N(10, 2), 夹杂刚度 ~ N(20, 5)
    # 并在边界处添加一些模糊/平滑效果
    stiffness = np.zeros_like(gt_labels, dtype=float)
    stiffness[gt_labels == 0] = np.random.normal(10, 2, size=np.sum(gt_labels==0))
    stiffness[gt_labels == 1] = np.random.normal(20, 3, size=np.sum(gt_labels==1))
    
    # 简单平滑一下，模拟反演结果的模糊性
    # (实际工程中不需要这一步，直接用反演结果即可)
    
    return points, elements, stiffness, gt_labels

def get_boundary_gradients(points, elements, stiffness, gt_labels):
    """
    计算真值边界附近的刚度变化率
    """
    # 构建边到单元的映射来寻找邻居
    # key: tuple(sorted(node_indices)), value: list of element_indices
    edge_to_elems = defaultdict(list)
    
    for eid, elem in enumerate(elements):
        edges = [
            tuple(sorted((elem[0], elem[1]))),
            tuple(sorted((elem[1], elem[2]))),
            tuple(sorted((elem[2], elem[0])))
        ]
        for edge in edges:
            edge_to_elems[edge].append(eid)
            
    boundary_gradients = []
    
    # 遍历所有边，寻找连接 GT=0 和 GT=1 的边
    for edge, eids in edge_to_elems.items():
        if len(eids) == 2: # 内部边
            idx1, idx2 = eids
            # 检查是否跨越真值边界
            if gt_labels[idx1] != gt_labels[idx2]:
                # 计算刚度差
                k_diff = abs(stiffness[idx1] - stiffness[idx2])
                # k_diff = (np.abs(stiffness[idx1] - stiffness[idx2]) / (stiffness[idx1] + stiffness[idx2])) * 2
                # k_diff = np.abs(np.log10(stiffness[idx1]) - np.log10(stiffness[idx2]))
                
                # 计算重心距离 (可选，如果不除以距离则是单纯的刚度跳变)
                p1 = points[elements[idx1]].mean(axis=0)
                p2 = points[elements[idx2]].mean(axis=0)
                dist = np.linalg.norm(p1 - p2)
                
                gradient = k_diff / dist if dist > 0 else 0
                boundary_gradients.append(gradient)
                
    return np.array(boundary_gradients)

def analyze_stiffness_separability(points, elements, stiffness, gt_edges, gt_labels):
    """
    主分析函数：计算指标并绘图
    """

    fig = plt.figure(figsize=(18, 6))
    
    # --- 0. 绘制原始分布图 ---
    ax0 = fig.add_subplot(1, 3, 1)
    tripcolor = ax0.tripcolor(points[:,0], points[:,1], elements, facecolors=stiffness, cmap="RdYlBu_r", edgecolors='k')
    ax0.axis('off')
    # ax0.set_title("Covarai Map\n(with GT Boundary)")
    ax0.set_aspect('equal')
    cbar0 = fig.colorbar(tripcolor, ax=ax0, fraction=0.046, pad=0.04, shrink=0.8, aspect=20)
    cbar0.ax.tick_params(length=0, labelsize=12)

    # tick_locs = np.arange(np.ceil(np.min(stiffness)), np.floor(np.max(stiffness))+1)
    # cbar0.set_ticks(tick_locs)
    # cbar0.set_ticklabels([f"$10^{{ {int(loc):d} }}$" for loc in tick_locs])
    
    # 绘制真值边界
    ax0.tricontour(points[:,0], points[:,1], elements, gt_edges, levels=[1], colors='black', linewidths=3, linestyles='dashed')

    # --- 1. AUC-ROC ---
    ax1 = fig.add_subplot(1, 3, 2)
    fpr, tpr, thresholds_roc = roc_curve(gt_labels, stiffness)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- 2. F1-score Curve ---
    ax2 = fig.add_subplot(1, 3, 3)
    # F1 score 需要遍历阈值
    # 使用 precision_recall_curve 计算不同阈值下的 P 和 R
    precisions, recalls, thresholds_pr = precision_recall_curve(gt_labels, stiffness)
    
    # 计算 F1 (注意: precisions/recalls 比 thresholds 多一个元素)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    # 移除最后一个仅仅用于绘图的元素以匹配 thresholds
    f1_scores = f1_scores[:-1] 
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds_pr[best_idx]
    best_f1 = f1_scores[best_idx]

    ax2.plot(thresholds_pr, f1_scores, color='green', lw=2, label='F1 Score')
    ax2.axvline(best_thresh, color='r', linestyle='--', alpha=0.5, label=f'Best Thresh={best_thresh:.1f}')
    ax2.scatter(best_thresh, best_f1, color='red')
    ax2.set_xlabel('Stiffness Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'F1-score Curve (Max = {best_f1:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 3. Wasserstein Distance (EMD) ---
    # ax3 = fig.add_subplot(2, 3, 4)
    
    vals_pos = stiffness[gt_labels == 1]
    vals_neg = stiffness[gt_labels == 0]
    
    w_dist = wasserstein_distance(vals_pos, vals_neg)
    print(f"Wasserstein Distance between Inclusion and Background: {w_dist:.4f}")
    
    # 绘制直方图展示分布差异
    # ax3.hist(vals_neg, bins=30, alpha=0.5, label='Background', density=True, color='blue')
    # ax3.hist(vals_pos, bins=30, alpha=0.5, label='Inclusion', density=True, color='red')
    # ax3.set_title(f"Stiffness Distributions\nWasserstein Dist = {w_dist:.2f}")
    # ax3.set_xlabel("Stiffness Value")
    # ax3.legend()
    
    # --- 4. Boundary Stiffness Gradient ---
    # ax4 = fig.add_subplot(2, 3, 5)
    gradients = get_boundary_gradients(points, elements, stiffness, gt_labels)
    print(f"Mean Stiffness Gradient at GT Boundary: {np.mean(gradients):.4f} ± {np.std(gradients):.4f}")
    
    # if len(gradients) > 0:
    #     ax4.hist(gradients, bins=20, color='purple', alpha=0.7, edgecolor='black')
    #     ax4.set_title(f"Gradient at GT Boundary\nMean={np.mean(gradients):.2f}, Std={np.std(gradients):.2f}")
    #     ax4.set_xlabel("Stiffness Change per Unit Length")
    #     ax4.set_ylabel("Count of Boundary Edges")
    # else:
    #     ax4.text(0.5, 0.5, "No Boundary Detected", ha='center')

    # 布局调整
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{VISUALIZATION_DIR}/stiffness_separability_analysis.svg", dpi=300)
    print(f"Analysis figure saved to {VISUALIZATION_DIR}/stiffness_separability_analysis.svg")

# --- 执行代码 ---
if __name__ == "__main__":
    mesh_file = f"{MESH_DIR}/pd_stretch_demo_mesh_init.msh"
    node_np, ele_np = read_mshv2_triangular(mesh_file)

    tri = mtri.Triangulation(node_np[:, 0], node_np[:, 1], ele_np)
    elements = tri.triangles

    # variance metric
    variance_vec = np.loadtxt(f"{OUTPUT_DIR}/P_diag_ekf.csv", delimiter=",")
    k_vals = np.log10(np.sqrt(variance_vec))

    # EKF value metric
    value_vec = np.loadtxt(f"{OUTPUT_DIR}/estimated_stiffness_ekf.csv", delimiter=",")
    k_vals = np.log10(value_vec)

    # # Gauss-Newton value metric
    value_vec = np.loadtxt(f"{OUTPUT_DIR}/stretch_weight_update.csv", delimiter=",")
    k_vals = value_vec

    hard_ele_list = [151, 174, 176, 177, 178, 179, 182, 186, 218, 219, 220, 306]
    # hard_ele_list = [5, 6, 31, 86, 94, 124, 125, 131, 136, 140, 145, 151, 177, 
    #                  179, 185, 213, 219, 252, 255]
    # hard_ele_list = [48, 55, 63, 91, 94, 126, 128, 138, 174, 201, 204, 247, 250, 254, 255, 
    #                  256, 297, 327, 330, 379, 390, 391, 393, 396, 397, 401, 403, 412] + \
    #                 [238, 278, 287, 288, 359, 365, 366, 367, 370, 374, 414]
    gt = np.zeros(node_np.shape[0], dtype=int)
    for e_i in hard_ele_list:
        id1, id2, id3 = ele_np[e_i]
        gt[id1] = 1
        gt[id2] = 1
        gt[id3] = 1

    gt_labels = np.zeros(ele_np.shape[0], dtype=int)
    gt_labels[hard_ele_list] = 1
    
    # 2. 分析与绘图
    analyze_stiffness_separability(node_np, elements, k_vals, gt, gt_labels)