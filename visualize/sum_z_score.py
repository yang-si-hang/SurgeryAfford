"""
Sum each exploration data and visualization

Date: 2026-04-22
"""
import numpy as np
from pathlib import Path
import os
import glob
from const import *


FILE_DIR = "data/demo/real_track/05030806"
prefix = "zscore_shift_per_sample"          # ← 替换为你需要的前缀

npy_files = glob.glob(os.path.join(FILE_DIR, f"{prefix}*.npy"))
npy_files.sort()  # 保证顺序一致

print(f"共找到 {len(npy_files)} 个匹配文件:")
for f in npy_files:
    print(f"  {f}")

arrays = []
for npy_file in npy_files:
    arr = np.load(npy_file)       # .npy 直接返回 ndarray
    arrays.append(arr)
    print(f"  {os.path.basename(npy_file)}: shape={arr.shape}, dtype={arr.dtype}")

# 沿 axis=0 拼接（假设每个文件的 array 维度一致，除第 0 维外）
all_arrays = np.concatenate(arrays, axis=0)
print(f"\n合并后 array shape: {all_arrays.shape}, dtype: {all_arrays.dtype}")

# --- 逐元素统计（沿第 0 维，保留后续维度） ---
elem_mean = np.mean(all_arrays, axis=0)
elem_var  = np.var(all_arrays, axis=0)
elem_std  = np.std(all_arrays, axis=0)

print(f"\n逐元素均值 shape: {elem_mean.shape}")
print(f"逐元素方差 shape: {elem_var.shape}")

# --- Visualization ---
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

mesh_file = np.load(Path(FILE_DIR) / "tissue_mesh.npz")
vertices = np.asarray(mesh_file["vertices"], dtype=np.float64)
faces = np.asarray(mesh_file["faces"], dtype=np.int32)

node_pos = np.asarray(vertices)[:, :2]
triangles = np.asarray(faces).astype(np.int32)
triang = mtri.Triangulation(node_pos[:, 0], node_pos[:, 1], triangles)

vmax = float(np.max((elem_mean)))
if vmax <= 0.0:
    vmax = 1.0
vmin = float(np.min(elem_mean))
print(f"value min: {vmin}; max: {vmax}")
np.savetxt(OUTPUT_DIR / "zscore_mean.csv", elem_mean, delimiter=",")

fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
trip = ax.tripcolor(
    triang,
    facecolors=elem_mean,
    cmap="RdBu_r",
    edgecolors="k",
    vmin=vmin,
    vmax=4,
)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Mean Z-Score Shift")
cbar = fig.colorbar(trip, ax=ax, pad=0.01, shrink=0.8, aspect=20, fraction=0.02)
cbar.set_label("z_model - z_real")
fig.savefig(f"{VISUALIZATION_DIR/'zscore_shift_mean.svg'}", dpi=300)
print(f"Saved per-sample z-score shift to {VISUALIZATION_DIR / 'zscore_shift_mean.svg'}")
plt.close(fig)