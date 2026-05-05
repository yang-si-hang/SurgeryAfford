"""
Analyze strain mismatch on real experimental trajectories.

Date: 2026-04-22

This script compares model-predicted strain against measured strain for each
frame of a real HDF5 tracking sequence and visualizes the mean per-element
shift on the triangular mesh.

Workflow:
1. Load tracked node positions from the configured HDF5 file.
2. Load the reference mesh from tissue_mesh.npz in the same directory.
3. Build the action at each frame from the contact-node displacement relative
   to frame 0.
4. Apply that action to the model over ACTION_DURATION and compute model-side
   Green-Lagrange strain.
5. Compute real-side Green-Lagrange strain directly from the measured node
   positions of the same frame.
6. Compute modified z-scores using median and MAD only on internal_cell_list.
7. Set modified z-score and shift to zero on all non-internal elements.
8. Average the per-frame shift field and save both the numeric output and the
   mesh visualization.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import meshio
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy import sparse
import taichi as ti
import torch

from const import DATA_DIR, OUTPUT_DIR
from deformation_model.diffpd_2d import Soft2D


DEFAULT_DATA_DIR = DATA_DIR / "demo" / "real_track" / "05030806"
HDF5_FILE = "test-4-cut_data.hdf5"
SCALE = 6.0e-4
FIXED_NODES = list(range(30, 43)) + list(range(58, 63))
boudary_node_list = list(range(0, 63))
CONTACT_IDX = [20, 21]  # 4,160 22,165 12,174
RESULT_DIR = OUTPUT_DIR / "strain_zscore_shift"
INIT_STIFFNESS_VALUE = 400.0
ACTION_DURATION = 1.0
MAX_SAMPLES = None
STD_EPS = 1e-12
MODIFIED_ZSCORE_SCALE = 0.6745


@ti.data_oriented
class Soft2DForce(Soft2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strain_val = ti.field(dtype=ti.f64, shape=(self.ELEMENT_N,))

    @ti.kernel
    def cal_deformation_gradient(self):
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos[idx1], self.node_pos[idx2], self.node_pos[idx3]
            x_f = ti.Matrix.cols([b - a, c - a])
            f_i_mat = ti.cast(x_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = f_i_mat

            u_mat, sig_mat, v_mat = ti.svd(f_i_mat, ti.f64)
            self.ele_u[f_i] = u_mat
            self.ele_v[f_i] = v_mat
            self.stretch_stress[f_i] = ti.Vector([sig_mat[0, 0], sig_mat[1, 1]], dt=ti.f64)
            self.Bp_shear[f_i] = u_mat @ ti.Matrix([[1.0, 0.0], [0.0, 1.0]], ti.f64) @ v_mat.transpose()

    @ti.kernel
    def reconstruct_stretch_weight(self, stretch_weight: ti.types.ndarray()):
        for f_i in range(self.ELEMENT_N):
            self.stretch_weight[f_i] = stretch_weight[f_i] * self.ele_volume[f_i]

    @ti.kernel
    def gl_strain(self):
        for f_i in range(self.ELEMENT_N):
            green_lagrange = 0.5 * (self.F[f_i].transpose() @ self.F[f_i] - ti.Matrix.identity(ti.f64, 2))
            self.strain_val[f_i] = ti.sqrt((green_lagrange ** 2).sum())

    @ti.kernel
    def reset_state(self):
        for i in range(self.PARTICLE_N):
            self.node_pos[i] = self.node_pos_init[i]
            self.node_pos_new[i] = self.node_pos_init[i]
            self.node_vel[i] = ti.Vector([0.0, 0.0], dt=ti.f64)

        for i in range(self.CON_N):
            self.contact_vel[i] = ti.Vector([0.0, 0.0], dt=ti.f64)


def extract_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edge_set = set()
    for face in np.asarray(faces, dtype=np.int32):
        i, j, k = (int(face[0]), int(face[1]), int(face[2]))
        edge_set.add(tuple(sorted((i, j))))
        edge_set.add(tuple(sorted((j, k))))
        edge_set.add(tuple(sorted((k, i))))
    return np.asarray(sorted(edge_set), dtype=np.int32)


def normalize_action(action: Union[torch.Tensor, np.ndarray], contact_count: int) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float64)
    if action_np.ndim == 1:
        action_np = np.expand_dims(action_np, axis=0)

    if action_np.shape[-1] != 2:
        raise ValueError(f"Action must have shape (*, 2), but got {action_np.shape}.")

    if action_np.shape[0] == 1 and contact_count > 1:
        action_np = np.repeat(action_np, contact_count, axis=0)

    if action_np.shape[0] != contact_count:
        raise ValueError(
            f"Action contact dimension {action_np.shape[0]} does not match contact count {contact_count}."
        )

    return action_np


def modified_zscore(vec: np.ndarray, eps: float = STD_EPS) -> np.ndarray:
    median_val = float(np.median(vec))
    mad_val = float(np.median(np.abs(vec - median_val)))
    if mad_val <= eps:
        return np.zeros_like(vec)
    return MODIFIED_ZSCORE_SCALE * (vec - median_val) / mad_val


def masked_modified_zscore(vec: np.ndarray, active_indices: np.ndarray, eps: float = STD_EPS) -> np.ndarray:
    masked_score = np.zeros_like(vec, dtype=np.float64)
    if active_indices.size == 0:
        return masked_score
    masked_score[active_indices] = modified_zscore(np.asarray(vec, dtype=np.float64)[active_indices], eps=eps)
    return masked_score


def load_real_experiment(data_dir: Path, hdf5_file: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    hdf5_path = data_dir / hdf5_file
    mesh_path = data_dir / "tissue_mesh.npz"

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        positions = h5_file["trajectory_data/positions"][:] * SCALE
        frames = h5_file["trajectory_data/frame_indices"][:]

    mesh_file = np.load(mesh_path)
    vertices = np.asarray(mesh_file["vertices"], dtype=np.float64) * SCALE
    faces = np.asarray(mesh_file["faces"], dtype=np.int32)
    edges = np.asarray(mesh_file["edges"], dtype=np.int32) if "edges" in mesh_file.files else extract_edges_from_faces(faces)

    mesh_data = {
        "V": vertices,
        "F": faces,
        "E": edges,
    }
    validate_real_inputs(positions, frames, mesh_data)
    return positions, frames, mesh_data


def validate_real_inputs(positions: np.ndarray, frames: np.ndarray, mesh_data: Dict[str, np.ndarray]) -> None:
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions with shape (T, N, 2), but got {positions.shape}.")
    if frames.ndim != 1:
        raise ValueError(f"Expected frame indices with shape (T,), but got {frames.shape}.")
    if len(frames) != positions.shape[0]:
        raise ValueError(f"Frame count {len(frames)} does not match trajectory length {positions.shape[0]}.")

    node_count = positions.shape[1]
    mesh_node_count = int(mesh_data["V"].shape[0])
    if mesh_node_count != node_count:
        raise ValueError(
            f"Trajectory node count {node_count} does not match mesh node count {mesh_node_count}."
        )

    if not CONTACT_IDX:
        raise ValueError("CONTACT_IDX must not be empty.")

    invalid_contact = [idx for idx in CONTACT_IDX if idx < 0 or idx >= node_count]
    if invalid_contact:
        raise ValueError(f"CONTACT_IDX contains invalid node indices: {invalid_contact}")

    invalid_fixed = [idx for idx in FIXED_NODES if idx < 0 or idx >= node_count]
    if invalid_fixed:
        raise ValueError(f"FIXED_NODES contains invalid node indices: {invalid_fixed}")

    overlap = sorted(set(CONTACT_IDX).intersection(FIXED_NODES))
    if overlap:
        raise ValueError(f"CONTACT_IDX overlaps with FIXED_NODES: {overlap}")


def build_model(
    mesh_data: Dict[str, np.ndarray],
    fixed_nodes: List[int],
    contact_idx: List[int],
    init_stiffness: np.ndarray,
) -> Soft2DForce:
    soft_model = Soft2DForce(
        shape=mesh_data,
        fix=fixed_nodes,
        contact=contact_idx,
        E=1.0e1,
        nu=0.3,
        dt=1.0e-2,
        density=1.0e1,
        print_info=False,
    )
    soft_model.reconstruct_stretch_weight(init_stiffness)
    soft_model.precomputation()
    soft_model.pre_fact_lhs_solve = sparse.linalg.factorized(sparse.csc_matrix(soft_model.lhs.to_numpy()))
    return soft_model


def compute_model_strain(soft_model: Soft2DForce, action_value: np.ndarray) -> np.ndarray:
    soft_model.reset_state()
    action_steps = max(1, int(round(ACTION_DURATION / soft_model.dt)))
    contact_velocity = action_value / (action_steps * soft_model.dt)
    soft_model.contact_vel.from_numpy(contact_velocity)
    for step_idx in range(action_steps):
        soft_model.substep(step_num=step_idx)
    soft_model.contact_vel.fill(0.0)
    soft_model.cal_deformation_gradient()
    soft_model.gl_strain()
    return soft_model.strain_val.to_numpy()


def compute_real_strain(soft_model: Soft2DForce, node_pos: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    node_pos_np = np.asarray(node_pos, dtype=np.float64)
    soft_model.reset_state()
    soft_model.node_pos.from_numpy(node_pos_np[:, :2])
    soft_model.node_pos_new.from_numpy(node_pos_np[:, :2])
    soft_model.cal_deformation_gradient()
    soft_model.gl_strain()
    return soft_model.strain_val.to_numpy()


def plot_mean_shift(mesh_data: Dict[str, np.ndarray], mean_shift: np.ndarray, output_path: Path) -> None:
    node_pos = np.asarray(mesh_data["V"])[:, :2]
    triangles = np.asarray(mesh_data["F"]).astype(np.int32)
    triang = mtri.Triangulation(node_pos[:, 0], node_pos[:, 1], triangles)

    vmax = float(np.max((mean_shift)))
    if vmax <= 0.0:
        vmax = 1.0
    vmin = float(np.min(mean_shift))
    print(f"value min: {vmin}; max: {vmax}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    trip = ax.tripcolor(
        triang,
        facecolors=mean_shift,
        cmap="RdBu_r",
        edgecolors="k",
        vmin=-3,
        vmax=3,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mean Modified Z-Score Shift")
    cbar = fig.colorbar(trip, ax=ax, pad=0.01, shrink=0.8, aspect=20, fraction=0.046)
    cbar.set_label("mz_model - mz_real")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_real_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    positions, frames, mesh_data = load_real_experiment(data_dir, HDF5_FILE)
    element_count = int(mesh_data["F"].shape[0])
    init_stiffness = np.full((element_count,), INIT_STIFFNESS_VALUE, dtype=np.float64)
    soft_model = build_model(mesh_data, FIXED_NODES, CONTACT_IDX, init_stiffness)
    cells = [("triangle", mesh_data["F"].astype(np.int32))]

    frame0_contact_pos = positions[0, CONTACT_IDX, :2].copy()
    shift_list = []

    
    internal_cell_list = []
    for i, face_nodes in enumerate(mesh_data["F"]):
        if not any(v in boudary_node_list for v in face_nodes):
            internal_cell_list.append(i)
    internal_cell_indices = np.asarray(internal_cell_list, dtype=np.int32)
    if internal_cell_indices.size == 0:
        raise ValueError("internal_cell_list is empty, cannot compute masked modified z-score.")

    sample_count = positions.shape[0] if MAX_SAMPLES is None else min(int(positions.shape[0]), int(MAX_SAMPLES))
    print(f"Processing {sample_count} frames from {data_dir / HDF5_FILE} ...")
    print(f"Using CONTACT_IDX={CONTACT_IDX} and FIXED_NODES={FIXED_NODES}")
    print(f"Using {internal_cell_indices.size} internal cells for modified z-score and shift.")

    for sample_idx in range(sample_count):
        action_value = positions[sample_idx, CONTACT_IDX, :2] - frame0_contact_pos
        action_value = normalize_action(action_value, len(CONTACT_IDX))

        model_strain = compute_model_strain(soft_model, action_value)
        model_pos = soft_model.node_pos.to_numpy()
        
        real_strain = compute_real_strain(soft_model, positions[sample_idx])
        real_pos = soft_model.node_pos.to_numpy()

        z_model = masked_modified_zscore(model_strain, internal_cell_indices)
        z_real = masked_modified_zscore(real_strain, internal_cell_indices)
        shift = z_model - z_real
        shift_list.append(shift)

        # mesh = meshio.Mesh(
        #     model_pos,
        #     cells,
        #     cell_data={
        #         "strain": [model_strain],
        #         "modified_z_score": [z_model],
        #         "shift": [shift],
        #     }
        # )
        # mesh.write(f"{OUTPUT_DIR}/model_strain_{sample_idx}.vtu")

        # mesh = meshio.Mesh(
        #     real_pos,
        #     cells,
        #     cell_data={
        #         "strain": [real_strain],
        #         "modified_z_score": [z_real],
        #         "shift": [shift],
        #     }
        # )
        # mesh.write(f"{OUTPUT_DIR}/real_strain_{sample_idx}.vtu")

        if sample_idx < 3 or sample_idx == sample_count - 1:
            action_norm = float(np.linalg.norm(action_value.reshape(-1)))
            shift_internal = shift[internal_cell_indices]
            print(
                f"Frame {int(frames[sample_idx]):03d}: contact={CONTACT_IDX}, "
                f"action_norm={action_norm:.4e}, "
                f"shift_mean={shift_internal.mean():.4e}, shift_std={shift_internal.std():.4e}"
            )

    shift_array = np.stack(shift_list, axis=0)
    mean_shift = shift_array.mean(axis=0)
    return shift_array, mean_shift, mesh_data


if __name__ == "__main__":
    ti.init(arch=ti.cuda, debug=False, default_fp=ti.f64)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    shift_per_sample, shift_mean, mesh_data = analyze_real_dataset(DEFAULT_DATA_DIR)

    np.save(RESULT_DIR / "zscore_shift_per_sample.npy", shift_per_sample)
    np.savetxt(RESULT_DIR / "zscore_shift_mean.csv", shift_mean, delimiter=",", fmt="%.4f")
    plot_mean_shift(mesh_data, shift_mean, RESULT_DIR / "zscore_shift_mean.svg")

    print(f"Saved per-sample z-score shift to {RESULT_DIR / 'zscore_shift_per_sample.npy'}")
    print(f"Saved mean z-score shift to {RESULT_DIR / 'zscore_shift_mean.csv'}")
    print(f"Saved visualization to {RESULT_DIR / 'zscore_shift_mean.svg'}")
