""" 与Sofa相关的处理函数
created on 2025-11-12
"""
from typing import List
import numpy as np
import numpy.typing as npt
import copy

def add_move(handle_list:List, dt:float, movement:npt.NDArray):
    """ Use `LinearMovemetConstraint` to add a simulation step-wise movement
    Args:
        handle_list (List): The node of the object
        dt (float): The time step
        movement (npt.NDArray): The additional movement
    """
    if movement.ndim == 1:
        movement = np.expand_dims(movement, axis=0)
    if movement.ndim >=2 and movement.shape[1] == 2:
        movement = np.concatenate((movement, np.zeros((movement.shape[0], 1))), axis=1)
    if len(handle_list) != movement.shape[0]:
        raise ValueError("The number of handles must match the number of movement vectors.")
    for i, handle in enumerate(handle_list):
        times_array = handle.findData('keyTimes').value
        movements_array = handle.findData('movements').value

        last_time = times_array[-1]
        last_movement = movements_array[-1, :]

        handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
        handle.findData('movements').value = np.append(movements_array, [movement[i,:] + last_movement], axis=0)

def get_node_pos(handle, marker_idx:list)->npt.NDArray:
    """ 从sofa中获取指定节点的位置
    """
    marker_pos = np.zeros((len(marker_idx), 3))
    # node_pos = handle.findData('position').value
    for i, idx in enumerate(marker_idx):
        pos_tmp = copy.deepcopy(handle.findData('position').value[idx])
        marker_pos[i] = pos_tmp
    return marker_pos