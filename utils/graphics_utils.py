#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def rgbd_to_pointcloud(rgb_image, depth_image, intrinsics, camera_to_world, max_depth=3.0, device='cuda'):
    """
    Convert RGBD images to a 3D point cloud in world coordinates using PyTorch.

    Args:
        rgb_image: HxWx3 numpy array (0-255) or torch tensor
        depth_image: HxW numpy array (meters) or torch tensor
        intrinsics: 3x3 numpy array or torch tensor
        camera_to_world: 4x4 numpy array or torch tensor
        max_depth: maximum valid depth (meters)
        device: torch device

    Returns:
        points_world: Nx3 torch tensor (world coords)
        colors: Nx3 torch tensor (0-255)
    """
    # Convert to tensors
    if not isinstance(rgb_image, torch.Tensor):
        rgb_image = torch.tensor(rgb_image, dtype=torch.float32, device=device)
    if not isinstance(depth_image, torch.Tensor):
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=device)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    if not isinstance(camera_to_world, torch.Tensor):
        camera_to_world = torch.tensor(camera_to_world, dtype=torch.float32, device=device)

    H, W = depth_image.shape

    # Create pixel coordinate grid
    v_coords, u_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u_coords = u_coords.reshape(-1).float()
    v_coords = v_coords.reshape(-1).float()

    # Get depth values
    z = depth_image.reshape(-1)

    # Valid depth mask
    valid = (z > 0) & (z <= max_depth)
    u_coords, v_coords, z = u_coords[valid], v_coords[valid], z[valid]

    # Unproject to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy
    points_cam = torch.stack([x, y, z], dim=1)  # Nx3

    # Transform to world coordinates
    R = camera_to_world[:3, :3]
    t = camera_to_world[:3, 3]
    points_world = (R @ points_cam.T).T + t  # Nx3

    # Get colors
    u_idx = u_coords.long()
    v_idx = v_coords.long()
    colors = rgb_image[v_idx, u_idx]  # Nx3

    return points_world, colors