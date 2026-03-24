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


def rgbd_to_pointcloud(rgb_image, depth_image, intrinsics, camera_to_world, max_depth=3.0):
    """
    Convert RGBD images to a 3D point cloud in world coordinates using PyTorch.

    Args:
        rgb_image: HxWx3 torch tensor
        depth_image: HxW torch tensor (meters)
        intrinsics: 3x3 torch tensor
        camera_to_world: 4x4 torch tensor
        max_depth: maximum valid depth (meters)

    Returns:
        points_world: Nx3 torch tensor (world coords)
        colors: Nx3 torch tensor (0-255)
    """
    
    H, W = depth_image.shape
    device = depth_image.device

    # Create pixel coordinate grid
    v_coords, u_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    v_coords = v_coords.reshape(-1).float()
    u_coords = u_coords.reshape(-1).float()
    
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


def sample_virtual_fps(scene_center, radius, anchor_position, num_samples, n_candidates=4096):
    """
    Sample virtual camera positions on the upper hemisphere using farthest point sampling (PyTorch).

    Args:
        scene_center: (3,) tensor — center of the scene
        radius: float — radius of the hemisphere
        anchor_position: (3,) tensor — seed camera position for FPS
        num_samples: int — number of new cameras to sample
        n_candidates: int — number of candidate points on the hemisphere

    Returns:
        cameras_c2w: (num_samples, 4, 4) tensor of camera-to-world matrices
    """
    device = anchor_position.device

    # Generate candidate points on upper hemisphere
    u = torch.rand(n_candidates, device=device)
    v = torch.rand(n_candidates, device=device)
    theta = torch.arccos(1 - u)
    phi = 2 * torch.pi * v

    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.cos(theta),
        torch.sin(theta) * torch.sin(phi),
    ], dim=1)  # (n_candidates, 3)

    candidates = scene_center.unsqueeze(0) + radius * directions  # (n_candidates, 3)

    # FPS with anchor as seed
    selected = anchor_position.unsqueeze(0)  # (1, 3)

    new_positions = []
    for _ in range(num_samples):
        dists = torch.cdist(candidates, selected)  # (n_candidates, M)
        min_dists = dists.min(dim=1).values         # (n_candidates,)
        farthest_idx = torch.argmax(min_dists)
        new_pos = candidates[farthest_idx]
        new_positions.append(new_pos)
        selected = torch.cat([selected, new_pos.unsqueeze(0)], dim=0)

    new_positions = torch.stack(new_positions, dim=0)  # (num_samples, 3)

    # Build camera-to-world matrices (look at scene_center)
    up = torch.tensor([0.0, -1.0, 0.0], device=device)

    z_axes = scene_center.unsqueeze(0) - new_positions  # (N, 3)
    z_axes = z_axes / z_axes.norm(dim=1, keepdim=True)

    x_axes = torch.linalg.cross(up.unsqueeze(0).expand_as(z_axes), z_axes)
    x_axes = x_axes / x_axes.norm(dim=1, keepdim=True)

    y_axes = torch.linalg.cross(z_axes, x_axes)

    cameras_c2w = torch.zeros(num_samples, 4, 4, device=device)
    cameras_c2w[:, :3, 0] = x_axes
    cameras_c2w[:, :3, 1] = y_axes
    cameras_c2w[:, :3, 2] = z_axes
    cameras_c2w[:, :3, 3] = new_positions
    cameras_c2w[:, 3, 3] = 1.0

    return cameras_c2w

def pointcloud_to_rgbd_batch(points3D, colors, intrinsics, world_to_cameras, width, height):
    """
    Batch-render a point cloud into multiple RGBD images with z-buffering.

    Args:
        points3D:          (N, 3) tensor — world-space points
        colors:            (N, 3) tensor — per-point RGB (0-255)
        intrinsics:        (B, 3, 3) tensor — per-camera intrinsics
        world_to_cameras:  (B, 4, 4) tensor — per-camera world-to-camera transforms
        width, height:     int — image dimensions (same for all cameras)

    Returns:
        color_imgs: (B, H, W, 3) float tensor
        depth_maps: (B, H, W) float tensor (meters, 0 = no hit)
    """
    
    device = points3D.device
    B = world_to_cameras.shape[0]
    H, W = height, width

    R = world_to_cameras[:, :3, :3]  # (B, 3, 3)
    t = world_to_cameras[:, :3, 3]   # (B, 3)

    # Transform all points into all camera frames: (B, N, 3)
    points_cam = torch.einsum('bij,nj->bni', R, points3D) + t[:, None, :]

    z = points_cam[:, :, 2]  # (B, N)

    # Project to pixel coords (clamp z to avoid div-by-zero; invalid z filtered below)
    z_safe = z.clamp(min=1e-8)
    fx = intrinsics[:, 0, 0][:, None]  # (B, 1)
    fy = intrinsics[:, 1, 1][:, None]
    cx = intrinsics[:, 0, 2][:, None]
    cy = intrinsics[:, 1, 2][:, None]

    u = torch.floor(points_cam[:, :, 0] / z_safe * fx + cx).long()  # (B, N)
    v = torch.floor(points_cam[:, :, 1] / z_safe * fy + cy).long()

    valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)  # (B, N)

    # Gather valid entries (flattened across batch)
    batch_idx, point_idx = torch.where(valid)  # both (M,)
    u_val = u[batch_idx, point_idx]
    v_val = v[batch_idx, point_idx]
    z_val = z[batch_idx, point_idx]
    c_val = colors[point_idx]  # same point cloud for all cameras

    # Flat pixel index with per-image offset: unique across the entire batch
    pixel_idx = batch_idx * (H * W) + v_val * W + u_val  # (M,)

    # ---- CUDA-safe z-buffer (double-sort + first-occurrence) ----
    # 1) sort by depth ascending (nearest first)
    z_order = torch.argsort(z_val)
    # 2) stable sort by pixel preserves depth order within each pixel
    pixel_order = torch.argsort(pixel_idx[z_order], stable=True)
    final_order = z_order[pixel_order]

    pixel_final = pixel_idx[final_order]
    z_final = z_val[final_order]
    c_final = c_val[final_order]

    # First occurrence per pixel = closest point
    first_mask = torch.cat([
        torch.tensor([True], device=device),
        pixel_final[1:] != pixel_final[:-1]
    ])

    # Write only unique winners — fully deterministic on CUDA
    winners = pixel_final[first_mask]
    flat_depth = torch.zeros(B * H * W, device=device)
    flat_color = torch.zeros(B * H * W, 3, device=device)
    flat_depth[winners] = z_final[first_mask]
    flat_color[winners] = c_final[first_mask].float()

    return flat_color.reshape(B, H, W, 3), flat_depth.reshape(B, H, W)