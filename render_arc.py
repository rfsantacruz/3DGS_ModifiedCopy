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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from scene.dataset_readers import storePly

def look_at(cam_pos, target, up=np.array([0,0,1.])):
    """Compute world-to-camera rotation using a LookAt convention."""
    f = target - cam_pos
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1)  # world-to-camera

def rotate_point_around_axis(point, center, axis, angle_rad):
    """Rotate a point around an axis passing through 'center'."""
    axis = axis / np.linalg.norm(axis)
    rot = Rot.from_rotvec(axis * angle_rad)
    return center + rot.apply(point - center)

def generate_arc_cameras(init_cam, angles_deg, d=2.0, up=np.array([0,-1.0,0.0])):
    """
    Given a base camera (R_wc, t_wc), generate a camera for each angle in 'angles_deg'
    rotated around the viewing target point at distance d.
    
    angles_deg: list of angles in degrees (e.g. [-90, -45, -30, 0, 30, 45, 60])
    Returns: list of (R, t) camera extrinsics in COLMAP format.
    """
    
    R_wc = init_cam.R.T  # world-to-camera rotation
    t_wc = init_cam.T  # camera translation in world

    # 1. Camera center
    C = -R_wc.T @ t_wc  # world coordinates
   
    # 2. Compute forward direction (optical axis)
    forward = R_wc.T @ np.array([0, 0, 1.0])  # z-axis in world

    # 3. Target point the camera is looking at
    T = C + d * forward

    cameras = []

    # 4. For each angle generate new camera
    for angle in angles_deg:
        angle_rad = np.deg2rad(angle)

        # New camera center
        C_new = rotate_point_around_axis(C, T, up, angle_rad)

        # New rotation looking at the same target
        R_new = look_at(C_new, T, up)

        # Convert to COLMAP translation
        t_new = -R_new @ C_new
        
        new_cam = copy.deepcopy(init_cam)
        new_cam.R = R_new.T
        new_cam.T = t_new
        new_cam.world_view_transform = torch.tensor(getWorld2View2(new_cam.R, new_cam.T, new_cam.trans, new_cam.scale)).transpose(0, 1).cuda()
        new_cam.projection_matrix = getProjectionMatrix(znear=new_cam.znear, zfar=new_cam.zfar, fovX=new_cam.FoVx, fovY=new_cam.FoVy).transpose(0,1).cuda()
        new_cam.full_proj_transform = (new_cam.world_view_transform.unsqueeze(0).bmm(new_cam.projection_matrix.unsqueeze(0))).squeeze(0)
        new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]
        cameras.append(new_cam)

    return cameras


def generate_circle_camera(init_cam, n_frames=120, depth=1.0):
    """ Virtual camera path that is a circular motion around a point in front of the given camera pose. """
    
    pose = init_cam.world_view_transform.inverse().cpu().numpy().T[:-1, :]
    # Find look-at point in front of camera
    mid_point = np.array([0., 0., depth, 1.])
    # Project look-at point to world coordinates
    cam2world = np.concatenate([pose[:, :4], [[0., 0., 0., 1.]]], axis=0)
    mid_point = cam2world @ mid_point
    mid_point = mid_point[:3]
    # Calculate distance from camera to look-at point in XY-plane
    x, z, y = pose[:, 3]
    radius = np.linalg.norm([x - mid_point[0], y - mid_point[2]])
    # Offset to get pose as starting camera position
    dist = y - mid_point[2]
    offset = np.arcsin(dist / radius)
    # Get points on circle moving around look-at point with same distance
    points_on_circle = np.array([[
        np.cos(2 * np.pi / n_frames * x + offset) * radius + mid_point[0],
        np.sin(2 * np.pi / n_frames * x + offset) * radius + mid_point[2]] for x in range(n_frames)])
    # Get camera positions of these points with equal height (Z-axis)
    translations = np.stack([points_on_circle[:, 0], np.array([z]).repeat(n_frames), points_on_circle[:, 1]], axis=1)
    # Calculate look-at matrices and create render poses
    cameras = []
    for t in translations:
        r = look_at(t, mid_point, up=np.array([0., -1.0, 0.0]))
        new_cam = copy.deepcopy(init_cam)
        new_cam.R = r.T
        new_cam.T = t
        new_cam.world_view_transform = torch.tensor(getWorld2View2(new_cam.R, new_cam.T, new_cam.trans, new_cam.scale)).transpose(0, 1).cuda()
        new_cam.projection_matrix = getProjectionMatrix(znear=new_cam.znear, zfar=new_cam.zfar, fovX=new_cam.FoVx, fovY=new_cam.FoVy).transpose(0,1).cuda()
        new_cam.full_proj_transform = (new_cam.world_view_transform.unsqueeze(0).bmm(new_cam.projection_matrix.unsqueeze(0))).squeeze(0)
        new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]
        cameras.append(new_cam)
        
    return cameras


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
   

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_output = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_output["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        if 'depth' in render_output and ((view.sensorinvdepthmap is not None) or (view.mlinvdepthmap is not None)):
            render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
            gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depths")
            makedirs(render_depth_path, exist_ok=True); makedirs(gt_depth_path, exist_ok=True);
            
            depth = 1.0 / (render_output['depth'])
            depth = (depth-depth.min()) / (depth.max()-depth.min()) 
            torchvision.utils.save_image(depth, os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))

            if view.sensorinvdepthmap is not None:
                depth_gt = view.sensorinvdepthmap; depth_gt[depth_gt > 0] = 1.0 / depth_gt[depth_gt > 0]
                depth_gt = (depth_gt-depth_gt.min()) / (depth_gt.max()-depth_gt.min())
                torchvision.utils.save_image(depth_gt, os.path.join(gt_depth_path, '{0:05d}_sensor.png'.format(idx)))

            if view.mlinvdepthmap is not None:
                depth_gt = view.mlinvdepthmap; depth_gt[depth_gt > 0] = 1.0 / depth_gt[depth_gt > 0]
                depth_gt = (depth_gt-depth_gt.min()) / (depth_gt.max()-depth_gt.min())
                torchvision.utils.save_image(depth_gt, os.path.join(gt_depth_path, '{0:05d}_ml.png'.format(idx)))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, mode : str, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, isotropic_scaling=pipeline.isotropic_scaling)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        if mode == 'arc':
            # cameras center
            d = 3.0; angles_deg = [-45, -30, -15, 0, 15, 30, 45]; 
            # up_vector = np.array([0,-1.0,0.0])
            up_vector = [sc.R.T[1, :] for sc in scene.getTrainCameras()]
            up_vector = np.mean(up_vector, axis=0)
            up_vector /= np.linalg.norm(up_vector)

            for i, cam in enumerate(scene.getTrainCameras()):
                new_cameras = generate_arc_cameras(cam, angles_deg, d, up_vector)
                render_set(dataset.model_path, "arc/cam_{:02d}".format(i), scene.loaded_iter, new_cameras, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
                centers = np.array([c.camera_center.cpu().numpy() for c in new_cameras]);
                colors = np.tile(np.array([[255.0, 0.0, 0.0]]), (len(new_cameras), 1))
                storePly(os.path.join(dataset.model_path, "arc/cam_{:02d}/cameras.ply".format(i)), centers, colors)

        elif mode == 'circular':
            n_frames=5; depth=1.0;
            for i, cam in enumerate(scene.getTrainCameras()):
                new_cameras = generate_circle_camera(cam, n_frames, depth)
                render_set(dataset.model_path, "circular/cam_{:02d}".format(i), scene.loaded_iter, new_cameras, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
                centers = np.array([c.camera_center.cpu().numpy() for c in new_cameras]);
                colors = np.tile(np.array([[255.0, 0.0, 0.0]]), (len(new_cameras), 1))
                storePly(os.path.join(dataset.model_path, "circular/cam_{:02d}/cameras.ply".format(i)), centers, colors)




        else:
            raise ValueError("Unknown rendering mode: {}".format(mode))

        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mode", default='arc', choices=['arc', 'circular'])
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.mode, SPARSE_ADAM_AVAILABLE)