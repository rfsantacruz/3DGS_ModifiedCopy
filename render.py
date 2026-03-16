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

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, isotropic_scaling=pipeline.isotropic_scaling)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)