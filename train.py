#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import sys
import json
import time
import os
import shutil
from collections import defaultdict
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import psnr, ssim, inverse_depth_smoothness_loss_mask, tv_loss
from gaussian_renderer import render, render_range_map
from scene import Scene, GaussianModel, RayDropPrior
from head.semantic_head import build_semantic_head
from utils.general_utils import seed_everything, visualize_depth
from utils.graphics_utils import pano_to_lidar
from utils.system_utils import save_ply
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import numpy as np
from omegaconf import OmegaConf
from utils.graphics_utils import depth_to_normal
from utils.metrics_utils import DepthMeter, PointsMeter, RaydropMeter, IntensityMeter, SemanticMeter
from utils.semantic_mapping import load_semantic_label_mapping, colorize_semantic_tensor
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from scene.unet import UNet
from datetime import datetime
from utils.nerfstudio_utils import create_viewer_from_args

EPS = 1e-5


def training(args):
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians, shuffle=args.shuffle)
    with open(os.path.join(args.model_path, 'scale_factor.txt'), 'w') as f:
        f.writelines(str(args.scale_factor))

    gaussians.training_setup(args)

    viewer = create_viewer_from_args(args)
    if viewer is not None:
        try:
            viewer.update_gaussians(0, scene.gaussians)
        except Exception as exc:
            print(f"[Viewer] Initialization update failed: {exc}")
            viewer.close()
            viewer = None

    semantic_head = None
    semantic_optimizer = None
    semantic_class_weights = None
    semantic_ignore_index = getattr(args, "semantic_ignore_index", -1)

    if getattr(args, "lambda_semantic", 0.0) > 0:
        if not hasattr(args, "semantic_num_classes") or args.semantic_num_classes is None:
            raise ValueError("`semantic_num_classes` must be specified when `lambda_semantic` is positive.")
        if getattr(args, "semantic_dim", 0) <= 0:
            raise ValueError("`semantic_dim` must be a positive integer when `lambda_semantic` is positive.")

        head_type = getattr(args, "semantic_head_type", "conv")
        norm_type = getattr(args, "semantic_norm_type", "bn")
        if norm_type is None:
            norm_type = "bn"
        dropout_prob = getattr(args, "semantic_dropout", 0.0)
        if dropout_prob is None:
            dropout_prob = 0.0

        semantic_kwargs = {
            "norm_type": norm_type,
            "dropout": dropout_prob,
        }

        if head_type.lower() in ("unet", "unet-style", "u-net"):
            base_channels = getattr(args, "semantic_unet_base_channels", 64)
            if base_channels is None:
                base_channels = 64
            depth_value = getattr(args, "semantic_unet_depth", 3)
            if depth_value is None:
                depth_value = 3
            semantic_kwargs.update({
                "base_channels": base_channels,
                "depth": depth_value,
            })
        else:
            hidden_cfg = getattr(args, "semantic_conv_hidden_dims", (128, 64))
            if hidden_cfg is None:
                hidden_dims = [128, 64]
            elif isinstance(hidden_cfg, (int, float)):
                hidden_dims = [int(hidden_cfg)]
            else:
                hidden_dims = [int(dim) for dim in list(hidden_cfg)]
            semantic_kwargs.update({
                "hidden_channels": hidden_dims,
            })

        semantic_head = build_semantic_head(
            head_type=head_type,
            in_channels=args.semantic_dim,
            num_classes=args.semantic_num_classes,
            **semantic_kwargs,
        ).cuda()
        semantic_head.train()

        semantic_optimizer = torch.optim.Adam(
            semantic_head.parameters(),
            lr=getattr(args, "semantic_head_lr", 1e-3),
            weight_decay=getattr(args, "semantic_head_weight_decay", 0.0),
        )

        class_weights_cfg = getattr(args, "semantic_class_weights", None)
        if class_weights_cfg is not None:
            semantic_class_weights = torch.tensor(
                list(class_weights_cfg),
                dtype=torch.float32,
                device="cuda",
            )
            if semantic_class_weights.numel() != args.semantic_num_classes:
                raise ValueError("`semantic_class_weights` length must match `semantic_num_classes`.")

    start_w, start_h = scene.getWH()
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w).cuda()
    lidar_raydrop_prior.training_setup(args)

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

        lidar_raydrop_prior_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                                      os.path.basename(args.start_checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
        (lidar_raydrop_prior_params, _) = torch.load(lidar_raydrop_prior_checkpoint)
        lidar_raydrop_prior.restore(lidar_raydrop_prior_params)

        if semantic_head is not None:
            semantic_head_checkpoint = os.path.join(
                os.path.dirname(args.start_checkpoint),
                os.path.basename(args.start_checkpoint).replace("chkpnt", "semantic_head_chkpnt"),
            )
            if os.path.exists(semantic_head_checkpoint):
                semantic_state = torch.load(semantic_head_checkpoint, map_location="cuda")
                semantic_head.load_state_dict(semantic_state["state_dict"])
                if "iteration" in semantic_state:
                    first_iter = semantic_state["iteration"]

        for i in range(first_iter // args.scale_increase_interval):
            scene.upScale()

    bg_color = [1, 1, 1, 1] if args.white_background else [0, 0, 0, 1] # 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.test_only or first_iter == args.iterations:
        with torch.no_grad():
            complete_eval(first_iter, args.test_iterations, scene, render, (args, background),
                          {}, env_map=lidar_raydrop_prior, semantic_head=semantic_head)
        if viewer is not None:
            viewer.close()
            return

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress", miniters=10)
    loss_log_interval = getattr(args, "loss_log_interval", 50)

    for iteration in progress_bar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if semantic_optimizer is not None:
            semantic_optimizer.zero_grad(set_to_none=True)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]

        # render v, t scale map and semantic feature
        v = gaussians.get_inst_velocity
        t_scale = gaussians.get_scaling_t.clamp_max(2)
        semantic_feature = gaussians.get_semantic_feature  # [N, semantic_dim]
        other = [t_scale, v, semantic_feature]

        if np.random.random() < args.lambda_self_supervision:
            time_shift = 3 * (np.random.random() - 0.5) * scene.time_interval
        else:
            time_shift = None
        render_pkg = render(viewpoint_cam, gaussians, args, background, env_map=lidar_raydrop_prior, other=other, time_shift=time_shift, is_training=True)

        depth = render_pkg["depth"]
        depth_median = render_pkg["depth_median"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        log_dict = {}

        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:4]
        rendered_semantic = feature[4:]

        intensity_sh_map = render_pkg['intensity_sh']
        raydrop_map = render_pkg['raydrop']

        if args.sky_depth:
            sky_depth = 900
            depth = depth / alpha.clamp_min(EPS)
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        loss = 0
        semantic_supervised = False

        # Semantic loss
        if args.lambda_semantic > 0 and semantic_head is not None and rendered_semantic.numel() > 0:
            gt_semantic = getattr(viewpoint_cam, "pts_semantic", None)
            if gt_semantic is not None:
                gt_semantic = gt_semantic.long()
                if gt_semantic.dim() == 3 and gt_semantic.shape[0] == 1:
                    gt_semantic = gt_semantic.squeeze(0)
                if gt_semantic.dim() == 2:
                    gt_semantic = gt_semantic.unsqueeze(0)

                effective_ignore_index = semantic_ignore_index if semantic_ignore_index is not None else -100

                num_classes = getattr(args, "semantic_num_classes", None)
                needs_clone = (gt_semantic < 0).any() or (num_classes is not None and (gt_semantic >= num_classes).any())
                if needs_clone:
                    gt_semantic = gt_semantic.clone()
                    if (gt_semantic < 0).any():
                        gt_semantic[gt_semantic < 0] = effective_ignore_index
                    if num_classes is not None:
                        high_mask = gt_semantic >= num_classes
                        if high_mask.any():
                            gt_semantic[high_mask] = effective_ignore_index

                valid_mask = gt_semantic != effective_ignore_index

                if valid_mask.any():
                    semantic_input = rendered_semantic.unsqueeze(0)
                    semantic_logits = semantic_head(semantic_input)
                    ce_kwargs = {}
                    if semantic_class_weights is not None:
                        ce_kwargs["weight"] = semantic_class_weights
                    ce_kwargs["ignore_index"] = effective_ignore_index
                    semantic_loss = F.cross_entropy(semantic_logits, gt_semantic, **ce_kwargs)
                    with torch.no_grad():
                        eval_mask = valid_mask
                        preds = semantic_logits.detach().argmax(dim=1, keepdim=False)
                        semantic_accuracy = (preds[eval_mask] == gt_semantic[eval_mask]).float().mean()
                        log_dict['semantic_acc'] = semantic_accuracy.item()
                    log_dict['loss_semantic'] = semantic_loss.item()
                    loss += args.lambda_semantic * semantic_loss
                    semantic_supervised = True

        if args.lambda_distortion > 0:
            lambda_dist = args.lambda_distortion if iteration > 3000 else 0.0
            distortion = render_pkg["distortion"]
            loss_distortion = distortion.mean()
            log_dict['loss_distortion'] = loss_distortion.item()
            loss += lambda_dist * loss_distortion

        if args.lambda_lidar > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = pts_depth > 0
            if mask.sum() == 0:
                loss_lidar = torch.tensor(0.0, device=depth.device)
            else:
                loss_lidar = F.l1_loss(pts_depth[mask], depth[mask])
            if args.lidar_decay > 0:
                iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
            else:
                iter_decay = 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_lidar_median > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = pts_depth > 0
            loss_lidar_median = F.l1_loss(pts_depth[mask], depth_median[mask])
            log_dict['loss_lidar_median'] = loss_lidar_median.item()
            loss += args.lambda_lidar_median * loss_lidar_median

        if args.lambda_t_reg > 0:
            loss_t_reg = -torch.abs(t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

        if args.lambda_v_reg > 0:
            loss_v_reg = torch.abs(v_map).mean()
            log_dict['loss_v_reg'] = loss_v_reg.item()
            loss += args.lambda_v_reg * loss_v_reg

        # Intensity sh
        if args.lambda_intensity_sh > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = pts_depth > 0
            pts_intensity = viewpoint_cam.pts_intensity.cuda()
            loss_intensity_sh = torch.nn.functional.l1_loss(pts_intensity[mask], intensity_sh_map[mask])
            log_dict['loss_intensity_sh'] = loss_intensity_sh.item()
            loss += args.lambda_intensity_sh * loss_intensity_sh

        if args.lambda_raydrop > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            gt_raydrop = 1.0 - (pts_depth > 0).float()
            loss_raydrop = torch.nn.functional.binary_cross_entropy(raydrop_map, gt_raydrop)
            log_dict['loss_raydrop'] = loss_raydrop.item()
            loss += args.lambda_raydrop * loss_raydrop

        # chamfer loss
        if args.lambda_chamfer > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = (pts_depth > 0).float()
            cham_fn = chamfer_3DDist()
            pred_lidar = pano_to_lidar(depth * mask, args.vfov, args.hfov) / args.scale_factor
            gt_lidar = pano_to_lidar(pts_depth, args.vfov, args.hfov) / args.scale_factor
            dist1, dist2, _, _ = cham_fn(pred_lidar[None], gt_lidar[None])

            loss_chamfer = dist1.mean() + dist2.mean()
            log_dict['loss_chamfer'] = loss_chamfer.item()
            loss += args.lambda_chamfer * loss_chamfer

        if args.lambda_smooth > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            gt_grad_x = pts_depth[:, :, :-1] - pts_depth[:, :, 1:]
            gt_grad_y = pts_depth[:, :-1, :] - pts_depth[:, 1:, :]
            mask_x = (torch.where(pts_depth[:, :, :-1] > 0, 1, 0) *
                      torch.where(pts_depth[:, :, 1:] > 0, 1, 0))
            mask_y = (torch.where(pts_depth[:, :-1, :] > 0, 1, 0) *
                      torch.where(pts_depth[:, 1:, :] > 0, 1, 0))

            grad_clip = 0.01 * args.scale_factor
            grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip, 1, 0) * mask_x
            grad_mask_x = grad_mask_x.bool()
            grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip, 1, 0) * mask_y
            grad_mask_y = grad_mask_y.bool()

            pred_grad_x = depth[:, :, :-1] - depth[:, :, 1:]
            pred_grad_y = depth[:, :-1, :] - depth[:, 1:, :]
            loss_smooth = (F.l1_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x])
                           + F.l1_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y]))
            log_dict['loss_smooth'] = loss_smooth.item()
            loss += args.lambda_smooth * loss_smooth

        if args.lambda_tv > 0:
            loss_tv = tv_loss(depth)
            log_dict['loss_tv'] = loss_tv.item()
            loss += args.lambda_tv * loss_tv

        # 每个gaussian的opa 而不是render的 没用
        if args.lambda_gs_opa > 0:
            o = gaussians.get_opacity.clamp(1e-6, 1 - 1e-6)
            loss_gs_opa = ((1 - o) ** 2).mean()
            log_dict['loss_depth_opa'] = loss_gs_opa.item()
            loss = loss + args.lambda_gs_opa * loss_gs_opa

        # Normal Consistency in 2dgs
        if args.lambda_normal_consistency > 0:
            lambda_normal = args.lambda_normal_consistency if iteration > 7000 else 0.0
            surf_normal = depth_to_normal(depth, args.vfov, args.hfov)
            render_normal = render_pkg["normal"]
            loss_normal_consistency = (1 - (render_normal * surf_normal).sum(dim=0)[1:-1, 1:-1]).mean()
            log_dict['loss_normal_consistency'] = loss_normal_consistency.item()
            loss = loss + lambda_normal * loss_normal_consistency

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o * torch.log(o)).mean()
            log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

        if args.lambda_depth_var > 0:
            depth_var = render_pkg["depth_square"] - depth ** 2
            loss_depth_var = depth_var.clamp_min(1e-6).sqrt().mean()
            log_dict["loss_depth_var"] = loss_depth_var.item()
            lambda_depth_var = args.lambda_depth_var if iteration > 3000 else 0.0
            loss = loss + lambda_depth_var * loss_depth_var

        if args.lambda_semantic_chamfer > 0 and semantic_supervised and rendered_semantic.numel() > 0:
            """
            Per-class Chamfer distance loss with proper 3D point conversion.
            """
            pts_depth = viewpoint_cam.pts_depth.cuda()
            valid_mask = pts_depth > 0
            
            # Reuse semantic predictions
            if 'semantic_logits' not in locals() or semantic_logits is None:
                with torch.no_grad():
                    semantic_input = rendered_semantic.unsqueeze(0)
                    semantic_logits = semantic_head(semantic_input)
            
            semantic_pred = semantic_logits.detach().argmax(dim=1).squeeze(0)  # [H, W]
            
            # Get GT semantic
            gt_semantic = getattr(viewpoint_cam, "pts_semantic", None)
            if gt_semantic is None:
                log_dict['loss_semantic_chamfer'] = 0.0
            else:
                if gt_semantic.dim() == 3 and gt_semantic.shape[0] == 1:
                    gt_semantic = gt_semantic.squeeze(0)
                
                # Class weights
                class_weights_cfg = getattr(args, "semantic_chamfer_class_weights", {})
                default_weight = 1.0
                
                # Per-class Chamfer
                cham_fn = chamfer_3DDist()
                
                # Find common classes
                pred_classes = torch.unique(semantic_pred[valid_mask.squeeze(0)])
                gt_classes = torch.unique(gt_semantic[valid_mask.squeeze(0)])
                common_classes = [c.item() for c in pred_classes 
                                if c in gt_classes 
                                and c != getattr(args, "semantic_ignore_index", -1)]
                loss_semantic_chamfer = 0.0
                total_weight = 0.0
                class_losses = {}
                skipped_classes = {}
                
                for cls_id in common_classes:
                    # Masks
                    pred_mask = (semantic_pred == cls_id) & valid_mask.squeeze(0)
                    gt_mask = (gt_semantic == cls_id) & valid_mask.squeeze(0)
                    
                    pred_count = pred_mask.sum().item()
                    gt_count = gt_mask.sum().item()
                
                    if not (pred_count > 0  and gt_count > 0):
                        continue
                        
                    # Create class-specific depth maps
                    pred_depth_cls_map = torch.zeros_like(depth)
                    pred_depth_cls_map.squeeze(0)[pred_mask] = depth.squeeze(0)[pred_mask]
                    
                    gt_depth_cls_map = torch.zeros_like(pts_depth)
                    gt_depth_cls_map.squeeze(0)[gt_mask] = pts_depth.squeeze(0)[gt_mask]
                
                    # Convert to 3D points
                    pred_points_3d = pano_to_lidar(
                        pred_depth_cls_map, 
                        args.vfov, 
                        args.hfov
                    ) / args.scale_factor  # [3, N_total]
                    
                    gt_points_3d = pano_to_lidar(
                        gt_depth_cls_map,
                        args.vfov,
                        args.hfov
                    ) / args.scale_factor  # [3, M_total]
                    
                    # Filter zero points (ensure last dimension is xyz)
                    pred_nonzero = pred_points_3d.abs().sum(dim=1) > 1e-6
                    gt_nonzero = gt_points_3d.abs().sum(dim=1) > 1e-6
                    
                    pred_points_3d = pred_points_3d[pred_nonzero]  # [N, 3]
                    gt_points_3d = gt_points_3d[gt_nonzero]        # [M, 3]
                    
                    # Sample for efficiency
                    max_points = 1000
                    if pred_points_3d.shape[0] > max_points:
                        indices = torch.randperm(pred_points_3d.shape[0], device=pred_points_3d.device)[:max_points]
                        pred_points_3d = pred_points_3d[indices]
                    if gt_points_3d.shape[0] > max_points:
                        indices = torch.randperm(gt_points_3d.shape[0], device=gt_points_3d.device)[:max_points]
                        gt_points_3d = gt_points_3d[indices]
                    
                    # Compute Chamfer
                    pred_3d = pred_points_3d.unsqueeze(0)  # [1, N, 3]
                    gt_3d = gt_points_3d.unsqueeze(0)      # [1, M, 3]
                    
                    dist1, dist2, _, _ = cham_fn(pred_3d, gt_3d)
                    cls_chamfer = dist1.mean() + dist2.mean()
                    
                    # Weighted accumulation
                    cls_weight = class_weights_cfg.get(cls_id, default_weight)
                    loss_semantic_chamfer += cls_weight * cls_chamfer
                    total_weight += cls_weight
                    
                    class_losses[cls_id] = cls_chamfer.item()
                
                if total_weight > 0:
                    loss_semantic_chamfer = loss_semantic_chamfer / total_weight
                    log_dict['loss_semantic_chamfer'] = loss_semantic_chamfer.item()
                    loss += args.lambda_semantic_chamfer * loss_semantic_chamfer
                else:
                    if iteration % 100 == 0:
                        print(f"\n  [FAIL] No classes processed! Reasons:")
                    log_dict['loss_semantic_chamfer'] = 0.0

        if args.lambda_semantic_smooth > 0 and semantic_supervised and rendered_semantic.numel() > 0:
            """
            Match GT gradient patterns within the same semantic class.
            
            Key fixes:
            1. Use median-based normalization (robust to outliers)
            2. Compare with GT gradients (not just minimize)
            3. Separate weights for depth and intensity
            4. Use Huber loss for robustness
            """
            pts_depth = viewpoint_cam.pts_depth.cuda()
            valid_mask = pts_depth > 0
            
            # Reuse semantic predictions
            if 'semantic_logits' not in locals() or semantic_logits is None:
                with torch.no_grad():
                    semantic_input = rendered_semantic.unsqueeze(0)
                    semantic_logits = semantic_head(semantic_input)
            
            semantic_pred = semantic_logits.detach().argmax(dim=1).squeeze(0)  # [H, W]
            
            # ===== Compute Gradients =====
            # GT gradients
            gt_depth_grad_x = torch.abs(pts_depth[:, :, :-1] - pts_depth[:, :, 1:])  # [1, H, W-1]
            gt_depth_grad_y = torch.abs(pts_depth[:, :-1, :] - pts_depth[:, 1:, :])  # [1, H-1, W]
            
            # Predicted gradients
            pred_depth_grad_x = torch.abs(depth[:, :, :-1] - depth[:, :, 1:])
            pred_depth_grad_y = torch.abs(depth[:, :-1, :] - depth[:, 1:, :])
            
            # Intensity gradients
            gt_intensity_grad_x = torch.abs(viewpoint_cam.pts_intensity[:, :, :-1] - 
                                            viewpoint_cam.pts_intensity[:, :, 1:])
            gt_intensity_grad_y = torch.abs(viewpoint_cam.pts_intensity[:, :-1, :] - 
                                            viewpoint_cam.pts_intensity[:, 1:, :])
            
            pred_intensity_grad_x = torch.abs(intensity_sh_map[:, :, :-1] - 
                                            intensity_sh_map[:, :, 1:])
            pred_intensity_grad_y = torch.abs(intensity_sh_map[:, :-1, :] - 
                                            intensity_sh_map[:, 1:, :])
            
            # ===== Semantic and Valid Masks =====
            semantic_same_x = (semantic_pred[:, :-1] == semantic_pred[:, 1:]).float()
            semantic_same_y = (semantic_pred[:-1, :] == semantic_pred[1:, :]).float()
            
            valid_mask_x = (pts_depth[:, :, :-1] > 0) & (pts_depth[:, :, 1:] > 0)
            valid_mask_y = (pts_depth[:, :-1, :] > 0) & (pts_depth[:, 1:, :] > 0)
            
            # ===== FIX 1: Robust Normalization (median-based) =====
            with torch.no_grad():
                # Only consider same-class neighbors
                same_class_mask_x = (semantic_same_x > 0) & valid_mask_x.squeeze(0)
                same_class_mask_y = (semantic_same_y > 0) & valid_mask_y.squeeze(0)
                
                # Depth normalization
                valid_gt_depth_x = gt_depth_grad_x[valid_mask_x]
                valid_gt_depth_y = gt_depth_grad_y[valid_mask_y]
                
                if valid_gt_depth_x.numel() > 100:
                    # Use 75th percentile (more robust to very small values)
                    depth_scale_x = torch.quantile(valid_gt_depth_x, 0.75)
                    depth_scale_y = torch.quantile(valid_gt_depth_y, 0.75)
                    
                    # Average both directions
                    depth_scale = (depth_scale_x + depth_scale_y) / 2
                    
                    # Clamp to reasonable range
                    min_scale = 0.02 * args.scale_factor  # 2cm minimum
                    max_scale = 0.5 * args.scale_factor   # 50cm maximum
                    depth_scale = torch.clamp(depth_scale, min_scale, max_scale)
                else:
                    depth_scale = 0.1 * args.scale_factor  # Default: 10cm
                
                # ===== Intensity Scale (SAME METHOD!) =====
                if hasattr(viewpoint_cam, 'pts_intensity') and viewpoint_cam.pts_intensity is not None:
                    # Compute GT intensity gradients
                    gt_intensity_grad_x = torch.abs(viewpoint_cam.pts_intensity[:, :, :-1] - 
                                                viewpoint_cam.pts_intensity[:, :, 1:])
                    gt_intensity_grad_y = torch.abs(viewpoint_cam.pts_intensity[:, :-1, :] - 
                                                viewpoint_cam.pts_intensity[:, 1:, :])
                    
                    # Extract same-class gradients
                    valid_gt_intensity_x = gt_intensity_grad_x.squeeze(0)[same_class_mask_x]
                    valid_gt_intensity_y = gt_intensity_grad_y.squeeze(0)[same_class_mask_y]
                    
                    if valid_gt_intensity_x.numel() > 100:
                        # Use 75th percentile (same as depth)
                        scale_x = torch.quantile(valid_gt_intensity_x, 0.75)
                        scale_y = torch.quantile(valid_gt_intensity_y, 0.75)
                        intensity_scale = (scale_x + scale_y) / 2
                        
                        # Clamp to reasonable range for intensity [0, 1]
                        min_scale = 0.01  # 1% change
                        max_scale = 0.3   # 30% change
                        intensity_scale = torch.clamp(intensity_scale, min_scale, max_scale)
                    else:
                        intensity_scale = 0.05  # Default
                else:
                    intensity_scale = 0.05  # Fallback
            
            # Normalize gradients
            pred_depth_grad_x_norm = pred_depth_grad_x / depth_scale
            pred_depth_grad_y_norm = pred_depth_grad_y / depth_scale
            gt_depth_grad_x_norm = gt_depth_grad_x / depth_scale
            gt_depth_grad_y_norm = gt_depth_grad_y / depth_scale
            
            pred_intensity_grad_x_norm = pred_intensity_grad_x / intensity_scale
            pred_intensity_grad_y_norm = pred_intensity_grad_y / intensity_scale
            gt_intensity_grad_x_norm = gt_intensity_grad_x / intensity_scale
            gt_intensity_grad_y_norm = gt_intensity_grad_y / intensity_scale
            
            # ===== FIX 2: Match GT gradients (not minimize) =====
            # Combined masks
            weight_x = (semantic_same_x * valid_mask_x.squeeze(0)).float()
            weight_y = (semantic_same_y * valid_mask_y.squeeze(0)).float()
            
            num_valid_x = weight_x.sum()
            num_valid_y = weight_y.sum()
            
            if num_valid_x >= 100 and num_valid_y >= 100:
                # ===== Depth Smoothness (match GT gradient pattern) =====
                # Use Huber loss to match gradients
                loss_depth_smooth_x = F.huber_loss(
                    pred_depth_grad_x_norm.squeeze(0)[weight_x > 0],
                    gt_depth_grad_x_norm.squeeze(0)[weight_x > 0],
                    reduction='mean',
                    delta=1.0  # Normalized space
                )
                loss_depth_smooth_y = F.huber_loss(
                    pred_depth_grad_y_norm.squeeze(0)[weight_y > 0],
                    gt_depth_grad_y_norm.squeeze(0)[weight_y > 0],
                    reduction='mean',
                    delta=1.0
                )
                
                loss_semantic_smooth_depth = loss_depth_smooth_x + loss_depth_smooth_y
                
                # ===== Intensity Smoothness =====
                # Check if intensity GT is available
                if hasattr(viewpoint_cam, 'pts_intensity') and viewpoint_cam.pts_intensity is not None:
                    loss_intensity_smooth_x = F.huber_loss(
                        pred_intensity_grad_x_norm.squeeze(0)[weight_x > 0],
                        gt_intensity_grad_x_norm.squeeze(0)[weight_x > 0],
                        reduction='mean',
                        delta=1.0
                    )
                    loss_intensity_smooth_y = F.huber_loss(
                        pred_intensity_grad_y_norm.squeeze(0)[weight_y > 0],
                        gt_intensity_grad_y_norm.squeeze(0)[weight_y > 0],
                        reduction='mean',
                        delta=1.0
                    )
                    
                    loss_semantic_smooth_intensity = loss_intensity_smooth_x + loss_intensity_smooth_y
                else:
                    loss_semantic_smooth_intensity = torch.tensor(0.0, device=depth.device)
                
            loss_semantic_smooth = loss_semantic_smooth_depth + loss_semantic_smooth_intensity
            log_dict['loss_semantic_smooth'] = loss_semantic_smooth.item()
            loss += args.lambda_semantic_smooth * loss_semantic_smooth
        
        if args.lambda_semantic_depth_refine > 0 and semantic_supervised and rendered_semantic.numel() > 0:
            """
            Refine depth geometry using semantic guidance.
            
            Key idea: Within the same semantic class, the relative depth differences
            should match the ground truth. This prevents over-smoothing while
            encouraging correct geometric structure.
            
            Reference: Inspired by PlaneRCNN (Liu et al., CVPR 2019) - plane fitting loss
            
            Args in config:
                lambda_semantic_depth_refine: Loss weight (recommended: 0.01-0.05)
                semantic_depth_refine_threshold: Max valid depth difference (default: 0.5m)
            """
            pts_depth = viewpoint_cam.pts_depth.cuda()
            valid_mask = pts_depth > 0
            
            # Reuse semantic predictions
            if 'semantic_logits' not in locals() or semantic_logits is None:
                with torch.no_grad():
                    semantic_input = rendered_semantic.unsqueeze(0)
                    semantic_logits = semantic_head(semantic_input)
            
            semantic_pred = semantic_logits.detach().argmax(dim=1).squeeze(0)  # [H, W]
            
            # ===== Compute relative depth differences =====
            # GT relative depth
            gt_depth_diff_x = pts_depth[:, :, :-1] - pts_depth[:, :, 1:]  # [1, H, W-1]
            gt_depth_diff_y = pts_depth[:, :-1, :] - pts_depth[:, 1:, :]  # [1, H-1, W]
            
            # Predicted relative depth
            pred_depth_diff_x = depth[:, :, :-1] - depth[:, :, 1:]
            pred_depth_diff_y = depth[:, :-1, :] - depth[:, 1:, :]
            
            # ===== Semantic masks =====
            semantic_same_x = (semantic_pred[:, :-1] == semantic_pred[:, 1:]).float()
            semantic_same_y = (semantic_pred[:-1, :] == semantic_pred[1:, :]).float()
            
            # Valid masks: both neighbors have valid GT depth
            valid_mask_x = (pts_depth[:, :, :-1] > 0) & (pts_depth[:, :, 1:] > 0)
            valid_mask_y = (pts_depth[:, :-1, :] > 0) & (pts_depth[:, 1:, :] > 0)
            
            # Combined masks
            weight_x = (semantic_same_x * valid_mask_x.squeeze(0)).float()
            weight_y = (semantic_same_y * valid_mask_y.squeeze(0)).float()

            loss_refine_x = F.huber_loss(
                pred_depth_diff_x.squeeze(0)[weight_x > 0],
                gt_depth_diff_x.squeeze(0)[weight_x > 0],
                reduction='mean',
                delta=0.1 * args.scale_factor  # Robust to outliers
            )
            loss_refine_y = F.huber_loss(
                pred_depth_diff_y.squeeze(0)[weight_y > 0],
                gt_depth_diff_y.squeeze(0)[weight_y > 0],
                reduction='mean',
                delta=0.1 * args.scale_factor  # Robust to outliers
            )
            
            loss_semantic_depth_refine = loss_refine_x + loss_refine_y
            log_dict['loss_semantic_depth_refine'] = loss_semantic_depth_refine.item()
            loss += args.lambda_semantic_depth_refine * loss_semantic_depth_refine

        loss.backward()
        log_dict['loss'] = loss.item()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            base_keys = ['loss'] if args.only_velodyne else ['loss', "loss_l1", "psnr"]
            loss_component_keys = sorted(k for k in log_dict.keys() if k.startswith("loss_"))
            keys_for_ema = base_keys + [k for k in loss_component_keys if k not in base_keys]
            for key in keys_for_ema:
                value = log_dict.get(key, None)
                if value is None or not np.isfinite(value):
                    continue
                ema_dict_for_log[key] = 0.4 * value + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k: f"{ema_dict_for_log[k]:.{5}f}" for k in sorted(ema_dict_for_log.keys())}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                postfix["points_num"] = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix(postfix)

            if loss_log_interval and ((iteration - first_iter) % loss_log_interval == 0 or iteration == first_iter + 1):
                loss_components = {k: log_dict[k] for k in sorted(log_dict.keys()) if k.startswith("loss")}
                if loss_components:
                    components_str = ", ".join(f"{k}:{loss_components[k]:.6f}" for k in loss_components)
                    progress_bar.write(f"[Iter {iteration:05d}] {components_str}")

            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            # Log and save
            complete_eval(iteration, args.test_iterations, scene, render, (args, background),
                          log_dict, env_map=lidar_raydrop_prior, semantic_head=semantic_head)

            # Densification
            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            if iteration < args.densify_until_iter and (args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.densify_until_num_points):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None

                    if size_threshold is not None:
                        size_threshold = size_threshold // scene.resolution_scales[0]

                    gaussians.densify_and_prune(args.densify_grad_threshold, args.densify_grad_abs_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold,
                                                args.densify_grad_t_threshold)

                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            lidar_raydrop_prior.optimizer.step()
            lidar_raydrop_prior.optimizer.zero_grad(set_to_none=True)
            if semantic_optimizer is not None:
                if semantic_supervised:
                    semantic_optimizer.step()
                semantic_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            if (
                viewer is not None
                and iteration in getattr(args, "rendering_iterations", [])
            ):
                """
                Update viewer with current Gaussian attributes at specified rendering iterations.
                
                Extracts depth, intensity, and semantic features from Gaussians and passes them
                to the viewer for visualization.
                """
                attribute_data = {}

                # Depth: Euclidean distance from origin
                depth_attr = torch.norm(gaussians.get_xyz, dim=1)
                attribute_data["depth"] = depth_attr.detach().cpu().numpy()

                # Intensity: Direct attribute from Gaussians
                intensity_attr = getattr(gaussians, "get_intensity", None)
                if torch.is_tensor(intensity_attr):
                    attribute_data["intensity"] = intensity_attr.detach().cpu().numpy().reshape(-1)

                # Semantic: Mean of semantic feature channels
                semantic_attr = getattr(gaussians, "get_semantic_feature", None)
                if torch.is_tensor(semantic_attr):
                    attribute_data["semantic_mean"] = semantic_attr.mean(dim=1).detach().cpu().numpy()

                try:
                    viewer.update_gaussians(iteration, gaussians, attribute_data)
                except Exception as exc:
                    print(f"[Viewer] Update failed at iteration {iteration}: {exc}")

            if iteration % args.vis_step == 0 or iteration == 1:
                other_img = []

                depth_another = render_pkg['depth_mean'] if args.median_depth else render_pkg['depth_median']
                other_img.append(visualize_depth(depth_another, scale_factor=args.scale_factor))

                if viewpoint_cam.pts_depth is not None:
                    pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth, scale_factor=args.scale_factor)
                    other_img.append(pts_depth_vis)

                feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                v_map = feature[1:4]
                v_norm_map = v_map.norm(dim=0, keepdim=True)
                v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                other_img.append(v_color)

                if args.lambda_raydrop > 0:
                    raydrop_map = render_pkg['raydrop']
                    raydrop_map = visualize_depth(raydrop_map, near=0.01, far=1)
                    other_img.append(raydrop_map)

                    gt_raydrop = 1.0 - (viewpoint_cam.pts_depth > 0).float()
                    gt_raydrop = visualize_depth(gt_raydrop, near=0.01, far=1)
                    other_img.append(gt_raydrop)

                if viewpoint_cam.pts_intensity is not None:
                    intensity_sh_map = render_pkg['intensity_sh']
                    intensity_sh_map = intensity_sh_map * mask
                    intensity_sh_map = intensity_sh_map.clamp(0.0, 1.0).repeat(3, 1, 1)
                    other_img.append(intensity_sh_map)

                    mask = (viewpoint_cam.pts_depth > 0).float()
                    pts_intensity_vis = viewpoint_cam.pts_intensity.clamp(0.0, 1.0).repeat(3, 1, 1)
                    other_img.append(pts_intensity_vis)

                if args.lambda_normal_consistency > 0:
                    other_img.append(render_normal / 2 + 0.5)
                    other_img.append(surf_normal / 2 + 0.5)

                if args.lambda_edge_guidance > 0:
                    gt_x_grad = visualize_depth(gt_x_grad / gt_x_grad.max(), near=0.01, far=1)
                    other_img.append(gt_x_grad)

                depth_var = render_pkg["depth_square"] - depth ** 2
                depth_var = depth_var / depth_var.max()
                depth_var = visualize_depth(depth_var, near=0.01, far=1)
                other_img.append(depth_var)

                if args.lambda_distortion > 0:
                    distortion = distortion / distortion.max()
                    distortion = visualize_depth(distortion, near=0.01, far=1)
                    other_img.append(distortion)

                grid = make_grid([visualize_depth(depth, scale_factor=args.scale_factor),
                                  ] + other_img, nrow=4)

                save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))

            if iteration % args.scale_increase_interval == 0:
                scene.upScale()
                next_w, next_h = scene.getWH()
                lidar_raydrop_prior.upscale(next_h, next_w)

            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt/chkpnt" + str(iteration) + ".pth")
                torch.save((lidar_raydrop_prior.capture(), iteration), scene.model_path + "/ckpt/lidar_raydrop_prior_chkpnt" + str(iteration) + ".pth")
                if semantic_head is not None:
                    torch.save(
                        {"state_dict": semantic_head.state_dict(), "iteration": iteration},
                        scene.model_path + "/ckpt/semantic_head_chkpnt" + str(iteration) + ".pth"
                    )
        

    if viewer is not None:
        viewer.close()


def complete_eval(iteration, test_iterations, scene: Scene, renderFunc, renderArgs, log_dict, env_map=None, semantic_head=None):
    if iteration in test_iterations or iteration == 1:
        scale = scene.resolution_scales[scene.scale_index]
        if iteration < args.iterations:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},)
        else:
            if args.scene_type == "KittiMot":
                num = len(scene.getTrainCameras()) // 2
                eval_train_frame = num // 5
                traincamera = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                      {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:] + traincamera[num:][-eval_train_frame:]})
            else:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                      {'name': 'train', 'cameras': scene.getTrainCameras()})

        h, w = args.hw
        h //= scale
        w //= scale

        semantic_eval = (
            semantic_head is not None
            and getattr(args, "semantic_num_classes", None) is not None
            and getattr(args, "lambda_semantic", 0.0) > 0
        )
        semantic_ignore_index = getattr(args, "semantic_ignore_index", -1)
        semantic_class_count = getattr(args, "semantic_num_classes", 0)
        semantic_mode = None
        if semantic_eval:
            semantic_mode = semantic_head.training
            semantic_head.eval()
            color_table = getattr(args, "semantic_color_map", None)
            semantic_color_base = torch.tensor(color_table, dtype=torch.float32) if color_table is not None else None
        else:
            semantic_color_base = None
        semantic_color_lut = None
        semantic_label_map = None

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),  # for intensity sh
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov)
        ]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for metric in metrics:
                    metric.clear()
                if semantic_eval:
                    semantic_meter = SemanticMeter(num_classes=semantic_class_count, ignore_index=semantic_ignore_index)
                else:
                    semantic_meter = None

                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir, exist_ok=True)

                for idx in range(len(config['cameras']) // 2):
                    cam_front = config['cameras'][idx * 2]
                    cam_back = config['cameras'][idx * 2 + 1]

                    if semantic_eval:
                        depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano, semantic_pred_pano, gt_semantic_pano, semantic_feature_pano = \
                            render_range_map(args, cam_front, cam_back, scene.gaussians, renderFunc, renderArgs, env_map, [h, w],
                                             render_semantic=True, semantic_head=semantic_head, semantic_ignore_index=semantic_ignore_index)
                    else:
                        depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano \
                            = render_range_map(args, cam_front, cam_back, scene.gaussians, renderFunc, renderArgs, env_map, [h, w])
                        semantic_feature_pano = None

                    raydrop_pano_mask = torch.where(raydrop_pano > 0.5, 1, 0)
                    gt_raydrop_pano = torch.where(gt_depth_pano > 0, 0, 1)

                    if iteration == args.iterations:
                        savedir = os.path.join(args.model_path, "ray_drop_datasets")
                        # Save render data: [raydrop, intensity, depth, semantic_pred, semantic_feature (if available)]
                        render_data = torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]], semantic_pred_pano])
                        if semantic_eval and semantic_feature_pano is not None:
                            render_data = torch.cat([render_data, semantic_feature_pano])
                        torch.save(render_data, os.path.join(savedir, f"render_{config['name']}", f"{cam_front.colmap_id:03d}.pt"))
                        torch.save(torch.cat([gt_raydrop_pano, gt_intensity_pano, gt_depth_pano, gt_semantic_pano]), os.path.join(savedir, f"gt", f"{cam_front.colmap_id:03d}.pt"))

                    depth_pano = depth_pano * (1.0 - raydrop_pano_mask)
                    intensity_sh_pano = intensity_sh_pano * (1.0 - raydrop_pano_mask)

                    grid = [visualize_depth(depth_pano[[0]], scale_factor=args.scale_factor),
                            intensity_sh_pano.clamp(0.0, 1.0).repeat(3, 1, 1),
                            visualize_depth(depth_pano[[1]], scale_factor=args.scale_factor),
                            gt_intensity_pano.clamp(0.0, 1.0).repeat(3, 1, 1),
                            visualize_depth(depth_pano[[2]], scale_factor=args.scale_factor),
                            visualize_depth(raydrop_pano_mask, near=0.01, far=1),
                            visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                            visualize_depth(gt_raydrop_pano, near=0.01, far=1)]

                    if semantic_eval and semantic_meter is not None:
                        # Debug: Check semantic data before update
                        if idx == 0 and iteration in test_iterations:
                            pred_unique = torch.unique(semantic_pred_pano)
                            gt_unique = torch.unique(gt_semantic_pano)
                            valid_gt = (gt_semantic_pano != semantic_ignore_index) & (gt_semantic_pano >= 0) & (gt_semantic_pano < semantic_class_count)
                            print(f"\n[Semantic Eval Debug - Iter {iteration}, Frame {cam_front.colmap_id}]")
                            print(f"  Pred unique values: {pred_unique.cpu().numpy()}")
                            print(f"  GT unique values: {gt_unique.cpu().numpy()}")
                            print(f"  Valid GT pixels: {valid_gt.sum().item()}/{gt_semantic_pano.numel()}")
                            print(f"  Pred shape: {semantic_pred_pano.shape}, GT shape: {gt_semantic_pano.shape}")
                        semantic_meter.update(semantic_pred_pano, gt_semantic_pano)
                        desired_device = semantic_pred_pano.device
                        if semantic_color_lut is None or semantic_color_lut.device != desired_device:
                            if semantic_color_base is not None:
                                semantic_color_lut = semantic_color_base.to(desired_device)
                            else:
                                if semantic_label_map is None:
                                    semantic_label_map = load_semantic_label_mapping(getattr(args, "semantic_label_map_path", None),
                                                                                     semantic_ignore_index)
                                semantic_color_lut = semantic_label_map.torch_color_lut(device=desired_device)
                        semantic_pred_color = colorize_semantic_tensor(semantic_pred_pano, semantic_color_lut, semantic_ignore_index)
                        semantic_gt_color = colorize_semantic_tensor(gt_semantic_pano, semantic_color_lut, semantic_ignore_index)
                        grid.extend([
                            semantic_pred_color,
                            semantic_gt_color
                        ])

                    grid = make_grid(grid, nrow=2)
                    save_image(grid, os.path.join(outdir, f"{cam_front.colmap_id:03d}.png"))

                    for i, metric in enumerate(metrics):
                        if i == 0:  # hard code
                            metric.update(raydrop_pano, gt_raydrop_pano)
                        elif i == 1:
                            metric.update(intensity_sh_pano, gt_intensity_pano)
                        elif i == 2:
                            metric.update(depth_pano[[0]], gt_depth_pano)
                        else:
                            metric.update(depth_pano[[i - 3]], gt_depth_pano)

                # Ray drop
                RMSE, Acc, F1 = metrics[0].measure()
                # Intensity sh
                rmse_i_sh, medae_i_sh, lpips_loss_i_sh, ssim_i_sh, psnr_i_sh = metrics[1].measure()
                # depth
                rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
                C_D_mix, F_score_mix = metrics[3].measure().astype(float)
                C_D_mean, F_score_mean = metrics[4].measure().astype(float)
                C_D_median, F_score_median = metrics[5].measure().astype(float)

                metrics_payload = {"split": config['name'], "iteration": iteration,
                                   "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                                   "Point Cloud mix": {"C-D": C_D_mix, "F-score": F_score_mix},
                                   "Point Cloud mean": {"C-D": C_D_mean, "F-score": F_score_mean},
                                   "Point Cloud median": {"C-D": C_D_median, "F-score": F_score_median},
                                   "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                                   "Intensity SH": {"RMSE": rmse_i_sh, "MedAE": medae_i_sh, "LPIPS": lpips_loss_i_sh, "SSIM": ssim_i_sh, "PSNR": psnr_i_sh}}

                if semantic_eval and semantic_meter is not None:
                    semantic_acc, semantic_miou = semantic_meter.measure()
                    metrics_payload["Semantic"] = {"Acc": semantic_acc, "mIoU": semantic_miou}

                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    json.dump(metrics_payload, f, indent=1)

        if semantic_eval and semantic_mode is not None and semantic_mode:
            semantic_head.train()

        torch.cuda.empty_cache()


def refine():
    refine_output_dir = os.path.join(args.model_path, "refine")
    if os.path.exists(refine_output_dir):
        shutil.rmtree(refine_output_dir)
    os.makedirs(refine_output_dir)
    gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")
    train_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_train")

    # Check if semantic feature is available
    use_semantic = False
    semantic_dim = 0
    if getattr(args, "lambda_semantic", 0.0) > 0:
        # Check first file to see if semantic feature is included
        sample_files = list(os.listdir(train_dir))
        if sample_files:
            sample_data = torch.load(os.path.join(train_dir, sample_files[0]))
            # Format: [raydrop, intensity, depth, semantic_pred, semantic_feature (optional)]
            if sample_data.shape[0] > 4:
                use_semantic = True
                semantic_dim = sample_data.shape[0] - 4  # Subtract raydrop(1) + intensity(1) + depth(1) + semantic_pred(1)
                print(f"[Refine] Semantic feature detected: dim={semantic_dim}, will use in UNet input")

    in_channels = 3 + semantic_dim if use_semantic else 3
    unet = UNet(in_channels=in_channels, out_channels=1)
    unet.cuda()
    unet.train()

    raydrop_input_list = []
    raydrop_gt_list = []

    print("Preparing for Raydrop Refinemet ...")
    for data in tqdm(os.listdir(train_dir)):
        render_data = torch.load(os.path.join(train_dir, data))  # [3+1+semantic_dim, H, W] or [4, H, W]
        # Extract: [raydrop, intensity, depth] + [semantic_feature (if available)]
        if use_semantic and render_data.shape[0] > 4:
            # [raydrop, intensity, depth, semantic_pred, semantic_feature]
            unet_input = torch.cat([render_data[:3], render_data[4:]], dim=0).unsqueeze(0)  # [1, 3+semantic_dim, H, W]
        else:
            # [raydrop, intensity, depth]
            unet_input = render_data[:3].unsqueeze(0)  # [1, 3, H, W]
        raydrop_input_list.append(unet_input)
        gt_raydrop = torch.load(os.path.join(gt_dir, data))[[0]].unsqueeze(0)
        raydrop_gt_list.append(gt_raydrop)

    torch.cuda.empty_cache()

    raydrop_input = torch.cat(raydrop_input_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]
    raydrop_gt = torch.cat(raydrop_gt_list, dim=0).cuda().float().contiguous()  # [B, 1, H, W]

    loss_total = []

    refine_bs = None  # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
    refine_epoch = 1000

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_epoch)
    bce_fn = torch.nn.BCELoss()

    print("Start UNet Optimization ...")
    for i in range(refine_epoch):
        optimizer.zero_grad()

        if refine_bs is not None:
            idx = np.random.choice(raydrop_input.shape[0], refine_bs, replace=False)
            input = raydrop_input[idx, ...]
            gt = raydrop_gt[idx, ...]
        else:
            input = raydrop_input
            gt = raydrop_gt

        # random mask
        mask = torch.ones_like(input).to(input.device)
        box_num_max = 32
        box_size_y_max = int(0.1 * input.shape[2])
        box_size_x_max = int(0.1 * input.shape[3])
        for j in range(np.random.randint(box_num_max)):
            box_size_y = np.random.randint(1, box_size_y_max)
            box_size_x = np.random.randint(1, box_size_x_max)
            yi = np.random.randint(input.shape[2] - box_size_y)
            xi = np.random.randint(input.shape[3] - box_size_x)
            mask[:, :, yi:yi + box_size_y, xi:xi + box_size_x] = 0.

        raydrop_refine = unet(input * mask)
        bce_loss = bce_fn(raydrop_refine, gt)
        loss = bce_loss

        loss.backward()

        loss_total.append(loss.item())

        if i % 50 == 0:
            input_mask = torch.where(input > 0.5, 1, 0)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            idx = np.random.choice(range(raydrop_mask.shape[0]))
            grid = [visualize_depth(input_mask[idx], near=0.01, far=1),
                    visualize_depth(raydrop_mask[idx], near=0.01, far=1),
                    visualize_depth(gt[idx], near=0.01, far=1)]
            grid = make_grid(grid, nrow=1)
            save_image(grid, os.path.join(refine_output_dir, f"{i:04d}.png"))
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{log_time}] iter:{i}, lr:{optimizer.param_groups[0]['lr']:.6f}, raydrop loss:{loss.item()}")

        optimizer.step()
        scheduler.step()

    file_path = f"{args.model_path}/ckpt/refine.pth"
    torch.save(unet.state_dict(), file_path)

    torch.cuda.empty_cache()


def refine_test():
    file_path = f"{args.model_path}/ckpt/refine.pth"
    unet = UNet(in_channels=3, out_channels=1)
    unet.load_state_dict(torch.load(file_path))
    unet.cuda()
    unet.eval()

    # Load semantic head if available
    semantic_head = None
    semantic_eval = (
        getattr(args, "lambda_semantic", 0.0) > 0
        and hasattr(args, "semantic_num_classes")
        and args.semantic_num_classes is not None
        and getattr(args, "semantic_dim", 0) > 0
    )
    
    if semantic_eval:
        # Try to load semantic head from checkpoint
        semantic_head_checkpoint = os.path.join(args.model_path, "ckpt", f"semantic_head_chkpnt{args.iterations}.pth")
        if not os.path.exists(semantic_head_checkpoint):
            # Try to find any semantic head checkpoint
            ckpt_dir = os.path.join(args.model_path, "ckpt")
            if os.path.exists(ckpt_dir):
                semantic_checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("semantic_head_chkpnt") and f.endswith(".pth")]
                if semantic_checkpoints:
                    semantic_head_checkpoint = os.path.join(ckpt_dir, sorted(semantic_checkpoints)[-1])
        
        if os.path.exists(semantic_head_checkpoint):
            from head.semantic_head import build_semantic_head
            
            head_type = getattr(args, "semantic_head_type", "conv")
            norm_type = getattr(args, "semantic_norm_type", "bn")
            if norm_type is None:
                norm_type = "bn"
            dropout_prob = getattr(args, "semantic_dropout", 0.1)
            if dropout_prob is None:
                dropout_prob = 0.0

            semantic_kwargs = {
                "norm_type": norm_type,
                "dropout": dropout_prob,
            }

            if head_type.lower() in ("unet", "unet-style", "u-net"):
                base_channels = getattr(args, "semantic_unet_base_channels", 64)
                if base_channels is None:
                    base_channels = 64
                depth_value = getattr(args, "semantic_unet_depth", 3)
                if depth_value is None:
                    depth_value = 3
                semantic_kwargs.update({
                    "base_channels": base_channels,
                    "depth": depth_value,
                })
            else:
                hidden_cfg = getattr(args, "semantic_conv_hidden_dims", (128, 64))
                if hidden_cfg is None:
                    hidden_dims = [128, 64]
                elif isinstance(hidden_cfg, (int, float)):
                    hidden_dims = [int(hidden_cfg)]
                else:
                    hidden_dims = [int(dim) for dim in list(hidden_cfg)]
                semantic_kwargs.update({
                    "hidden_channels": hidden_dims,
                })

            semantic_head = build_semantic_head(
                head_type=head_type,
                in_channels=args.semantic_dim,
                num_classes=args.semantic_num_classes,
                **semantic_kwargs,
            ).cuda()
            semantic_state = torch.load(semantic_head_checkpoint, map_location="cuda")
            semantic_head.load_state_dict(semantic_state["state_dict"])
            semantic_head.eval()
            print(f"[Refine Test] Loaded semantic head from {semantic_head_checkpoint}")
        else:
            semantic_eval = False
            print(f"[Refine Test] Semantic head checkpoint not found, skipping semantic evaluation")

    for mode in ["train", "test"]:
        outdir = os.path.join(args.model_path, "eval", f"{mode}_refine_render")
        os.makedirs(outdir, exist_ok=True)

        test_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_{mode}")
        gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")

        test_input_list = []
        gt_list = []
        semantic_pred_list = []
        gt_semantic_list = []
        name_list = []
        print(f"Preparing for Refinemet {mode} ...")
        for data in tqdm(os.listdir(test_dir)):
            render_data = torch.load(os.path.join(test_dir, data))  # [4+semantic_dim, H, W] or [4, H, W]: [raydrop, intensity, depth, semantic_pred, semantic_feature (optional)]
            gt_data = torch.load(os.path.join(gt_dir, data))  # [4, H, W]: [gt_raydrop, gt_intensity, gt_depth, gt_semantic]
            
            # Extract data: [raydrop, intensity, depth] for UNet input
            test_input_list.append(render_data[:3].unsqueeze(0))  # [1, 3, H, W]
            gt_list.append(gt_data[:3].unsqueeze(0))  # [1, 3, H, W]
            
            # Extract semantic data if available
            if semantic_eval and semantic_head is not None:
                if render_data.shape[0] >= 4:
                    semantic_pred_list.append(render_data[[3]].long().unsqueeze(0))  # [1, 1, H, W]
                else:
                    semantic_pred_list.append(None)
                
                if gt_data.shape[0] >= 4:
                    gt_semantic_list.append(gt_data[[3]].long().unsqueeze(0))  # [1, 1, H, W]
                else:
                    gt_semantic_list.append(None)
            
            name_list.append(data)

        test_input = torch.cat(test_input_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]
        gt = torch.cat(gt_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
        ]
        
        # Add semantic meter if semantic evaluation is enabled
        semantic_meter = None
        semantic_color_lut = None
        if semantic_eval and semantic_head is not None:
            semantic_ignore_index = getattr(args, "semantic_ignore_index", -1)
            semantic_class_count = getattr(args, "semantic_num_classes", 0)
            semantic_meter = SemanticMeter(num_classes=semantic_class_count, ignore_index=semantic_ignore_index)
            
            # Prepare semantic color LUT for visualization
            color_table = getattr(args, "semantic_color_map", None)
            semantic_color_base = torch.tensor(color_table, dtype=torch.float32) if color_table is not None else None
            if semantic_color_base is None:
                semantic_label_map = load_semantic_label_mapping(
                    getattr(args, "semantic_label_map_path", None),
                    semantic_ignore_index
                )
                semantic_color_lut = semantic_label_map.torch_color_lut(device="cuda")
            else:
                semantic_color_lut = semantic_color_base.to("cuda")

        with torch.no_grad():
            raydrop_refine = unet(test_input)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            for idx in tqdm(range(gt.shape[0])):
                raydrop_pano = raydrop_refine[idx, [0]]
                raydrop_pano_mask = raydrop_mask[idx, [0]]
                intensity_pano = test_input[idx, [1]] * (1 - raydrop_pano_mask)
                depth_pano = test_input[idx, [2]] * (1 - raydrop_pano_mask)

                gt_raydrop_pano = gt[idx, [0]]
                gt_intensity_pano = gt[idx, [1]]
                gt_depth_pano = gt[idx, [2]]

                grid = [visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                        visualize_depth(depth_pano, scale_factor=args.scale_factor),
                        gt_intensity_pano.clamp(0, 1).repeat(3, 1, 1),
                        intensity_pano.clamp(0, 1).repeat(3, 1, 1), ]

                # Semantic evaluation and visualization
                if semantic_eval and semantic_meter is not None:
                    if semantic_pred_list[idx] is not None and gt_semantic_list[idx] is not None:
                        semantic_pred = semantic_pred_list[idx]  # [1, 1, H, W]
                        gt_semantic = gt_semantic_list[idx]  # [1, 1, H, W]
                        
                        # Flatten for meter update
                        semantic_pred_flat = semantic_pred.squeeze(0).squeeze(0)  # [H, W]
                        gt_semantic_flat = gt_semantic.squeeze(0).squeeze(0)  # [H, W]
                        
                        # Update semantic meter
                        semantic_meter.update(semantic_pred_flat, gt_semantic_flat)
                        
                        # Colorize semantic predictions and GT for visualization
                        if semantic_color_lut is not None:
                            semantic_pred_color = colorize_semantic_tensor(
                                semantic_pred_flat, semantic_color_lut, semantic_ignore_index
                            )
                            semantic_gt_color = colorize_semantic_tensor(
                                gt_semantic_flat, semantic_color_lut, semantic_ignore_index
                            )
                            grid.extend([semantic_pred_color, semantic_gt_color])

                grid = make_grid(grid, nrow=2, padding=0)
                save_image(grid, os.path.join(outdir, name_list[idx].replace(".pt", ".png")))
                save_ply(pano_to_lidar(depth_pano, args.vfov, (-180, 180)),
                         os.path.join(outdir, name_list[idx].replace(".pt", ".ply")))

                for i, metric in enumerate(metrics):
                    if i == 0:  # hard code
                        metric.update(raydrop_pano, gt_raydrop_pano)
                    elif i == 1:
                        metric.update(intensity_pano, gt_intensity_pano)
                    else:
                        metric.update(depth_pano, gt_depth_pano)

            # Ray drop
            RMSE, Acc, F1 = metrics[0].measure()
            # Intensity
            rmse_i, medae_i, lpips_loss_i, ssim_i, psnr_i = metrics[1].measure()
            # depth
            rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
            C_D, F_score = metrics[3].measure().astype(float)

            # Build metrics payload
            metrics_payload = {
                "split": f"{mode}",
                "iteration": "refine",
                "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                "Point Cloud": {"C-D": C_D, "F-score": F_score},
                "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                "Intensity": {"RMSE": rmse_i, "MedAE": medae_i, "LPIPS": lpips_loss_i, "SSIM": ssim_i, "PSNR": psnr_i},
            }
            
            # Add semantic metrics if available
            # Note: Semantic evaluation requires semantic feature from rendering,
            # which is not available in refine_test since it uses pre-saved data.
            # To enable semantic evaluation, semantic feature needs to be included
            # in the saved data or re-rendered with semantic feature.
            if semantic_eval and semantic_meter is not None and semantic_meter.total_samples > 0:
                semantic_acc, semantic_miou = semantic_meter.measure()
                metrics_payload["Semantic"] = {"Acc": semantic_acc, "mIoU": semantic_miou}

            with open(os.path.join(outdir, "metrics.json"), "w") as f:
                json.dump(metrics_payload, f, indent=1)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--show_log", action="store_true")
    args_read, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args_read.base_config)
    second_conf = OmegaConf.load(args_read.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    OmegaConf.update(args, "start_checkpoint", args_read.start_checkpoint)
    OmegaConf.update(args, "debug_cuda", args_read.debug_cuda)
    OmegaConf.update(args, "test_only", args_read.test_only)
    OmegaConf.update(args, "median_depth", args_read.median_depth)

    args.model_path = os.path.join(args.model_path, f"seq_{args.sequence_id}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(args.model_path) and not args.test_only and args.start_checkpoint is None:
        shutil.rmtree(args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    if not args.dynamic:
        args.t_grad = False

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if args.test_only:
        args.shuffle = False
        for iteration in args.checkpoint_iterations:
            path = os.path.join(args.model_path, "ckpt", f"chkpnt{iteration}.pth")
            if os.path.exists(path):
                args.start_checkpoint = path
                resolution_idx = len(args.resolution_scales) - 1
                for i in range(iteration // args.scale_increase_interval):
                    resolution_idx = max(0, resolution_idx - 1)
        args.resolution_scales = [args.resolution_scales[resolution_idx]]
        with open(os.path.join(args.model_path, "scale_factor.txt"), 'r') as file:
            data = file.read()
            args.scale_factor = float(data)

    if args.debug_cuda:
        args.resolution_scales = [args.resolution_scales[-1]]

    if args.exhaust_test:
        args.test_iterations += [i for i in range(0, args.iterations, args.test_interval)]

    print(args)

    print("Optimizing " + args.model_path)
    with open(os.path.join(args.model_path, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    if os.path.exists(os.path.join(args.model_path, 'ray_drop_datasets')) and not args.test_only:
        shutil.rmtree(os.path.join(args.model_path, 'ray_drop_datasets'))
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_train'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_test'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ckpt'), exist_ok=True)

    if not args.test_only and not args.debug_cuda and not args_read.show_log:
        f = open(os.path.join(args.model_path, 'log.txt'), 'w')
        sys.stdout = f
        sys.stderr = f
    seed_everything(args.seed)

    if not args.test_only:
        training(args)

    # Training done
    print("\nTraining complete.")

    if not args.test_only:
        refine()
    refine_test()
    print("\nRefine complete.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
