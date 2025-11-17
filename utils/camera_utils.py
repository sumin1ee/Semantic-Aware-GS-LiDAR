#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import torch
import cv2
from scene.cameras import Camera
import numpy as np
from scene.scene_utils import CameraInfo
from tqdm import tqdm
from torchvision.utils import save_image


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    orig_h, orig_w = args.hw

    if args.resolution == -1:
        global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    vfov = args.vfov
    hfov = args.hfov
    if cam_info.pointcloud_camera is not None:
        intensity = cam_info.intensity
        if intensity is None:
            intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

        w = resolution[0]
        h = resolution[1]

        pts_depth = np.zeros([1, h, w])
        pts_intensity = np.zeros([1, h, w])
        pts_semantic = None
        point_camera = cam_info.pointcloud_camera
        x = point_camera[:, 0]
        y = point_camera[:, 1]
        z = point_camera[:, 2]
        phi = np.arctan2(x, z)
        theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
        VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
        HFOV_max = hfov[1] * np.pi / 180
        HFOV_min = hfov[0] * np.pi / 180

        theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
        phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
        uvz = np.stack((theta, phi, r, intensity), 1)

        semantic_values = cam_info.semantic
        if semantic_values is not None:
            semantic_values = semantic_values.reshape(-1)
            uvz = np.concatenate((uvz, semantic_values[:, None]), axis=1)
            pts_semantic = np.full([1, h, w], fill_value=-1, dtype=np.int64)

        uvz = uvz[uvz[:, 0] >= -0.5]
        uvz = uvz[uvz[:, 0] < h - 0.5]
        uvz = uvz[uvz[:, 1] >= -0.5]
        uvz = uvz[uvz[:, 1] < w - 0.5]
        uv = uvz[:, :2]
        uv = np.around(uv).astype(int)

        for i in range(uv.shape[0]):
            x, y = uv[i]
            depth_value = uvz[i, 2]
            intensity_value = uvz[i, 3]
            semantic_value = int(uvz[i, 4]) if semantic_values is not None else None
            if pts_depth[0, x, y] == 0 or depth_value < pts_depth[0, x, y]:
                pts_depth[0, x, y] = depth_value
                pts_intensity[0, x, y] = intensity_value
                if pts_semantic is not None:
                    pts_semantic[0, x, y] = semantic_value

        pts_depth = torch.from_numpy(pts_depth).float().cuda()
        pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
        if pts_semantic is not None:
            pts_semantic = torch.from_numpy(pts_semantic).long().cuda()
    else:
        pts_depth = None
        pts_intensity = None
        pts_semantic = None

    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        vfov=vfov,
        hfov=hfov,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        pts_depth=pts_depth,
        pts_intensity=pts_intensity,
        pts_semantic=pts_semantic,
        towards=cam_info.towards
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list
