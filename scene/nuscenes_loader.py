#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from scene.kitti360_loader import pad_poses, unpad_poses, transform_poses_pca


def _get_scene_samples(nusc: NuScenes, scene_name: str) -> List[dict]:
    """Collect all sample records for the target nuScenes scene.

    Args:
        nusc: Initialized nuScenes API instance.
        scene_name: Scene identifier such as ``scene-0103``.

    Returns:
        List of sample dictionaries ordered chronologically.
    """
    scene_record = next((scene for scene in nusc.scene if scene["name"] == scene_name), None)
    if scene_record is None:
        raise ValueError(f"nuScenes scene {scene_name} not found.")

    sample_token = scene_record["first_sample_token"]
    samples: List[dict] = []
    while sample_token:
        sample_record = nusc.get("sample", sample_token)
        samples.append(sample_record)
        sample_token = sample_record["next"]
    return samples


def _compute_lidar_global_matrix(nusc: NuScenes, sample_data_token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute transformation matrices for the LIDAR_TOP sensor.

    Args:
        nusc: Initialized nuScenes API instance.
        sample_data_token: Token of the LIDAR_TOP sample_data.

    Returns:
        Tuple containing (lidar2global, global2lidar, lidar_timestamp).
    """
    lidar_sd = nusc.get("sample_data", sample_data_token)
    calibrated_sensor = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    lidar2ego = transform_matrix(calibrated_sensor["translation"], Quaternion(calibrated_sensor["rotation"]), inverse=False)
    ego2global = transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=False)
    lidar2global = ego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global)
    timestamp = lidar_sd["timestamp"] * 1e-6
    return lidar2global, global2lidar, timestamp


def _load_lidar_points(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load lidar points and intensities from a nuScenes .bin file.

    Args:
        file_path: Absolute filesystem path to the lidar binary file.

    Returns:
        Tuple ``(points_xyz, intensity)`` where
            * ``points_xyz`` is an ``(N, 3)`` array in the sensor frame.
            * ``intensity`` is a ``(N,)`` array with per-point reflectance.
    """
    lidar_points = LidarPointCloud.from_file(file_path)
    points = lidar_points.points[:3, :].T
    intensity = lidar_points.points[3, :].copy()
    return points, intensity


def _load_lidar_semantics(nusc: NuScenes, sample_data_token: str) -> np.ndarray:
    """Load per-point semantic labels for a nuScenes LIDAR_TOP sample.

    Args:
        nusc: Initialized nuScenes API instance.
        sample_data_token: Token referencing the LIDAR_TOP sample_data entry.

    Returns:
        Semantic label array of shape ``(N,)`` with ``int64`` dtype. If the dataset does not
        provide lidar segmentation for the sample, an empty array is returned.
    """
    lidarseg_records = [ls for ls in nusc.lidarseg if ls.get("sample_data_token") == sample_data_token]
    if len(lidarseg_records) > 0:
        lidarseg_record = lidarseg_records[0]
        lidarseg_path = os.path.join(nusc.dataroot, lidarseg_record["filename"])
        if os.path.isfile(lidarseg_path):
            # Found lidarseg data even though token was missing
            semantics = np.fromfile(lidarseg_path, dtype=np.uint8).astype(np.int64)
            return semantics
    else:
        raise FileNotFoundError(f"nuScenes lidar segmentation file not found: {lidarseg_path}")



def readNuScenesInfo(args) -> SceneInfo:
    """Create a ``SceneInfo`` instance for nuScenes data.

    Args:
        args: Configuration object containing nuScenes-specific fields. The
            following attributes are required:
            * ``source_path``: nuScenes dataroot or symbolic link within the project.
            * ``scene_name``: Target scene identifier (e.g., ``scene-0103``).
            * ``nuscenes_version``: nuScenes split string (``v1.0-trainval``, etc.).
            * ``start_frame`` / ``end_frame``: Optional frame index bounds. ``-1`` denotes the last frame.
            * ``num_pts``: Number of points sampled for the global point cloud.
            * ``time_duration``: Two-element list defining normalized temporal bounds.
            * ``testhold``: Positive integer controlling the evaluation sampling stride.
            * ``vfov`` / ``hfov`` / ``hw``: LiDAR panorama configuration.
            * ``cam_num``: Number of virtual cameras per frame (should be ``2`` for 360° splitting).

    Returns:
        SceneInfo populated with nuScenes lidar data, camera lists, and normalization metadata.
    """
    if args.scene_name is None:
        raise ValueError("nuScenes loader requires args.scene_name to be specified.")
    if args.vfov is None or args.hfov is None:
        raise ValueError("nuScenes loader requires args.vfov and args.hfov.")
    if args.hw is None:
        raise ValueError("nuScenes loader requires args.hw to describe the panorama resolution.")

    dataroot = os.path.expanduser(args.source_path)
    if not os.path.isdir(dataroot):
        raise FileNotFoundError(f"nuScenes dataroot {dataroot} not found.")

    nusc = NuScenes(version=args.nuscenes_version, dataroot=dataroot, verbose=False)
    samples = _get_scene_samples(nusc, args.scene_name)

    start_idx = max(0, int(args.start_frame)) if args.start_frame is not None else 0
    end_idx = int(args.end_frame) if (args.end_frame is not None and int(args.end_frame) >= 0) else len(samples) - 1
    end_idx = min(end_idx, len(samples) - 1)

    frame_records = samples[start_idx:end_idx + 1]
    if len(frame_records) == 0:
        raise RuntimeError("No frames selected for nuScenes processing. Check start_frame/end_frame settings.")

    frames = len(frame_records)
    args.frames = frames

    point_list: List[np.ndarray] = []
    point_time: List[np.ndarray] = []
    cam_infos: List[CameraInfo] = []

    for frame_idx, sample in enumerate(tqdm(frame_records, desc="Reading nuScenesInfo")):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = nusc.get("sample_data", lidar_token)
        file_path = os.path.join(nusc.dataroot, lidar_sd["filename"])
        lidar2global, global2lidar, timestamp = _compute_lidar_global_matrix(nusc, lidar_token)

        points_lidar, intensity = _load_lidar_points(file_path)
        semantics = _load_lidar_semantics(nusc, lidar_token)
        mask = np.linalg.norm(points_lidar, axis=1) > 1.5
        masked_size = int(mask.sum())
        points_lidar = points_lidar[mask]
        intensity = intensity[mask]
        if semantics.size == mask.shape[0]:
            semantics = semantics[mask]
        elif semantics.size == masked_size:
            semantics = semantics.copy()
        else:
            semantics = np.full(masked_size, fill_value=-1, dtype=np.int64) # fill with -1 if size mismatch

        points_homo = np.concatenate([points_lidar, np.ones_like(points_lidar[:, :1])], axis=-1)
        points_world = (lidar2global @ points_homo.T).T[:, :3]
        point_list.append(points_world)

        normalized_time = args.time_duration[0] + (args.time_duration[1] - args.time_duration[0]) * frame_idx / max(frames - 1, 1)
        point_time.append(np.full((points_world.shape[0], 1), normalized_time, dtype=np.float32))

        w2l = global2lidar
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3]
        points_cam = points_world @ R + T

        uid_base = frame_idx + start_idx
        cam_infos.append(CameraInfo(uid=uid_base, R=R.copy(), T=T.copy(),
                                    timestamp=timestamp, pointcloud_camera=points_cam.copy(),
                                    intensity=intensity.copy(), towards="forward",
                                    semantic=semantics.copy()))

        R_back = R @ np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, -1]], dtype=np.float32)
        T_back = T * np.array([-1, 1, -1], dtype=np.float32)
        points_cam_back = points_world @ R_back + T_back
        cam_infos.append(CameraInfo(uid=uid_base + frames, R=R_back.copy(), T=T_back.copy(),
                                    timestamp=timestamp, pointcloud_camera=points_cam_back.copy(),
                                    intensity=intensity.copy(), towards="backward",
                                    semantic=semantics.copy()))

        if args.debug_cuda and frame_idx >= 15:
            break

    pointcloud = np.concatenate(point_list, axis=0)
    point_timestamp = np.concatenate(point_time, axis=0)

    num_pts = min(int(args.num_pts), pointcloud.shape[0])
    indices = np.random.choice(pointcloud.shape[0], num_pts, replace=False)
    pointcloud = pointcloud[indices]
    point_timestamp = point_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))

    if not args.test_only:
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, args.dynamic)
        np.savez(os.path.join(args.model_path, "transform_poses_pca.npz"), transform=transform, scale_factor=scale_factor)
        c2ws = pad_poses(c2ws)
    else:
        data = np.load(os.path.join(args.model_path, "transform_poses_pca.npz"))
        transform = data["transform"]
        scale_factor = data["scale_factor"].item()
        c2ws = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2ws)
        c2ws[:, :3, 3] *= scale_factor

    for idx, cam_info in enumerate(cam_infos):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])
        cam_info.T[:] = w2c[:3, 3]
        if cam_info.pointcloud_camera is not None:
            cam_info.pointcloud_camera[:] *= scale_factor

    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    args.scale_factor = float(scale_factor)

    mod = args.cam_num
    test_stride = max(1, int(args.testhold))
    if args.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if ((idx // mod) % test_stride) != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if ((idx // mod) % test_stride) == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if ((idx // mod) % test_stride) == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization["radius"] = 1

    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not args.test_only:
        rgbs = np.random.random((pointcloud.shape[0], 3)) * 255.0
        storePly(ply_path, pointcloud, rgbs, point_timestamp)

    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    time_interval = (args.time_duration[1] - args.time_duration[0]) / max(frames - 1, 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)
    return scene_info

