#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from scene.kitti360_loader import pad_poses, unpad_poses, transform_poses_pca
from scene.scene_utils import CameraInfo, SceneInfo, fetchPly, getNerfppNorm, storePly
from utils.semantic_mapping import load_semantic_label_mapping


def _read_calibration(calib_path: str) -> Dict[str, np.ndarray]:
    """Parse SemanticKITTI calibration text file.

    Args:
        calib_path: Absolute path to the sequence calibration file (e.g., ``calib.txt``).

    Returns:
        Dict where keys are calibration entry names (``Tr``, ``R_rect`` ...) and values are ``4x4`` matrices.
    """
    calib: Dict[str, np.ndarray] = {}
    with open(calib_path, "r") as file:
        for line in file:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", maxsplit=1)
            data = np.fromstring(value, sep=" ")
            if data.shape[0] == 12:
                matrix = data.reshape(3, 4)
                matrix = np.vstack([matrix, np.array([0, 0, 0, 1], dtype=np.float64)])
            elif data.shape[0] == 9:
                matrix = data.reshape(3, 3)
                matrix = np.block([[matrix, np.zeros((3, 1))],
                                   [np.zeros((1, 3)), np.ones((1, 1))]])
            else:
                continue
            calib[key.strip()] = matrix
    if "Tr" not in calib:
        raise FileNotFoundError(f"SemanticKITTI calibration {calib_path} missing 'Tr' entry.")
    return calib


def _read_poses(pose_path: str) -> List[np.ndarray]:
    """Load per-frame poses from SemanticKITTI pose files.

    Args:
        pose_path: Absolute path to ``poses/{sequence}.txt``.

    Returns:
        List of ``4x4`` transformation matrices mapping camera-0 to the world frame.
    """
    poses: List[np.ndarray] = []
    with open(pose_path, "r") as file:
        for line in file:
            data = np.fromstring(line.strip(), sep=" ")
            if data.size != 12:
                continue
            pose = data.reshape(3, 4)
            pose = np.vstack([pose, np.array([0, 0, 0, 1], dtype=np.float64)])
            poses.append(pose)
    if not poses:
        raise FileNotFoundError(f"No poses found at {pose_path}")
    return poses


def _semantic_from_label_file(label_path: str, label_mapping) -> np.ndarray:
    """Load SemanticKITTI per-point semantic ids.

    Args:
        label_path: Absolute path to the ``.label`` file.

    Returns:
        ``(N,)`` int64 numpy array storing semantic ids.
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantics = (labels & 0xFFFF).astype(np.int64)
    semantics = label_mapping.remap(semantics)
    return semantics


def _get_semantickitti_paths(source_path: str, sequence: str) -> Tuple[str, str, str]:
    """Resolve canonical SemanticKITTI folders for a sequence.

    Args:
        source_path: Dataset root containing ``sequences`` and ``poses`` folders.
        sequence: Zero-padded sequence identifier (e.g., ``"00"``).

    Returns:
        Tuple ``(velodyne_dir, label_dir, calib_path)`` for the sequence.
    """
    seq_dir = os.path.join(source_path, "sequences", sequence)
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    label_dir = os.path.join(seq_dir, "labels")
    calib_path = os.path.join(seq_dir, "calib.txt")
    pose_path = os.path.join(seq_dir, "poses.txt")
    if not os.path.isdir(velodyne_dir):
        raise FileNotFoundError(f"SemanticKITTI velodyne dir missing: {velodyne_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"SemanticKITTI label dir missing: {label_dir}")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"SemanticKITTI calib file missing: {calib_path}")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"SemanticKITTI pose file missing: {pose_path}")
    return velodyne_dir, label_dir, calib_path, pose_path


def readSemanticKITTIInfo(args) -> SceneInfo:
    """Build SceneInfo for SemanticKITTI LiDAR range-map training.

    Args:
        args: Namespace with SemanticKITTI-specific attributes:
            - ``source_path``: dataset root containing ``sequences`` and ``poses``.
            - ``sequence_id``: sequence number (e.g., ``"00"``).
            - ``start_frame`` / ``end_frame``: optional frame bounds (``-1`` -> last).
            - ``num_pts``: number of global points to subsample for initialization.
            - ``time_duration``: normalized temporal span.
            - ``testhold``: stride for evaluation sampling.
            - ``vfov`` / ``hfov`` / ``cam_num``: LiDAR panorama configuration.

    Returns:
        Populated ``SceneInfo`` structure compatible with the renderer.
    """
    if args.sequence_id is None:
        raise ValueError("SemanticKITTI loader requires args.sequence_id.")
    if args.vfov is None or args.hfov is None:
        raise ValueError("SemanticKITTI loader requires args.vfov/hfov.")

    sequence = f"{int(args.sequence_id):02d}"
    source_path = os.path.expanduser(args.source_path)
    velodyne_dir, label_dir, calib_path, pose_path = _get_semantickitti_paths(source_path, sequence)

    calib = _read_calibration(calib_path)
    poses_cam = _read_poses(pose_path)
    T_cam0_velo = calib["Tr"]

    lidar2world_all = [pose @ T_cam0_velo for pose in poses_cam]

    bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.label")))
    if len(bin_files) != len(label_files):
        raise RuntimeError("SemanticKITTI velodyne/label counts mismatch.")

    start_idx = max(0, int(getattr(args, "start_frame", 0) or 0))
    end_frame = getattr(args, "end_frame", -1)
    if end_frame is None or int(end_frame) < 0:
        end_idx = len(bin_files) - 1
    else:
        end_idx = min(len(bin_files) - 1, int(end_frame))
    frame_files = bin_files[start_idx:end_idx + 1]
    label_files = label_files[start_idx:end_idx + 1]
    poses_slice = lidar2world_all[start_idx:end_idx + 1]

    frames = len(frame_files)
    if frames == 0:
        raise RuntimeError("No SemanticKITTI frames selected.")
    args.frames = frames

    semantic_map_path = getattr(args, "semantic_label_map_path", None)
    semantic_ignore_index = getattr(args, "semantic_ignore_index", -1)
    semantic_mapping = load_semantic_label_mapping(semantic_map_path, semantic_ignore_index)
    args.semantic_label_map_path = getattr(semantic_mapping, "config_path", semantic_map_path)
    if getattr(args, "semantic_num_classes", None) in (None, 0):
        args.semantic_num_classes = int(semantic_mapping.num_classes)
    if getattr(args, "semantic_class_names", None) is None:
        args.semantic_class_names = list(semantic_mapping.class_names)
    args.semantic_color_map = semantic_mapping.color_lut.astype(float).tolist()

    point_list: List[np.ndarray] = []
    point_time: List[np.ndarray] = []
    cam_infos: List[CameraInfo] = []

    for frame_idx, (bin_path, label_path, lidar2world) in enumerate(
            tqdm(zip(frame_files, label_files, poses_slice),
                 total=frames,
                 desc="Reading SemanticKITTI")):
        points_raw = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        intensity = points_raw[:, 3]
        points = points_raw[:, :3]

        semantics = _semantic_from_label_file(label_path, semantic_mapping)
        if semantics.shape[0] != points.shape[0]:
            raise RuntimeError(f"Semantic count mismatch at {bin_path}")

        mask = np.linalg.norm(points, axis=1) > 1.5
        points = points[mask]
        intensity = intensity[mask]
        semantics = semantics[mask]

        points_h = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
        points_world = (lidar2world @ points_h.T).T[:, :3]
        point_list.append(points_world)

        timestamp = args.time_duration[0] + (args.time_duration[1] - args.time_duration[0]) * frame_idx / max(frames - 1, 1)
        point_time.append(np.full((points_world.shape[0], 1), timestamp, dtype=np.float32))

        w2l = np.array([0, -1, 0, 0,
                        0, 0, -1, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(lidar2world)
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3]
        points_cam = points_world @ R + T

        uid = frame_idx
        cam_infos.append(CameraInfo(uid=uid,
                                    R=R.copy(),
                                    T=T.copy(),
                                    timestamp=timestamp,
                                    pointcloud_camera=points_cam.copy(),
                                    intensity=intensity.copy(),
                                    towards="forward",
                                    semantic=semantics.copy()))

        R_back = R @ np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, -1]], dtype=np.float32)
        T_back = T * np.array([-1, 1, -1], dtype=np.float32)
        points_cam_back = points_world @ R_back + T_back
        cam_infos.append(CameraInfo(uid=uid + frames,
                                    R=R_back.copy(),
                                    T=T_back.copy(),
                                    timestamp=timestamp,
                                    pointcloud_camera=points_cam_back.copy(),
                                    intensity=intensity.copy(),
                                    towards="backward",
                                    semantic=semantics.copy()))

        if getattr(args, "debug_cuda", False) and frame_idx >= 15:
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
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, getattr(args, "dynamic", False))
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
        cam_info.pointcloud_camera[:] *= scale_factor

    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    args.scale_factor = float(scale_factor)

    mod = args.cam_num
    test_stride = max(1, int(getattr(args, "testhold", 4)))
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

