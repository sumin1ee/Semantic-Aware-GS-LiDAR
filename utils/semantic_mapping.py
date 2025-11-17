#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
"""Utilities for loading SemanticKITTI-style label remapping and color tables."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SEMANTIC_KITTI_LABEL_MAP_REL_PATH = os.path.join("configs", "semantickitti_label_map.yaml")
_MAPPING_CACHE: Dict[Tuple[str, int], "SemanticLabelMapping"] = {}


class SemanticLabelMapping:
    """Container for SemanticKITTI label remapping and colors.

    Args:
        name: Dataset name.
        learning_map: Mapping from raw ids to contiguous train ids.
        learning_map_inv: Reverse mapping from contiguous ids to representative raw ids.
        learning_ignore: Flags indicating whether a contiguous id should be ignored.
        labels: Dictionary from raw ids to string names.
        color_map: Dictionary from raw ids to BGR colors in [0, 255].
        ignore_index: Value used to mark ignored samples.
    """

    def __init__(self,
                 name: str,
                 learning_map: Dict[int, int],
                 learning_map_inv: Dict[int, int],
                 learning_ignore: Dict[int, bool],
                 labels: Dict[int, str],
                 color_map: Dict[int, Tuple[int, int, int]],
                 ignore_index: int = -1):
        self.name = name
        self.ignore_index = ignore_index

        self.learning_map = {int(k): int(v) for k, v in learning_map.items()}
        self.learning_map_inv = {int(k): int(v) for k, v in learning_map_inv.items()}
        self.learning_ignore = {int(k): bool(v) for k, v in learning_ignore.items()}
        self.labels = {int(k): str(v) for k, v in labels.items()}
        self.color_map_raw = {int(k): tuple(int(x) for x in v) for k, v in color_map.items()}

        self.num_classes = max(self.learning_map.values()) + 1

        forward_table_size = max(max(self.learning_map.keys()) + 1, 256)
        self.forward_table = np.full(forward_table_size, fill_value=self.ignore_index, dtype=np.int64)
        for raw_id, mapped_id in self.learning_map.items():
            if raw_id < forward_table_size:
                self.forward_table[raw_id] = mapped_id

        self.ignore_mask = np.zeros(self.num_classes, dtype=bool)
        for class_id, flag in self.learning_ignore.items():
            if 0 <= class_id < self.num_classes:
                self.ignore_mask[class_id] = flag

        self.class_names = []
        for class_id in range(self.num_classes):
            raw_id = self.learning_map_inv.get(class_id)
            class_name = self.labels.get(raw_id, f"class_{class_id}")
            self.class_names.append(class_name)

        self.color_lut = self._build_color_lut()

    def _build_color_lut(self) -> np.ndarray:
        lut = np.zeros((self.num_classes, 3), dtype=np.float32)
        for class_id in range(self.num_classes):
            raw_id = self.learning_map_inv.get(class_id)
            color_bgr = self.color_map_raw.get(raw_id)
            if color_bgr is None:
                lut[class_id] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                continue
            color_rgb = np.array(color_bgr[::-1], dtype=np.float32) / 255.0
            lut[class_id] = color_rgb
        return lut

    def remap(self, raw_semantics: np.ndarray) -> np.ndarray:
        """Map raw SemanticKITTI ids to contiguous training ids."""
        semantics = np.full(raw_semantics.shape, fill_value=self.ignore_index, dtype=np.int64)
        valid_mask = (raw_semantics >= 0) & (raw_semantics < self.forward_table.shape[0])
        semantics[valid_mask] = self.forward_table[raw_semantics[valid_mask]]

        if self.ignore_mask.any():
            ignore_mask = (semantics >= 0) & (semantics < self.num_classes) & self.ignore_mask[semantics]
            semantics[ignore_mask] = self.ignore_index
        return semantics

    def torch_color_lut(self, device=None):
        """Return color look-up table as a torch tensor in RGB [0, 1]."""
        if torch is None:
            raise ImportError("torch is required to build semantic color LUT.")
        lut = torch.from_numpy(self.color_lut)
        if device is not None:
            lut = lut.to(device)
        return lut

    @property
    def config_dict(self) -> Dict[str, object]:
        """Return serializable summary for logging/debug."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "class_names": self.class_names,
        }


def _resolve_config_path(config_path: Optional[str]) -> str:
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, SEMANTIC_KITTI_LABEL_MAP_REL_PATH)
    return os.path.abspath(os.path.expanduser(config_path))


def load_semantic_label_mapping(config_path: Optional[str], ignore_index: int) -> SemanticLabelMapping:
    """Load (and cache) label mapping from YAML."""
    resolved_path = _resolve_config_path(config_path)
    cache_key = (resolved_path, int(ignore_index))
    if cache_key in _MAPPING_CACHE:
        return _MAPPING_CACHE[cache_key]

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Semantic label map not found: {resolved_path}")

    cfg = OmegaConf.load(resolved_path)
    required_fields = ["labels", "learning_map", "learning_map_inv", "learning_ignore", "color_map"]
    for field in required_fields:
        if field not in cfg:
            raise KeyError(f"Semantic label map {resolved_path} missing field '{field}'.")

    mapping = SemanticLabelMapping(
        name=str(getattr(cfg, "name", "semantic_dataset")),
        learning_map=dict(cfg.learning_map),
        learning_map_inv=dict(cfg.learning_map_inv),
        learning_ignore=dict(cfg.learning_ignore),
        labels=dict(cfg.labels),
        color_map=dict(cfg.color_map),
        ignore_index=ignore_index,
    )
    setattr(mapping, "config_path", resolved_path)
    _MAPPING_CACHE[cache_key] = mapping
    return mapping


def colorize_semantic_tensor(labels, color_lut, ignore_index: int):
    """Map semantic ids to RGB tensor using provided color LUT."""
    if torch is None:
        raise ImportError("torch is required for semantic colorization.")

    if labels.dim() == 3 and labels.shape[0] == 1:
        label_map = labels.squeeze(0)
    else:
        label_map = labels

    valid_mask = (label_map >= 0) & (label_map < color_lut.shape[0])
    safe_labels = label_map.clone()
    safe_labels[~valid_mask] = 0

    colors = color_lut[safe_labels.long()]  # (H, W, 3)
    colors = colors.permute(2, 0, 1)
    colors = colors * valid_mask.unsqueeze(0).float()
    return colors

