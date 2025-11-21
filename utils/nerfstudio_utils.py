"""
Gaussian Viewer based on Viser 1.0.16 for 3D Gaussian Splatting.

This module provides a lightweight viewer that uses Viser to visualize
Gaussian primitives during training, with iteration history and multiple
visualization modes.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import OrderedDict

import numpy as np
import torch

from utils.general_utils import build_scaling_rotation

try:
    import viser
except ImportError as e:
    viser = None
    _VISER_IMPORT_ERROR = e
else:
    _VISER_IMPORT_ERROR = None


class GaussianNerfstudioViewer:
    """
    Viser 1.0.16-based viewer for monitoring Gaussian primitives as splats.

    Features:
        - Iteration slider to browse through training history
        - Mode buttons: depth, intensity, raydrop, semantic_mean
        - Real-time Gaussian splat visualization

    Args:
        host: Host address for the Viser server.
        port: Port number for the Viser server.
        max_points: Maximum number of Gaussian points to visualize.
        rendering_iterations: List of iteration numbers to save snapshots.
        log_filename: Optional path to a log file (unused).

    Attributes:
        max_points: Maximum number of Gaussians visualized.
        attribute_mode: Current attribute visualization mode.
        server: Underlying ViserServer instance.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7007,
        max_points: int = 200_000,
        rendering_iterations: Optional[List[int]] = None,
        log_filename: Optional[Path] = None,
    ) -> None:
        """
        Initialize the Viser server and internal state.

        Args:
            host: Host address for the viewer server.
            port: Port number for the viewer server.
            max_points: Maximum number of Gaussian points to visualize.
            rendering_iterations: List of iterations to save (e.g., [500, 1000, 2000]).
            log_filename: Optional path to a log file (currently unused).

        Raises:
            ImportError: If `viser` is not installed.
        """
        if viser is None:
            raise ImportError(
                "Viser is not available. Please install it via "
                "`pip install viser`."
            ) from _VISER_IMPORT_ERROR

        self.max_points: int = max(1, int(max_points))
        self.rendering_iterations = rendering_iterations or []

        # Visualization modes
        self._modes: List[str] = ["depth", "intensity", "semantic_mean"]
        self.attribute_mode: str = "depth"

        # Storage for Gaussian snapshots at each iteration
        # Format: {iteration: {"centers": ..., "covariances": ..., "attributes": {...}}}
        self._snapshots: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._iteration_list: List[int] = []  # Sorted list of saved iterations

        # Current display state
        self._current_iteration: int = 0
        self._display_centers: Optional[np.ndarray] = None
        self._display_covariances: Optional[np.ndarray] = None
        self._display_colors: Optional[np.ndarray] = None
        self._display_opacities: Optional[np.ndarray] = None
        self._suppress_slider_callback: bool = False

        # Viser server setup
        self.server: viser.ViserServer = viser.ViserServer(host=host, port=port)
        
        # Add Gaussian splat scene object
        self._splat_handle = self.server.scene.add_gaussian_splats(
            name="/gaussians",
            centers=np.zeros((0, 3), dtype=np.float32),
            covariances=np.zeros((0, 3, 3), dtype=np.float32),
            rgbs=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros((0, 1), dtype=np.float32),
        )

        # GUI: Folder for controls
        with self.server.gui.add_folder("Gaussian Viewer"):
            # Mode buttons
            self._mode_buttons = {}
            for mode in self._modes:
                btn = self.server.gui.add_button(f"Mode: {mode}")
                self._mode_buttons[mode] = btn
                
                # Closure to capture mode correctly
                def make_callback(m):
                    def callback(event):
                        self.set_attribute_mode(m)
                    return callback
                
                btn.on_click(make_callback(mode))
            
            # Iteration slider (ranges updated dynamically)
            self._iteration_slider = self.server.gui.add_slider(
                "Iteration",
                min=0,
                max=0,
                step=1,
                initial_value=0,
                disabled=True,
            )
            
            # Status text
            self._status_text = self.server.gui.add_text(
                "Status",
                initial_value="Waiting for Gaussians...",
            )

        # Register slider callback
        @self._iteration_slider.on_update
        def _on_iteration_update(event: viser.GuiEvent) -> None:
            """
            Callback when iteration slider is moved.
            
            Args:
                event: GUI event from Viser.
            """
            if self._suppress_slider_callback or not self._iteration_list:
                return
            
            slider_value = getattr(event, "value", None)
            if slider_value is None and hasattr(event, "target"):
                slider_value = getattr(event.target, "value", None)
            if slider_value is None:
                return

            slider_value = float(slider_value)
            iteration = min(self._iteration_list, key=lambda x: abs(x - slider_value))
            self._suppress_slider_callback = True
            self._iteration_slider.value = iteration
            self._suppress_slider_callback = False
            self._load_snapshot(iteration)

        print(f"[GaussianViewer] Viser server started on {host}:{port}")
        print(f"[GaussianViewer] Max points: {self.max_points}")
        print(f"[GaussianViewer] Modes: {self._modes}")
        print(f"[GaussianViewer] Rendering iterations: {self.rendering_iterations}")

    def close(self) -> None:
        """
        Shut down the viewer server and clean up resources.
        """
        if self.server is not None:
            try:
                self.server.stop()
                print("[GaussianViewer] Server shut down successfully.")
            except Exception as exc:
                print(f"[GaussianViewer] Error during shutdown: {exc}")

    def update_gaussians(
        self,
        iteration: int,
        gaussians: Any,
        attribute_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Update the viewer with Gaussian primitives at a specific iteration.

        This method:
          1. Checks if this iteration should be saved (in rendering_iterations).
          2. Extracts and samples Gaussian data.
          3. Stores snapshot for later browsing.
          4. Updates display if this is the latest iteration.

        Args:
            iteration: Current training iteration number.
            gaussians: GaussianModel with attributes:
                - get_xyz: [N, 3]
                - get_scaling: [N, 3]
                - get_rotation: [N, 4]
                - get_opacity: [N] or [N, 1]
                - _features_dc: [N, C, 3]
            attribute_data: Dict with keys "depth", "intensity", "raydrop", "semantic_mean"
                Each value is [N] or [N, 1] array.
        """
        # Check if we should save this iteration
        if iteration not in self.rendering_iterations:
            return

        # Extract Gaussian data
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        total = xyz.shape[0]
        if total == 0:
            print(f"[GaussianViewer] No Gaussians at iter {iteration}")
            return

        # Random sampling
        if total > self.max_points:
            sample_idx = np.random.choice(total, self.max_points, replace=False)
        else:
            sample_idx = np.arange(total)

        sampled_centers = xyz[sample_idx].astype(np.float32)

        # Extract Gaussian properties
        sample_idx_t = torch.from_numpy(sample_idx).long().to(gaussians.get_xyz.device)

        with torch.no_grad():
            # Covariance
            scaling = gaussians.get_scaling[sample_idx_t]
            rotation = gaussians.get_rotation[sample_idx_t]
            L = build_scaling_rotation(scaling, rotation)
            cov = torch.bmm(L, L.transpose(1, 2))
            sampled_covariances = cov.detach().cpu().numpy().astype(np.float32)

            # Opacity
            opacity = gaussians.get_opacity[sample_idx_t]
            opacity_np = opacity.detach().cpu().numpy()
            if opacity_np.ndim == 1:
                opacity_np = opacity_np[:, None]
            sampled_opacities = opacity_np.astype(np.float32)

            # SH DC term as base color
            sh_dc = getattr(gaussians, "_features_dc", None)
            if (
                torch.is_tensor(sh_dc)
                and sh_dc.shape[0] == gaussians.get_xyz.shape[0]
                and sh_dc.shape[1] >= 1
            ):
                base_rgb = torch.sigmoid(sh_dc[sample_idx_t, 0, :])
                sh_colors = base_rgb.detach().cpu().numpy()
            else:
                sh_colors = np.ones((sampled_centers.shape[0], 3), dtype=np.float32)

            sh_colors_uint8 = (sh_colors * 255.0).astype(np.uint8)

        # Extract and store attributes
        sampled_attributes = {}
        if attribute_data is not None:
            for key, value in attribute_data.items():
                arr = np.asarray(value).reshape(-1)
                if arr.shape[0] == total:
                    sampled_attributes[key] = arr[sample_idx].astype(np.float32)

        # Store snapshot
        self._snapshots[iteration] = {
            "centers": sampled_centers,
            "covariances": sampled_covariances,
            "opacities": sampled_opacities,
            "sh_colors": sh_colors_uint8,
            "attributes": sampled_attributes,
        }

        # Update iteration list
        if iteration not in self._iteration_list:
            self._iteration_list.append(iteration)
            self._iteration_list.sort()

            # Update slider range
            self._iteration_slider.min = float(self._iteration_list[0])
            self._iteration_slider.max = float(self._iteration_list[-1])
            self._iteration_slider.disabled = False

        # Update current iteration and display
        self._current_iteration = iteration
        self._suppress_slider_callback = True
        self._iteration_slider.value = float(iteration)
        self._suppress_slider_callback = False
        self._load_snapshot(iteration)

        print(f"[GaussianViewer] Snapshot saved for iteration {iteration} ({len(sample_idx)} points)")

    def _load_snapshot(self, iteration: int) -> None:
        """
        Load a specific iteration snapshot and update display.

        Args:
            iteration: Iteration number to load.
        """
        if iteration not in self._snapshots:
            return

        snapshot = self._snapshots[iteration]
        self._display_centers = snapshot["centers"]
        self._display_covariances = snapshot["covariances"]
        self._display_opacities = snapshot["opacities"]

        # Compute colors based on current mode
        self._display_colors = self._compute_colors(
            snapshot["sh_colors"],
            snapshot["attributes"],
            self.attribute_mode,
        )

        self._current_iteration = iteration
        self._update_scene()

        # Update status
        status = f"Iteration: {iteration} | Mode: {self.attribute_mode} | Points: {len(self._display_centers)}"
        self._status_text.value = status

    def _update_scene(self) -> None:
        """
        Update the Viser Gaussian splat with current display data.
        """
        if (
            self._display_centers is None
            or self._display_covariances is None
            or self._display_colors is None
            or self._display_opacities is None
        ):
            return

        centers = self._display_centers.astype(np.float32)
        covariances = self._display_covariances.astype(np.float32)
        # Viser expects RGB in [0, 1]
        rgbs = (self._display_colors.astype(np.float32) / 255.0).clip(0.0, 1.0)[:, :3]
        opacities = self._display_opacities.astype(np.float32)

        # Update splat handle properties
        self._splat_handle.centers = centers
        self._splat_handle.covariances = covariances
        self._splat_handle.rgbs = rgbs
        self._splat_handle.opacities = opacities

    def _compute_colors(
        self,
        sh_colors: np.ndarray,
        attributes: Dict[str, np.ndarray],
        mode: str,
    ) -> np.ndarray:
        """
        Compute RGB colors based on selected attribute mode.

        Args:
            sh_colors: Base SH colors [N, 3] uint8.
            attributes: Dict of attribute arrays.
            mode: One of "depth", "intensity", "raydrop", "semantic_mean".

        Returns:
            RGB colors [N, 3] uint8.
        """
        base = sh_colors.copy()

        # Get attribute values
        values = attributes.get(mode)
        if values is None:
            print(f"[GaussianViewer] Attribute '{mode}' not available, using SH colors")
            return base

        vals = np.nan_to_num(values)

        # Normalize to [0, 1]
        val_min = float(vals.min())
        val_max = float(vals.max())

        if val_max - val_min > 1e-6:
            vals = (vals - val_min) / (val_max - val_min)
        else:
            vals = np.zeros_like(vals)

        # Modulate brightness: 0.3 + 0.7 * normalized_value
        brightness = 0.3 + 0.7 * vals[:, None]  # [N, 1]
        colors = (base.astype(np.float32) * brightness).clip(0, 255)

        return colors.astype(np.uint8)

    def set_attribute_mode(self, mode: str) -> None:
        """
        Change the attribute visualization mode and refresh colors.

        Args:
            mode: One of "depth", "intensity", "raydrop", "semantic_mean".
        """
        if mode not in self._modes:
            print(f"[GaussianViewer] Invalid mode '{mode}'. Valid: {self._modes}")
            return

        self.attribute_mode = mode
        print(f"[GaussianViewer] Switched to mode: {mode}")

        # Reload current snapshot with new mode
        if self._current_iteration in self._snapshots:
            self._load_snapshot(self._current_iteration)


def create_viewer_from_args(args: Any) -> Optional[GaussianNerfstudioViewer]:
    """
    Factory function to create a Gaussian viewer from argument namespace.

    Expected args attributes:
      - use_viewer: bool, whether to enable visualization.
      - viewer_host: str, host address (default: "0.0.0.0").
      - viewer_port: int, port number (default: 7007).
      - viewer_max_points: int, max points (default: 200_000).
      - rendering_iterations: list of ints, iterations to save.

    Args:
        args: Argument or config object.

    Returns:
        GaussianNerfstudioViewer instance if enabled, else None.
    """
    if not getattr(args, "use_viewer", False):
        return None

    if viser is None:
        print("[GaussianViewer] `viser` not installed, skipping visualization.")
        return None

    try:
        rendering_iterations = getattr(args, "rendering_iterations", [])
        if not rendering_iterations:
            print("[GaussianViewer] Warning: No rendering_iterations specified, viewer will be empty.")

        viewer = GaussianNerfstudioViewer(
            host=getattr(args, "viewer_host", "0.0.0.0"),
            port=int(getattr(args, "viewer_port", 7007)),
            max_points=int(getattr(args, "viewer_max_points", 200_000)),
            rendering_iterations=rendering_iterations,
        )
        return viewer
    except Exception as exc:
        print(f"[GaussianViewer] Failed to initialize viewer: {exc}")
        import traceback
        traceback.print_exc()
        return None