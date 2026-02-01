"""
ML results overlay visualization.
"""

from typing import Dict, Any, Optional, List, Tuple

import pygame
import numpy as np

from config import settings
from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.visualization.base import BaseVisualizer


class MLOverlayVisualizer(BaseVisualizer):
    """
    Overlays ML inference results on the visualization.

    Displays:
    - Classification labels with confidence
    - Bounding boxes
    - Detected features/keypoints
    - Depth colormap
    - Pose estimation axes
    """

    def __init__(self):
        super().__init__("ml_overlay")

        # Colors for different elements
        self.bbox_color = (0, 255, 128)
        self.label_color = (255, 255, 255)
        self.label_bg_color = (0, 128, 64)
        self.keypoint_color = (255, 200, 0)
        self.pose_colors = {
            "x": (255, 100, 100),
            "y": (100, 255, 100),
            "z": (100, 100, 255),
        }

        # Fonts
        self.font = None
        self.font_large = None

        # Depth colormap
        self.depth_surface = None

    def _ensure_fonts(self):
        """Initialize fonts if needed."""
        if self.font is None:
            self.font = pygame.font.SysFont(settings.FONT_NAME, settings.FONT_SIZE)
            self.font_large = pygame.font.SysFont(
                settings.FONT_NAME, settings.FONT_SIZE_LARGE
            )

    def update(
        self,
        mesh: Mesh,
        camera: Camera,
        ml_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update overlay state."""
        pass

    def draw(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
        ml_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Draw ML results overlay."""
        self._ensure_fonts()

        if ml_results is None:
            self._draw_no_results(surface)
            return

        # Draw each type of result
        if "classification" in ml_results:
            self._draw_classification(surface, ml_results["classification"])

        if "bboxes" in ml_results:
            self._draw_bboxes(surface, ml_results["bboxes"])

        if "keypoints" in ml_results:
            self._draw_keypoints(surface, ml_results["keypoints"])

        if "depth" in ml_results:
            self._draw_depth_overlay(surface, ml_results["depth"])

        if "pose" in ml_results:
            self._draw_pose(surface, ml_results["pose"], camera)

        if "features" in ml_results:
            self._draw_features(surface, ml_results["features"])

    def _draw_no_results(self, surface: pygame.Surface):
        """Draw placeholder when no ML results available."""
        text = "ML: No results"
        text_surface = self.font.render(text, True, settings.FONT_COLOR_DIM)
        surface.blit(text_surface, (surface.get_width() - 120, 10))

    def _draw_classification(
        self, surface: pygame.Surface, classification: Dict[str, Any]
    ):
        """Draw classification label and confidence."""
        label = classification.get("label", "Unknown")
        confidence = classification.get("confidence", 0.0)

        # Format text
        text = f"{label}: {confidence:.1%}"

        # Draw background
        text_surface = self.font_large.render(text, True, self.label_color)
        text_rect = text_surface.get_rect()
        text_rect.topright = (surface.get_width() - 10, 10)

        bg_rect = text_rect.inflate(16, 8)
        pygame.draw.rect(surface, self.label_bg_color, bg_rect, border_radius=4)

        # Draw text
        surface.blit(text_surface, text_rect)

    def _draw_bboxes(
        self, surface: pygame.Surface, bboxes: List[Dict[str, Any]]
    ):
        """Draw bounding boxes."""
        for bbox in bboxes:
            x, y, w, h = bbox.get("rect", (0, 0, 0, 0))
            label = bbox.get("label", "")
            confidence = bbox.get("confidence", 0.0)

            # Draw box
            rect = pygame.Rect(x, y, w, h)
            pygame.draw.rect(surface, self.bbox_color, rect, 2)

            # Draw label
            if label:
                label_text = f"{label}: {confidence:.0%}"
                text_surface = self.font.render(label_text, True, self.label_color)

                # Background for label
                text_rect = text_surface.get_rect()
                text_rect.bottomleft = (x, y - 2)
                bg_rect = text_rect.inflate(8, 4)
                pygame.draw.rect(surface, self.label_bg_color, bg_rect)

                surface.blit(text_surface, text_rect)

    def _draw_keypoints(
        self, surface: pygame.Surface, keypoints: List[Tuple[int, int]]
    ):
        """Draw detected keypoints."""
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            pygame.draw.circle(surface, self.keypoint_color, (x, y), 4)
            pygame.draw.circle(surface, (255, 255, 255), (x, y), 4, 1)

    def _draw_depth_overlay(
        self, surface: pygame.Surface, depth_map: np.ndarray
    ):
        """Draw depth colormap overlay."""
        if depth_map is None or depth_map.size == 0:
            return

        # Normalize depth to 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_map)

        # Apply colormap (blue = near, red = far)
        h, w = depth_map.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:, :, 0] = (normalized * 255).astype(np.uint8)  # Red channel
        colored[:, :, 2] = ((1 - normalized) * 255).astype(np.uint8)  # Blue channel

        # Create surface and scale
        depth_surface = pygame.surfarray.make_surface(colored.transpose(1, 0, 2))
        depth_surface = pygame.transform.scale(
            depth_surface, (surface.get_width() // 4, surface.get_height() // 4)
        )
        depth_surface.set_alpha(180)

        # Draw in corner
        surface.blit(depth_surface, (10, surface.get_height() - depth_surface.get_height() - 10))

        # Label
        label = self.font.render("Depth", True, self.label_color)
        surface.blit(label, (10, surface.get_height() - depth_surface.get_height() - 28))

    def _draw_pose(
        self, surface: pygame.Surface, pose: Dict[str, Any], camera: Camera
    ):
        """Draw pose estimation axes."""
        center = pose.get("center", (surface.get_width() // 2, surface.get_height() // 2))
        rotation = pose.get("rotation", np.eye(3))
        scale = pose.get("scale", 50)

        cx, cy = int(center[0]), int(center[1])

        # Draw axes
        for i, (axis, color) in enumerate(self.pose_colors.items()):
            # Get axis direction from rotation matrix
            direction = rotation[:, i] * scale
            end_x = int(cx + direction[0])
            end_y = int(cy - direction[1])  # Flip Y for screen coords

            pygame.draw.line(surface, color, (cx, cy), (end_x, end_y), 3)

            # Draw axis label
            label = self.font.render(axis.upper(), True, color)
            surface.blit(label, (end_x + 5, end_y - 8))

    def _draw_features(
        self, surface: pygame.Surface, features: Dict[str, Any]
    ):
        """Draw detected features (edges, corners, etc.)."""
        # Draw edges
        if "edges" in features:
            for edge in features["edges"]:
                start = (int(edge[0][0]), int(edge[0][1]))
                end = (int(edge[1][0]), int(edge[1][1]))
                pygame.draw.line(surface, (0, 255, 255), start, end, 1)

        # Draw corners
        if "corners" in features:
            for corner in features["corners"]:
                x, y = int(corner[0]), int(corner[1])
                pygame.draw.circle(surface, (255, 0, 255), (x, y), 3)

    def on_resize(self, width: int, height: int) -> None:
        """Handle resize."""
        self.depth_surface = None  # Recreate on next draw
