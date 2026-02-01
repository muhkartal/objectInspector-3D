"""
Point cloud visualization mode.
"""

import pygame
import numpy as np

from config import settings

from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.rendering.projector import Projector
from src.visualization.base import BaseVisualizer


class PointsVisualizer(BaseVisualizer):
    """
    Renders mesh vertices as points (point cloud style).

    Shows only the vertices without edges or faces.
    """

    def __init__(self):
        super().__init__("points")
        self.projector = None
        self.point_color = settings.POINT_COLOR
        self.point_size = settings.POINT_SIZE
        self.depth_shading = True  # Fade points by distance

    def update(
        self,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """Update state."""
        pass

    def draw(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """Draw vertices as points."""
        if self.projector is None:
            self.projector = Projector(surface.get_width(), surface.get_height())

        # Project all points
        points = self.projector.project_points(mesh, camera)

        if not points:
            return

        # Sort by depth (far to near) for proper ordering
        points.sort(key=lambda p: -p[1])

        # Find depth range for shading
        if self.depth_shading and len(points) > 1:
            depths = [p[1] for p in points]
            min_depth = min(depths)
            max_depth = max(depths)
            depth_range = max_depth - min_depth
            if depth_range < 0.001:
                depth_range = 1.0
        else:
            min_depth = 0
            depth_range = 1.0

        # Draw each point
        for point, depth in points:
            # Calculate depth-based color intensity
            if self.depth_shading:
                # Normalize depth to 0-1 (closer = brighter)
                t = 1.0 - (depth - min_depth) / depth_range
                t = max(0.3, min(1.0, t))  # Clamp for visibility
            else:
                t = 1.0

            # Apply depth shading and alpha
            color = tuple(
                int(c * t * self.alpha) for c in self.point_color
            )

            # Size can also vary with depth
            size = self.point_size
            if self.depth_shading:
                size = max(2, int(self.point_size * (0.5 + t * 0.5)))

            pygame.draw.circle(
                surface,
                color,
                (int(point[0]), int(point[1])),
                size,
            )

    def on_resize(self, width: int, height: int) -> None:
        """Update projector size."""
        if self.projector:
            self.projector.resize(width, height)

    def set_color(self, color: tuple):
        """Set point color."""
        self.point_color = color

    def set_point_size(self, size: int):
        """Set point size."""
        self.point_size = max(1, size)

    def set_depth_shading(self, enabled: bool):
        """Toggle depth-based shading."""
        self.depth_shading = enabled
