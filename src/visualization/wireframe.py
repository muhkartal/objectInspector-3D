"""
Wireframe visualization mode.
"""

import pygame
import numpy as np

from config import settings

from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.rendering.projector import Projector
from src.visualization.base import BaseVisualizer


class WireframeVisualizer(BaseVisualizer):
    """
    Renders mesh as wireframe (edges only).

    Shows the structural outline of the 3D object.
    """

    def __init__(self):
        super().__init__("wireframe")
        self.projector = None
        self.edge_color = settings.WIREFRAME_COLOR
        self.vertex_color = settings.POINT_COLOR
        self.show_vertices = False
        self.line_width = 1

    def update(
        self,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """Update projector if needed."""
        pass

    def draw(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """Draw wireframe edges."""
        if self.projector is None:
            self.projector = Projector(surface.get_width(), surface.get_height())

        # Project edges
        edges = self.projector.project_edges(mesh, camera)

        # Apply alpha for transitions
        color = self.edge_color
        if self.alpha < 1.0:
            color = tuple(int(c * self.alpha) for c in self.edge_color)

        # Draw all edges
        for start, end in edges:
            pygame.draw.line(
                surface,
                color,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                self.line_width,
            )

        # Optionally draw vertices
        if self.show_vertices:
            points = self.projector.project_points(mesh, camera)
            vertex_color = self.vertex_color
            if self.alpha < 1.0:
                vertex_color = tuple(int(c * self.alpha) for c in self.vertex_color)

            for point, depth in points:
                pygame.draw.circle(
                    surface,
                    vertex_color,
                    (int(point[0]), int(point[1])),
                    settings.POINT_SIZE // 2,
                )

    def on_resize(self, width: int, height: int) -> None:
        """Update projector size."""
        if self.projector:
            self.projector.resize(width, height)

    def set_color(self, color: tuple):
        """Set edge color."""
        self.edge_color = color

    def set_show_vertices(self, show: bool):
        """Toggle vertex display."""
        self.show_vertices = show

    def set_line_width(self, width: int):
        """Set line width."""
        self.line_width = max(1, width)
