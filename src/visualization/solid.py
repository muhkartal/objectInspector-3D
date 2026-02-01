"""
Solid shaded visualization mode.
"""

import pygame
import numpy as np

from config import settings
from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.rendering.projector import Projector
from src.rendering.lighting import Lighting, get_lighting
from src.visualization.base import BaseVisualizer


class SolidVisualizer(BaseVisualizer):
    """
    Renders mesh with solid shaded faces.

    Features:
    - Filled polygons with depth sorting
    - Simple lighting (ambient + diffuse)
    - Optional outline
    """

    def __init__(self):
        super().__init__("solid")
        self.projector = None
        self.lighting = get_lighting()
        self.base_color = settings.SHAPE_COLOR
        self.show_outline = True
        self.outline_color = (40, 40, 45)
        self.outline_width = 1

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
        """Draw solid shaded faces (Optimized)."""
        if self.projector is None:
            self.projector = Projector(surface.get_width(), surface.get_height())

        # Use fast projection (returns numpy arrays)
        face_verts_2d, _, face_normals, face_indices = self.projector.project_mesh_fast(
            mesh, camera, cull_backfaces=True
        )

        if len(face_verts_2d) == 0:
            return

        # Calculate centers for lighting (vectorized)
        visible_faces = mesh.faces[face_indices]
        v0 = mesh.vertices[visible_faces[:, 0]]
        v1 = mesh.vertices[visible_faces[:, 1]]
        v2 = mesh.vertices[visible_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0

        # Calculate view directions
        view_dirs = camera.position - face_centers
        lengths = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs = view_dirs / np.maximum(lengths, 1e-8)

        # Prepare base colors
        if mesh.colors is not None:
            # Use mesh face colors (average of vertex colors)
            c0 = mesh.colors[visible_faces[:, 0]]
            c1 = mesh.colors[visible_faces[:, 1]]
            c2 = mesh.colors[visible_faces[:, 2]]
            base_colors = (c0.astype(np.float32) + c1 + c2) / 3.0
        else:
            base_colors = np.full((len(face_indices), 3), self.base_color, dtype=np.float32)

        # Batch lighting calculation
        shaded_colors = self.lighting.shade_colors_batch(
            base_colors, face_normals, view_dirs
        )

        # Apply global alpha if needed
        if self.alpha < 1.0:
            shaded_colors = shaded_colors * self.alpha
            outline_color_rgb = tuple(int(c * self.alpha) for c in self.outline_color)
        else:
            outline_color_rgb = self.outline_color

        # Convert to list of points for pygame
        # face_verts_2d is (N, 3, 2) float
        points_list = face_verts_2d.astype(np.int32)

        # Draw loop (still needed for pygame, but simplified)
        for i in range(len(face_indices)):
            points = points_list[i]
            color = shaded_colors[i]

            try:
                # Use points directly (Nx2 array)
                pygame.draw.polygon(surface, color, points)

                # Draw outline
                if self.show_outline:
                    pygame.draw.polygon(
                        surface, outline_color_rgb, points, self.outline_width
                    )
            except (ValueError, TypeError):
                pass

    def on_resize(self, width: int, height: int) -> None:
        """Update projector size."""
        if self.projector:
            self.projector.resize(width, height)

    def set_color(self, color: tuple):
        """Set base face color."""
        self.base_color = color

    def set_show_outline(self, show: bool):
        """Toggle outline display."""
        self.show_outline = show

    def set_outline_color(self, color: tuple):
        """Set outline color."""
        self.outline_color = color
