"""
Exploded view visualization mode for multi-part assemblies.
Tesla-style engineering drawing visualization.
"""

from typing import Dict, Any, Optional, List, Tuple
import time

import pygame
import numpy as np

from config import settings
from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.rendering.projector import Projector
from src.rendering.lighting import get_lighting
from src.visualization.base import BaseVisualizer


class ExplodedViewVisualizer(BaseVisualizer):
    """
    Renders multi-part assemblies with exploded view support.

    Features:
    - Renders all assembly parts with proper depth sorting
    - Smooth explosion animation with easing
    - Integrates with existing lighting system
    - Supports label rendering via LabelRenderer
    """

    def __init__(self):
        super().__init__("exploded")
        self.projector: Optional[Projector] = None
        self.lighting = get_lighting()
        self.assembly = None
        self.show_outline = True
        self.outline_color = (40, 40, 45)
        self.outline_width = 1
        self.show_labels = True

        # Label renderer (set externally)
        self.label_renderer = None

        # Animation state
        self.last_update_time = time.time()

        # Cached depth-sorted parts
        self._cached_parts: List[Tuple[str, Mesh, Tuple[int, int, int], float]] = []

    def set_assembly(self, assembly) -> None:
        """Set the assembly to visualize."""
        self.assembly = assembly
        self._cached_parts = []

    def set_label_renderer(self, renderer) -> None:
        """Set the label renderer for callout labels."""
        self.label_renderer = renderer

    def update(
        self,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """Update animation state."""
        if self.assembly is None:
            return

        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update assembly animation
        self.assembly.update(dt)

        # Update cached parts with depth sorting
        self._update_cached_parts(camera)

    def _update_cached_parts(self, camera: Camera) -> None:
        """Update and depth-sort cached parts."""
        if self.assembly is None:
            return

        parts_with_depth = []

        for name, mesh, color in self.assembly.get_all_meshes():
            # Calculate depth (distance from camera to mesh center)
            mesh_center = mesh.center
            depth = np.linalg.norm(camera.position - mesh_center)
            parts_with_depth.append((name, mesh, color, depth))

        # Sort by depth (far to near for proper painter's algorithm)
        parts_with_depth.sort(key=lambda x: x[3], reverse=True)
        self._cached_parts = parts_with_depth

    def draw(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """Draw exploded assembly view."""
        if self.assembly is None:
            return

        if self.projector is None:
            self.projector = Projector(surface.get_width(), surface.get_height())

        # Draw each part (already sorted by depth)
        for name, part_mesh, color, _ in self._cached_parts:
            self._draw_part(surface, part_mesh, camera, color)

        # Draw labels if enabled and renderer available
        if self.show_labels and self.label_renderer is not None:
            self._draw_labels(surface, camera)

    def _draw_part(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a single part with shading."""
        # Project mesh
        face_verts_2d, _, face_normals, face_indices = self.projector.project_mesh_fast(
            mesh, camera, cull_backfaces=True
        )

        if len(face_verts_2d) == 0:
            return

        # Calculate lighting
        visible_faces = mesh.faces[face_indices]
        v0 = mesh.vertices[visible_faces[:, 0]]
        v1 = mesh.vertices[visible_faces[:, 1]]
        v2 = mesh.vertices[visible_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0

        # View directions
        view_dirs = camera.position - face_centers
        lengths = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs = view_dirs / np.maximum(lengths, 1e-8)

        # Base colors (use part color)
        base_colors = np.full((len(face_indices), 3), color, dtype=np.float32)

        # Apply lighting
        shaded_colors = self.lighting.shade_colors_batch(
            base_colors, face_normals, view_dirs
        )

        # Apply alpha
        if self.alpha < 1.0:
            shaded_colors = shaded_colors * self.alpha
            outline_color_rgb = tuple(int(c * self.alpha) for c in self.outline_color)
        else:
            outline_color_rgb = self.outline_color

        # Draw faces
        points_list = face_verts_2d.astype(np.int32)

        for i in range(len(face_indices)):
            points = points_list[i]
            face_color = shaded_colors[i]

            try:
                pygame.draw.polygon(surface, face_color, points)

                if self.show_outline:
                    pygame.draw.polygon(
                        surface, outline_color_rgb, points, self.outline_width
                    )
            except (ValueError, TypeError):
                pass

    def _draw_labels(self, surface: pygame.Surface, camera: Camera) -> None:
        """Draw labels for all parts using the label renderer."""
        if self.label_renderer is None or self.assembly is None:
            return

        # Get all label positions
        label_data = []
        for name, position in self.assembly.get_all_label_positions():
            # Project 3D position to 2D
            screen_pos = self._project_point(position, camera, surface)
            if screen_pos is not None:
                label_data.append((name, screen_pos, position))

        # Draw labels
        self.label_renderer.draw_labels(surface, label_data)

    def _project_point(
        self,
        point: np.ndarray,
        camera: Camera,
        surface: pygame.Surface
    ) -> Optional[Tuple[int, int]]:
        """Project a 3D point to screen coordinates."""
        if self.projector is None:
            return None

        # Use projector's vertex projection
        points = np.array([point], dtype=np.float32)
        screen_coords, depths = self.projector.project_vertices(points, camera)

        if len(screen_coords) == 0 or depths[0] < 0:
            return None

        x, y = screen_coords[0]
        width, height = surface.get_size()

        # Check if on screen
        if 0 <= x < width and 0 <= y < height:
            return (int(x), int(y))

        return None

    def on_resize(self, width: int, height: int) -> None:
        """Update projector size."""
        if self.projector:
            self.projector.resize(width, height)

    def set_show_outline(self, show: bool) -> None:
        """Toggle outline display."""
        self.show_outline = show

    def set_show_labels(self, show: bool) -> None:
        """Toggle label display."""
        self.show_labels = show

    def get_explosion_factor(self) -> float:
        """Get current explosion factor."""
        if self.assembly is None:
            return 0.0
        return self.assembly.explosion_factor

    def set_explosion_factor(self, factor: float, animate: bool = False) -> None:
        """Set explosion factor."""
        if self.assembly is not None:
            self.assembly.set_explosion(factor, animate=animate)
