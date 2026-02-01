"""
Synthetic input source - renders 3D shapes to create ML input.
"""

from typing import Optional

import threading
import time

import pygame
import numpy as np

from config import settings

from src.input.base import InputSource
from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.rendering.projector import Projector
from src.rendering.lighting import get_lighting


class SyntheticInput(InputSource):
    """
    Generates synthetic frames by rendering 3D shapes.

    Useful for testing ML models without a camera.
    """

    def __init__(self, width: int = None, height: int = None):
        super().__init__("synthetic")

        self.width = width or settings.ML_PROCESS_WIDTH
        self.height = height or settings.ML_PROCESS_HEIGHT

        # Offscreen surface for rendering
        self.surface: Optional[pygame.Surface] = None
        self.projector: Optional[Projector] = None
        self.lighting = get_lighting()

        # Current mesh and camera to render
        self.mesh: Optional[Mesh] = None
        self.camera: Optional[Camera] = None

        # Background color
        self.bg_color = settings.BACKGROUND_COLOR

    def start(self) -> bool:
        """Initialize synthetic input."""
        try:
            # Ensure pygame is initialized
            if not pygame.get_init():
                pygame.init()

            self.surface = pygame.Surface((self.width, self.height))
            self.projector = Projector(self.width, self.height)
            self.running = True

            print(f"Synthetic input started: {self.width}x{self.height}")
            return True

        except Exception as e:
            print(f"Error starting synthetic input: {e}")
            return False

    def stop(self) -> None:
        """Stop synthetic input."""
        self.running = False
        self.surface = None
        print("Synthetic input stopped")

    def set_scene(self, mesh: Mesh, camera: Camera):
        """
        Set the scene to render.

        Args:
            mesh: Mesh to render
            camera: Camera for view/projection
        """
        self.mesh = mesh
        self.camera = camera

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Render and return the current frame.

        Returns:
            RGB numpy array or None
        """
        if not self.running or self.surface is None:
            return None

        if self.mesh is None or self.camera is None:
            # Return blank frame
            return np.full(
                (self.height, self.width, 3),
                self.bg_color,
                dtype=np.uint8,
            )

        # Clear surface
        self.surface.fill(self.bg_color)

        # Project and render mesh
        self._render_mesh()

        # Convert surface to numpy array
        frame = pygame.surfarray.array3d(self.surface)
        # Transpose from (W, H, 3) to (H, W, 3)
        frame = frame.transpose(1, 0, 2)

        return frame

    def _render_mesh(self):
        """Render the current mesh to the surface."""
        if self.mesh is None or self.camera is None or self.projector is None:
            return

        # Project faces
        projected_faces = self.projector.project_mesh(
            self.mesh, self.camera, cull_backfaces=True, sort_faces=True
        )

        camera_pos = self.camera.position
        base_color = settings.SHAPE_COLOR

        # Draw each face
        for face in projected_faces:
            points = [(int(v[0]), int(v[1])) for v in face.vertices_2d]

            if len(points) < 3:
                continue

            # Calculate lighting
            face_center = face.vertices_3d.mean(axis=0)
            view_dir = camera_pos - face_center
            view_dir = view_dir / np.linalg.norm(view_dir)

            color = face.color if face.color else base_color
            shaded_color = self.lighting.shade_color(color, face.normal, view_dir)

            try:
                pygame.draw.polygon(self.surface, shaded_color, points)
            except (ValueError, TypeError):
                pass

    def set_background(self, color: tuple):
        """Set background color."""
        self.bg_color = color

    def set_resolution(self, width: int, height: int):
        """Change render resolution."""
        self.width = width
        self.height = height

        if self.running:
            self.surface = pygame.Surface((width, height))
            self.projector = Projector(width, height)
