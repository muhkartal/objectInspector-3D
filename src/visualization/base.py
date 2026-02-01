"""
Abstract base class for visualizers.
"""

from abc import ABC, abstractmethod

import pygame



from src.geometry.mesh import Mesh
from src.rendering.camera import Camera


class BaseVisualizer(ABC):
    """
    Abstract base class for visualization modes.

    All visualizers must implement:
    - update(): Update internal state
    - draw(): Render to surface
    """

    def __init__(self, name: str):
        self.name = name
        self.active = False
        self.alpha = 1.0  # For transitions

    @abstractmethod
    def update(
        self,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """
        Update visualizer state.

        Args:
            mesh: Current mesh to visualize
            camera: Camera for view calculations
        """
        pass

    @abstractmethod
    def draw(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
    ) -> None:
        """
        Render visualization to surface.

        Args:
            surface: Pygame surface to draw on
            mesh: Mesh to visualize
            camera: Camera for projection
        """
        pass

    def on_resize(self, width: int, height: int) -> None:
        """Handle window resize."""
        pass

    def on_activate(self) -> None:
        """Called when this visualizer becomes active."""
        self.active = True

    def on_deactivate(self) -> None:
        """Called when this visualizer is no longer active."""
        self.active = False

    def set_alpha(self, alpha: float) -> None:
        """Set transparency for transitions."""
        self.alpha = max(0.0, min(1.0, alpha))
