"""
Visualizer manager with transitions between modes.
"""

from typing import Dict, Optional, List

import pygame
import time

from config import settings
from src.geometry.mesh import Mesh
from src.rendering.camera import Camera
from src.visualization.base import BaseVisualizer
from src.visualization.wireframe import WireframeVisualizer
from src.visualization.solid import SolidVisualizer
from src.visualization.points import PointsVisualizer
from src.visualization.exploded import ExplodedViewVisualizer


class VisualizerManager:
    """
    Manages multiple visualizers with smooth transitions.

    Features:
    - Switch between visualization modes
    - Smooth crossfade transitions
    - Mode indicator display
    """

    def __init__(self):
        # Create visualizers
        self.visualizers: Dict[str, BaseVisualizer] = {
            "wireframe": WireframeVisualizer(),
            "solid": SolidVisualizer(),
            "points": PointsVisualizer(),
            "exploded": ExplodedViewVisualizer(),
        }

        # Current and transition state
        self.current_mode = settings.DEFAULT_VISUALIZATION_MODE
        self.previous_mode: Optional[str] = None
        self.transition_start: float = 0
        self.transition_duration = settings.TRANSITION_DURATION
        self.transitioning = False

        # Mode order for cycling
        self.mode_order = ["solid", "wireframe", "points", "exploded"]

        # Current assembly for exploded view
        self.current_assembly = None

        # Activate initial mode
        if self.current_mode in self.visualizers:
            self.visualizers[self.current_mode].on_activate()

    def get_current_visualizer(self) -> Optional[BaseVisualizer]:
        """Get the currently active visualizer."""
        return self.visualizers.get(self.current_mode)

    def set_mode(self, mode: str):
        """
        Switch to a visualization mode.

        Args:
            mode: Mode name (wireframe, solid, points)
        """
        if mode not in self.visualizers:
            print(f"Unknown visualization mode: {mode}")
            return

        if mode == self.current_mode:
            return

        # Start transition
        self.previous_mode = self.current_mode
        self.current_mode = mode
        self.transition_start = time.time()
        self.transitioning = True

        # Notify visualizers
        if self.previous_mode in self.visualizers:
            self.visualizers[self.previous_mode].on_deactivate()
        self.visualizers[self.current_mode].on_activate()

    def cycle_mode(self):
        """Cycle to next visualization mode."""
        try:
            current_idx = self.mode_order.index(self.current_mode)
            next_idx = (current_idx + 1) % len(self.mode_order)
            self.set_mode(self.mode_order[next_idx])
        except ValueError:
            self.set_mode(self.mode_order[0])

    def update(
        self,
        mesh: Mesh,
        camera: Camera,
    ):
        """Update all active visualizers."""
        # Update transition
        if self.transitioning:
            elapsed = time.time() - self.transition_start
            if elapsed >= self.transition_duration:
                self.transitioning = False
                self.previous_mode = None

        # Update current visualizer
        current = self.get_current_visualizer()
        if current:
            current.update(mesh, camera)

        # Update previous during transition
        if self.transitioning and self.previous_mode:
            prev = self.visualizers.get(self.previous_mode)
            if prev:
                prev.update(mesh, camera)

    def draw(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        camera: Camera,
    ):
        """Draw visualization with transitions."""
        # Calculate transition alpha
        if self.transitioning:
            elapsed = time.time() - self.transition_start
            t = min(1.0, elapsed / self.transition_duration)
            # Smooth easing
            t = t * t * (3 - 2 * t)  # Smoothstep
        else:
            t = 1.0

        # Draw previous visualizer (fading out)
        if self.transitioning and self.previous_mode:
            prev = self.visualizers.get(self.previous_mode)
            if prev:
                prev.set_alpha(1.0 - t)
                prev.draw(surface, mesh, camera)

        # Draw current visualizer
        current = self.get_current_visualizer()
        if current:
            current.set_alpha(t if self.transitioning else 1.0)
            current.draw(surface, mesh, camera)

    def draw_mode_indicator(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
    ):
        """Draw mode indicator dots."""
        dot_radius = 4
        dot_spacing = 14
        active_color = (255, 255, 255)
        inactive_color = (80, 80, 80)

        for i, mode in enumerate(self.mode_order):
            cx = x + i * dot_spacing
            color = active_color if mode == self.current_mode else inactive_color
            pygame.draw.circle(surface, color, (cx, y), dot_radius)

    def on_resize(self, width: int, height: int):
        """Handle window resize for all visualizers."""
        for viz in self.visualizers.values():
            viz.on_resize(width, height)

    def get_mode_names(self) -> List[str]:
        """Get list of available mode names."""
        return list(self.mode_order)

    def set_visualizer_color(self, color: tuple):
        """Set color for all visualizers that support it."""
        for viz in self.visualizers.values():
            if hasattr(viz, "set_color"):
                viz.set_color(color)

    def set_assembly(self, assembly) -> None:
        """Set the assembly for exploded view mode."""
        self.current_assembly = assembly
        exploded_viz = self.visualizers.get("exploded")
        if exploded_viz and hasattr(exploded_viz, "set_assembly"):
            exploded_viz.set_assembly(assembly)

    def get_exploded_visualizer(self):
        """Get the exploded view visualizer."""
        return self.visualizers.get("exploded")

    def set_explosion_factor(self, factor: float, animate: bool = False) -> None:
        """Set the explosion factor for the exploded view."""
        exploded_viz = self.get_exploded_visualizer()
        if exploded_viz and hasattr(exploded_viz, "set_explosion_factor"):
            exploded_viz.set_explosion_factor(factor, animate=animate)

    def get_explosion_factor(self) -> float:
        """Get the current explosion factor."""
        exploded_viz = self.get_exploded_visualizer()
        if exploded_viz and hasattr(exploded_viz, "get_explosion_factor"):
            return exploded_viz.get_explosion_factor()
        return 0.0
