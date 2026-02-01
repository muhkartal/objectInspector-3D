"""
Main pygame renderer with event handling.
"""

from typing import Callable, Optional, Dict, Any

import pygame
import numpy as np

from config import settings
from src.rendering.camera import Camera
from src.rendering.projector import Projector


class Renderer:
    """
    Main pygame rendering engine.

    Handles:
    - Window creation and management
    - Event processing
    - Drawing operations
    - FPS timing
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(settings.WINDOW_TITLE)

        # Create window
        self.width = settings.WINDOW_WIDTH
        self.height = settings.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE,
        )

        # Main drawing surface (for post-processing)
        self.surface = pygame.Surface((self.width, self.height))
        self.surface.set_colorkey((0, 0, 0))

        # Clock for FPS
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.frame_time = 0

        # Fonts
        self.font = pygame.font.SysFont(settings.FONT_NAME, settings.FONT_SIZE)
        self.font_large = pygame.font.SysFont(
            settings.FONT_NAME, settings.FONT_SIZE_LARGE
        )

        # Projector for 3D to 2D
        self.projector = Projector(self.width, self.height)

        # Input state
        self.mouse_pressed = {1: False, 2: False, 3: False}
        self.mouse_pos = (0, 0)
        self.mouse_delta = (0, 0)

        # Callbacks
        self.on_quit: Optional[Callable] = None
        self.on_key: Optional[Callable[[int, int], None]] = None
        self.on_resize: Optional[Callable[[int, int], None]] = None
        self.on_mouse_scroll: Optional[Callable[[float], None]] = None

    def handle_events(self, camera: Camera) -> bool:
        """
        Process pygame events.

        Args:
            camera: Camera to update based on input

        Returns:
            True if should continue, False if quit requested
        """
        self.mouse_delta = (0, 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.on_quit:
                    self.on_quit()
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.on_quit:
                        self.on_quit()
                    return False
                if self.on_key:
                    self.on_key(event.key, event.mod)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in self.mouse_pressed:
                    self.mouse_pressed[event.button] = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in self.mouse_pressed:
                    self.mouse_pressed[event.button] = False

            elif event.type == pygame.MOUSEMOTION:
                new_pos = event.pos
                self.mouse_delta = (
                    new_pos[0] - self.mouse_pos[0],
                    new_pos[1] - self.mouse_pos[1],
                )
                self.mouse_pos = new_pos

                # Orbit with left mouse button
                if self.mouse_pressed[1]:
                    camera.orbit(self.mouse_delta[0], self.mouse_delta[1])

                # Pan with middle or right mouse button
                if self.mouse_pressed[2] or self.mouse_pressed[3]:
                    camera.pan(self.mouse_delta[0], self.mouse_delta[1])

            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(event.y)
                if self.on_mouse_scroll:
                    self.on_mouse_scroll(event.y)

            elif event.type == pygame.VIDEORESIZE:
                self.resize(event.w, event.h)
                camera.set_aspect(event.w, event.h)
                if self.on_resize:
                    self.on_resize(event.w, event.h)

        return True

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (width, height),
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE,
        )
        self.surface = pygame.Surface((width, height))
        self.projector.resize(width, height)

    def clear(self, color: tuple = None):
        """Clear the screen with background color."""
        if color is None:
            color = settings.BACKGROUND_COLOR
        self.surface.fill(color)

    def draw_line(
        self,
        start: tuple,
        end: tuple,
        color: tuple,
        width: int = 1,
    ):
        """Draw a line."""
        # Convert numpy arrays to tuples if needed
        if hasattr(start, '__iter__') and not isinstance(start, tuple):
            start = (int(start[0]), int(start[1]))
        if hasattr(end, '__iter__') and not isinstance(end, tuple):
            end = (int(end[0]), int(end[1]))
        pygame.draw.line(self.surface, color, start, end, width)

    def draw_polygon(
        self,
        points: list,
        color: tuple,
        outline_color: tuple = None,
        outline_width: int = 1,
    ):
        """Draw a filled polygon with optional outline."""
        if len(points) < 3:
            return

        # Convert to integer coordinates
        int_points = [(int(p[0]), int(p[1])) for p in points]

        pygame.draw.polygon(self.surface, color, int_points)
        if outline_color:
            pygame.draw.polygon(self.surface, outline_color, int_points, outline_width)

    def draw_circle(
        self,
        center: tuple,
        radius: int,
        color: tuple,
        outline_color: tuple = None,
        outline_width: int = 1,
    ):
        """Draw a filled circle with optional outline."""
        int_center = (int(center[0]), int(center[1]))
        pygame.draw.circle(self.surface, color, int_center, radius)
        if outline_color:
            pygame.draw.circle(
                self.surface, outline_color, int_center, radius, outline_width
            )

    def draw_rect(
        self,
        rect: tuple,
        color: tuple,
        outline_color: tuple = None,
        outline_width: int = 1,
        border_radius: int = 0,
    ):
        """Draw a rectangle."""
        pygame.draw.rect(
            self.surface, color, rect, border_radius=border_radius
        )
        if outline_color:
            pygame.draw.rect(
                self.surface,
                outline_color,
                rect,
                outline_width,
                border_radius=border_radius,
            )

    def draw_text(
        self,
        text: str,
        position: tuple,
        color: tuple = None,
        large: bool = False,
        center: bool = False,
    ):
        """Draw text on screen."""
        if color is None:
            color = settings.FONT_COLOR

        font = self.font_large if large else self.font
        text_surface = font.render(text, True, color)

        if center:
            rect = text_surface.get_rect(center=position)
            self.surface.blit(text_surface, rect)
        else:
            self.surface.blit(text_surface, position)

    def draw_grid(self, camera: Camera):
        """Draw a reference grid on the XZ plane - optimized."""
        if not settings.SHOW_GRID:
            return

        # Use cached grid points
        if not hasattr(self, '_grid_points'):
            self._init_grid_cache()

        # Batch project all grid points
        screen_coords, _ = self.projector.project_vertices(self._grid_points, camera)

        # Draw grid lines using pre-computed indices
        n = len(self._grid_line_indices)
        for i in range(0, n, 2):
            idx1, idx2 = self._grid_line_indices[i], self._grid_line_indices[i + 1]
            color = self._grid_colors[i // 2]
            s1 = screen_coords[idx1]
            s2 = screen_coords[idx2]
            self.draw_line(s1, s2, color)

    def _init_grid_cache(self):
        """Initialize cached grid points and line indices."""
        size = settings.GRID_SIZE
        divisions = settings.GRID_DIVISIONS
        step = size / divisions

        points = []
        line_indices = []
        colors = []

        point_idx = 0
        for i in range(-divisions // 2, divisions // 2 + 1):
            # Lines parallel to X axis
            points.append([-size / 2, 0, i * step])
            points.append([size / 2, 0, i * step])
            line_indices.extend([point_idx, point_idx + 1])
            colors.append(settings.GRID_COLOR_AXIS if i == 0 else settings.GRID_COLOR)
            point_idx += 2

            # Lines parallel to Z axis
            points.append([i * step, 0, -size / 2])
            points.append([i * step, 0, size / 2])
            line_indices.extend([point_idx, point_idx + 1])
            colors.append(settings.GRID_COLOR_AXIS if i == 0 else settings.GRID_COLOR)
            point_idx += 2

        self._grid_points = np.array(points, dtype=np.float32)
        self._grid_line_indices = line_indices
        self._grid_colors = colors

    def draw_axes(self, camera: Camera, length: float = None):
        """Draw coordinate axes at origin - optimized."""
        if not settings.SHOW_AXES:
            return

        if length is None:
            length = settings.AXIS_LENGTH

        # Batch project all axis endpoints
        axis_points = np.array([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length],
        ], dtype=np.float32)

        screen_coords, _ = self.projector.project_vertices(axis_points, camera)
        origin = screen_coords[0]

        # Draw axes
        self.draw_line(origin, screen_coords[1], settings.AXIS_COLORS["x"], 2)
        self.draw_line(origin, screen_coords[2], settings.AXIS_COLORS["y"], 2)
        self.draw_line(origin, screen_coords[3], settings.AXIS_COLORS["z"], 2)

    def draw_fps(self):
        """Draw FPS counter."""
        if settings.SHOW_FPS:
            fps_text = f"FPS: {self.fps:.0f}"
            self.draw_text(fps_text, (10, 10), settings.FONT_COLOR_DIM)

    def draw_stats(self, stats: Dict[str, Any]):
        """Draw statistics overlay."""
        if not settings.SHOW_STATS:
            return

        y = 30
        for key, value in stats.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            self.draw_text(text, (10, y), settings.FONT_COLOR_DIM)
            y += 18

    def present(self):
        """Present the rendered frame to screen."""
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        # Update timing
        self.frame_time = self.clock.tick(settings.TARGET_FPS)
        self.fps = self.clock.get_fps()

    def get_surface(self) -> pygame.Surface:
        """Get the main drawing surface."""
        return self.surface

    def surface_to_array(self) -> np.ndarray:
        """Convert surface to numpy array (RGB)."""
        return pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)

    def array_to_surface(self, array: np.ndarray) -> pygame.Surface:
        """Convert numpy array to surface."""
        return pygame.surfarray.make_surface(array.transpose(1, 0, 2))

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
