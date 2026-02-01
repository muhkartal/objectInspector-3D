"""
Interactive slider widget for explosion control.
Tesla-style minimal design.
"""

from typing import Optional, Callable, Tuple

import pygame

from config import settings


class Slider:
    """
    Interactive slider widget with modern dark theme.

    Features:
    - Dark track with accent color fill
    - White circular handle
    - Mouse drag interaction
    - Optional value display
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int = None,
        min_value: float = 0.0,
        max_value: float = 1.0,
        initial_value: float = 0.0,
        label: str = "",
    ):
        """
        Initialize slider widget.

        Args:
            x: X position of slider left edge
            y: Y position of slider center
            width: Width of slider track (uses settings default if None)
            min_value: Minimum slider value
            max_value: Maximum slider value
            initial_value: Initial value
            label: Optional label text
        """
        self.x = x
        self.y = y
        self.width = width if width is not None else settings.SLIDER_WIDTH
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.label = label

        # Styling from settings
        self.track_color = settings.SLIDER_TRACK_COLOR
        self.fill_color = settings.SLIDER_FILL_COLOR
        self.handle_color = settings.SLIDER_HANDLE_COLOR
        self.handle_radius = settings.SLIDER_HANDLE_RADIUS
        self.track_height = settings.SLIDER_TRACK_HEIGHT

        # State
        self.dragging = False
        self.hovered = False

        # Callback
        self.on_change: Optional[Callable[[float], None]] = None

        # Font (initialized lazily)
        self._font: Optional[pygame.font.Font] = None

    def _ensure_font(self) -> None:
        """Initialize font if needed."""
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 12)

    @property
    def rect(self) -> pygame.Rect:
        """Get the full interactive area of the slider."""
        return pygame.Rect(
            self.x - self.handle_radius,
            self.y - self.handle_radius,
            self.width + self.handle_radius * 2,
            self.handle_radius * 2
        )

    @property
    def track_rect(self) -> pygame.Rect:
        """Get the track rectangle."""
        return pygame.Rect(
            self.x,
            self.y - self.track_height // 2,
            self.width,
            self.track_height
        )

    def _value_to_x(self, value: float) -> int:
        """Convert value to x position."""
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        normalized = max(0.0, min(1.0, normalized))
        return int(self.x + normalized * self.width)

    def _x_to_value(self, x: int) -> float:
        """Convert x position to value."""
        normalized = (x - self.x) / self.width
        normalized = max(0.0, min(1.0, normalized))
        return self.min_value + normalized * (self.max_value - self.min_value)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.

        Args:
            event: Pygame event

        Returns:
            True if event was consumed by this widget
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if self.rect.collidepoint(event.pos):
                    self.dragging = True
                    self._update_from_mouse(event.pos[0])
                    return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.dragging:
                self.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            self.hovered = self.rect.collidepoint(event.pos)

            if self.dragging:
                self._update_from_mouse(event.pos[0])
                return True

        return False

    def _update_from_mouse(self, mouse_x: int) -> None:
        """Update value from mouse x position."""
        new_value = self._x_to_value(mouse_x)

        if abs(new_value - self.value) > 0.001:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)

    def set_value(self, value: float) -> None:
        """Set slider value without triggering callback."""
        self.value = max(self.min_value, min(self.max_value, value))

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the slider widget."""
        self._ensure_font()

        # Draw track background
        track_rect = self.track_rect
        pygame.draw.rect(surface, self.track_color, track_rect, border_radius=2)

        # Draw filled portion
        handle_x = self._value_to_x(self.value)
        fill_rect = pygame.Rect(
            self.x,
            self.y - self.track_height // 2,
            handle_x - self.x,
            self.track_height
        )
        if fill_rect.width > 0:
            pygame.draw.rect(surface, self.fill_color, fill_rect, border_radius=2)

        # Draw handle
        handle_color = self.handle_color
        if self.dragging:
            # Slightly larger when dragging
            radius = self.handle_radius + 2
        elif self.hovered:
            radius = self.handle_radius + 1
        else:
            radius = self.handle_radius

        pygame.draw.circle(surface, handle_color, (handle_x, self.y), radius)

        # Draw handle border
        pygame.draw.circle(
            surface, self.track_color, (handle_x, self.y), radius, 2
        )

        # Draw label if provided
        if self.label:
            label_surface = self._font.render(self.label, True, (180, 180, 180))
            label_x = self.x
            label_y = self.y - self.handle_radius - 18
            surface.blit(label_surface, (label_x, label_y))

        # Draw value
        value_text = f"{self.value:.0%}"
        value_surface = self._font.render(value_text, True, (200, 200, 200))
        value_x = self.x + self.width + 10
        value_y = self.y - value_surface.get_height() // 2
        surface.blit(value_surface, (value_x, value_y))

    def set_position(self, x: int, y: int) -> None:
        """Update slider position."""
        self.x = x
        self.y = y


class ExplosionSlider(Slider):
    """
    Specialized slider for explosion control with additional features.
    """

    def __init__(self, x: int, y: int, width: int = None):
        super().__init__(
            x=x,
            y=y,
            width=width,
            min_value=0.0,
            max_value=1.0,
            initial_value=0.0,
            label="",  # No label - assembly info shown above
        )

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the explosion slider with endpoint labels."""
        self._ensure_font()

        # Draw track background
        track_rect = self.track_rect
        pygame.draw.rect(surface, self.track_color, track_rect, border_radius=2)

        # Draw filled portion
        handle_x = self._value_to_x(self.value)
        fill_rect = pygame.Rect(
            self.x,
            self.y - self.track_height // 2,
            handle_x - self.x,
            self.track_height
        )
        if fill_rect.width > 0:
            pygame.draw.rect(surface, self.fill_color, fill_rect, border_radius=2)

        # Draw handle
        handle_color = self.handle_color
        if self.dragging:
            radius = self.handle_radius + 2
        elif self.hovered:
            radius = self.handle_radius + 1
        else:
            radius = self.handle_radius

        pygame.draw.circle(surface, handle_color, (handle_x, self.y), radius)
        pygame.draw.circle(
            surface, self.track_color, (handle_x, self.y), radius, 2
        )

        # Draw value on the right
        value_text = f"{self.value:.0%}"
        value_surface = self._font.render(value_text, True, (200, 200, 200))
        value_x = self.x + self.width + 10
        value_y = self.y - value_surface.get_height() // 2
        surface.blit(value_surface, (value_x, value_y))

        # Draw endpoint labels BELOW the slider
        assembled_surface = self._font.render("Assembled", True, (100, 100, 100))
        exploded_surface = self._font.render("Exploded", True, (100, 100, 100))

        label_y = self.y + self.handle_radius + 8

        surface.blit(assembled_surface, (self.x, label_y))
        surface.blit(
            exploded_surface,
            (self.x + self.width - exploded_surface.get_width(), label_y)
        )
