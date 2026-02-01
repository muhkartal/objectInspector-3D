"""
UI Panel for displaying information and controls.
Styled to match audio-visualizer with Tesla-style explosion controls.
"""

from typing import Dict, Any, Optional, List, Callable

import pygame

from config import settings
from src.ui.slider import ExplosionSlider


class UIPanel:
    """
    UI overlay panel for displaying information and help.
    Styled similar to audio-visualizer.
    """

    def __init__(self):
        # Fonts
        self.font: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None

        # State
        self.show_help = False
        self.show_stats = settings.SHOW_STATS

        # Colors (matching audio-visualizer)
        self.color_title = (255, 255, 255)
        self.color_section = (150, 200, 255)
        self.color_item = (180, 180, 180)
        self.color_key = (255, 255, 255)
        self.color_dim = (120, 120, 120)

        # Explosion slider
        self.explosion_slider: Optional[ExplosionSlider] = None
        self.show_explosion_slider = False
        self.on_explosion_change: Optional[Callable[[float], None]] = None

        # Assembly info
        self.assembly_name: Optional[str] = None
        self.assembly_part_count: int = 0

        # Help content organized by sections
        self.help_sections = [
            {
                "title": "Shape Selection",
                "items": [
                    ("1-7", "Basic shapes (Cube, Sphere, etc.)"),
                    ("Space", "Toggle auto-rotation"),
                ]
            },
            {
                "title": "Camera Control",
                "items": [
                    ("Left Drag", "Orbit camera"),
                    ("Right Drag", "Pan camera"),
                    ("Scroll", "Zoom in/out"),
                    ("R", "Reset camera"),
                ]
            },
            {
                "title": "Visualization",
                "items": [
                    ("Tab", "Cycle view mode"),
                    ("", "(Solid / Wireframe / Points / Exploded)"),
                    ("E", "Toggle exploded view"),
                ]
            },
            {
                "title": "Assemblies",
                "items": [
                    ("A", "Cycle all assemblies (15 total)"),
                    ("F1-F4", "Basic: Engine, Gearbox, Watch"),
                    ("F5-F8", "Complex: Jet, Robot, Satellite, Microscope"),
                    ("F9-F12", "Arch: Differential, Bridge, Station, Turbine"),
                    ("L", "Load model file (.obj, .gltf)"),
                ]
            },
            {
                "title": "Effects",
                "items": [
                    ("G", "Toggle glow"),
                    ("V", "Toggle vignette"),
                ]
            },
            {
                "title": "Other",
                "items": [
                    ("H", "Toggle this help"),
                    ("Esc", "Quit"),
                ]
            },
        ]

    def _ensure_fonts(self):
        """Initialize fonts if needed."""
        if self.font is None:
            self.font = pygame.font.SysFont("consolas", 14)
            self.font_large = pygame.font.SysFont("consolas", 16)

    def draw(
        self,
        surface: pygame.Surface,
        state: Dict[str, Any],
    ):
        """Draw UI overlay."""
        self._ensure_fonts()

        width = surface.get_width()
        height = surface.get_height()

        # Draw top info bar
        self._draw_info_bar(surface, state, width)

        # Draw bottom controls hint
        self._draw_controls_hint(surface, width, height)

        # Draw explosion slider if in exploded mode
        if self.show_explosion_slider and self.explosion_slider:
            self._draw_explosion_panel(surface, width, height)

        # Draw help overlay if enabled
        if self.show_help:
            self._draw_help_overlay(surface, width, height)

    def _draw_info_bar(
        self,
        surface: pygame.Surface,
        state: Dict[str, Any],
        width: int,
    ):
        """Draw top information bar."""
        # Background
        bar_rect = pygame.Rect(0, 0, width, settings.PANEL_HEIGHT)
        bar_surface = pygame.Surface((width, settings.PANEL_HEIGHT), pygame.SRCALPHA)
        bar_surface.fill((*settings.PANEL_COLOR, settings.PANEL_ALPHA))
        surface.blit(bar_surface, (0, 0))

        y = 12

        # FPS
        fps = state.get("fps", 0)
        fps_text = f"FPS: {fps:.0f}"
        fps_surface = self.font.render(fps_text, True, self.color_dim)
        surface.blit(fps_surface, (10, y))

        # Current shape
        shape = state.get("shape", "cube")
        shape_text = f"Shape: {shape.capitalize()}"
        shape_surface = self.font.render(shape_text, True, self.color_item)
        surface.blit(shape_surface, (100, y))

        # View mode
        mode = state.get("mode", "solid")
        mode_text = f"View: {mode.capitalize()}"
        mode_surface = self.font.render(mode_text, True, self.color_item)
        surface.blit(mode_surface, (250, y))

    def _draw_controls_hint(
        self,
        surface: pygame.Surface,
        width: int,
        height: int,
    ):
        """Draw bottom controls hint bar (like audio-visualizer)."""
        controls = [
            "H: Help",
            "Tab: Mode",
            "1-7: Shapes",
            "A: Cycle Assemblies",
            "F1-F12: Load Assembly",
            "ESC: Quit"
        ]
        help_text = "  |  ".join(controls)

        text_surface = self.font.render(help_text, True, self.color_dim)
        text_rect = text_surface.get_rect(center=(width // 2, height - 15))

        # Draw background
        bg_rect = text_rect.inflate(20, 8)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 150))
        surface.blit(bg_surface, bg_rect.topleft)

        surface.blit(text_surface, text_rect)

    def _draw_help_overlay(
        self,
        surface: pygame.Surface,
        width: int,
        height: int,
    ):
        """Draw help overlay (styled like audio-visualizer)."""
        # Semi-transparent black background
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        surface.blit(overlay, (0, 0))

        # Title
        x_start = 50
        y = 50

        title = "Object Inspector 3D - Controls"
        title_surface = self.font_large.render(title, True, self.color_title)
        surface.blit(title_surface, (x_start, y))
        y += 40

        # Two-column layout
        col_width = (width - 100) // 2
        col = 0
        col_y_start = y

        # Draw sections
        for section in self.help_sections:
            # Check if we need to switch to second column
            if y > height - 120 and col == 0:
                col = 1
                y = col_y_start

            x = x_start + col * col_width

            # Section title
            section_surface = self.font_large.render(
                section["title"], True, self.color_section
            )
            surface.blit(section_surface, (x, y))
            y += 22

            # Section items
            for key, description in section["items"]:
                if key:
                    # Key + description format
                    key_surface = self.font.render(f"  {key}:", True, self.color_key)
                    surface.blit(key_surface, (x, y))

                    desc_surface = self.font.render(description, True, self.color_item)
                    surface.blit(desc_surface, (x + 100, y))
                else:
                    # Description only (continuation)
                    desc_surface = self.font.render(f"    {description}", True, self.color_dim)
                    surface.blit(desc_surface, (x, y))

                y += 18

            # Space between sections
            y += 10

        # Footer
        footer = "Press H to close"
        footer_surface = self.font.render(footer, True, self.color_dim)
        footer_rect = footer_surface.get_rect(center=(width // 2, height - 30))
        surface.blit(footer_surface, footer_rect)

    def toggle_help(self):
        """Toggle help overlay visibility."""
        self.show_help = not self.show_help

    def toggle_stats(self):
        """Toggle stats display."""
        self.show_stats = not self.show_stats

    def _draw_explosion_panel(
        self,
        surface: pygame.Surface,
        width: int,
        height: int,
    ):
        """Draw the explosion control panel."""
        # Panel dimensions
        panel_width = settings.SLIDER_WIDTH + 120
        panel_height = 80
        panel_x = width - panel_width - settings.SLIDER_MARGIN
        panel_y = settings.PANEL_HEIGHT + settings.SLIDER_MARGIN

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((30, 30, 35, 220))
        surface.blit(panel_surface, (panel_x, panel_y))

        # Draw border
        pygame.draw.rect(surface, (50, 50, 55), panel_rect, 1, border_radius=4)

        # Update slider position
        slider_x = panel_x + 10
        slider_y = panel_y + 45
        self.explosion_slider.set_position(slider_x, slider_y)

        # Draw slider
        self.explosion_slider.draw(surface)

        # Draw assembly info
        if self.assembly_name:
            title_text = f"{self.assembly_name}"
            title_surface = self.font.render(title_text, True, self.color_title)
            surface.blit(title_surface, (panel_x + 10, panel_y + 8))

            parts_text = f"{self.assembly_part_count} parts"
            parts_surface = self.font.render(parts_text, True, self.color_dim)
            surface.blit(parts_surface, (panel_x + 10, panel_y + 24))

    def setup_explosion_slider(self, on_change: Callable[[float], None] = None):
        """Initialize the explosion slider."""
        self.explosion_slider = ExplosionSlider(
            x=100,  # Will be updated in draw
            y=100,
            width=settings.SLIDER_WIDTH
        )
        self.on_explosion_change = on_change
        if on_change:
            self.explosion_slider.on_change = on_change

    def set_explosion_slider_visible(self, visible: bool):
        """Show or hide the explosion slider."""
        self.show_explosion_slider = visible

    def set_assembly_info(self, name: str, part_count: int):
        """Set the assembly information for display."""
        self.assembly_name = name
        self.assembly_part_count = part_count

    def set_explosion_value(self, value: float):
        """Set the slider value without triggering callback."""
        if self.explosion_slider:
            self.explosion_slider.set_value(value)

    def handle_slider_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for the explosion slider.

        Returns:
            True if event was consumed
        """
        if self.show_explosion_slider and self.explosion_slider:
            return self.explosion_slider.handle_event(event)
        return False
