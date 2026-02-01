"""
Post-processing effects for the renderer - Optimized version.
"""

from typing import Dict, Any, Optional

import pygame
import numpy as np

from config import settings


class PostProcessor:
    """
    Applies post-processing effects to the rendered frame.
    Optimized with cached surfaces and minimal allocations.
    """

    def __init__(self, width: int = None, height: int = None):
        self.width = width or settings.WINDOW_WIDTH
        self.height = height or settings.WINDOW_HEIGHT

        # Effect states
        self.glow_enabled = settings.GLOW_ENABLED
        self.glow_intensity = settings.GLOW_INTENSITY

        self.vignette_enabled = settings.VIGNETTE_ENABLED
        self.vignette_strength = settings.VIGNETTE_STRENGTH

        self.scanlines_enabled = settings.SCANLINES_ENABLED
        self.scanlines_intensity = settings.SCANLINES_INTENSITY

        # Pre-computed surfaces (cached)
        self._vignette_surface: Optional[pygame.Surface] = None
        self._scanlines_surface: Optional[pygame.Surface] = None

        # Reusable working surfaces for glow (avoid allocation each frame)
        self._glow_small: Optional[pygame.Surface] = None
        self._glow_tiny: Optional[pygame.Surface] = None
        self._glow_large: Optional[pygame.Surface] = None

        # Dimensions for glow surfaces
        self._glow_small_size = (max(1, self.width // 4), max(1, self.height // 4))
        self._glow_tiny_size = (max(1, self.width // 8), max(1, self.height // 8))

        # Initialize surfaces
        self._init_surfaces()

    def _init_surfaces(self):
        """Initialize reusable surfaces."""
        # Glow surfaces
        self._glow_small = pygame.Surface(self._glow_small_size)
        self._glow_tiny = pygame.Surface(self._glow_tiny_size)
        self._glow_large = pygame.Surface((self.width, self.height))

    def resize(self, width: int, height: int):
        """Handle window resize."""
        if width == self.width and height == self.height:
            return

        self.width = width
        self.height = height

        # Invalidate cached surfaces
        self._vignette_surface = None
        self._scanlines_surface = None

        # Resize glow surfaces
        self._glow_small_size = (max(1, width // 4), max(1, height // 4))
        self._glow_tiny_size = (max(1, width // 8), max(1, height // 8))
        self._glow_small = pygame.Surface(self._glow_small_size)
        self._glow_tiny = pygame.Surface(self._glow_tiny_size)
        self._glow_large = pygame.Surface((width, height))

    def apply(self, surface: pygame.Surface) -> pygame.Surface:
        """Apply all enabled effects to the surface."""
        if self.glow_enabled and self.glow_intensity > 0:
            self._apply_glow_fast(surface)

        if self.vignette_enabled and self.vignette_strength > 0:
            self._apply_vignette(surface)

        if self.scanlines_enabled and self.scanlines_intensity > 0:
            self._apply_scanlines(surface)

        return surface

    def _apply_glow_fast(self, surface: pygame.Surface):
        """Apply glow effect using pre-allocated surfaces."""
        # Downscale to small
        pygame.transform.smoothscale(surface, self._glow_small_size, self._glow_small)

        # Downscale to tiny (blur approximation)
        pygame.transform.smoothscale(self._glow_small, self._glow_tiny_size, self._glow_tiny)

        # Upscale back to full size
        pygame.transform.smoothscale(self._glow_tiny, (self.width, self.height), self._glow_large)

        # Blend with original (additive)
        self._glow_large.set_alpha(int(128 * self.glow_intensity))
        surface.blit(self._glow_large, (0, 0), special_flags=pygame.BLEND_ADD)

    def _apply_vignette(self, surface: pygame.Surface):
        """Apply vignette effect using cached surface."""
        if self._vignette_surface is None:
            self._vignette_surface = self._create_vignette_surface()

        self._vignette_surface.set_alpha(int(255 * self.vignette_strength))
        surface.blit(self._vignette_surface, (0, 0))

    def _create_vignette_surface(self) -> pygame.Surface:
        """Create a vignette overlay surface (cached)."""
        # Create at 1/4 resolution for speed
        scale = 4
        small_w = max(1, self.width // scale)
        small_h = max(1, self.height // scale)

        cx, cy = self.width / 2, self.height / 2
        max_dist = np.sqrt(cx * cx + cy * cy)

        # Create coordinate grids
        y, x = np.mgrid[0:small_h, 0:small_w]
        x = x * scale + scale / 2
        y = y * scale + scale / 2

        # Calculate distance from center (normalized)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist

        # Quadratic falloff
        vignette = np.clip(dist * dist, 0, 1)

        # Create surface with alpha
        alpha = (vignette * 255).astype(np.uint8)
        small_surface = pygame.Surface((small_w, small_h), pygame.SRCALPHA)

        # Fill with black and set alpha
        pixels = pygame.surfarray.pixels_alpha(small_surface)
        pixels[:] = alpha.T
        del pixels

        # Scale to full size
        return pygame.transform.smoothscale(small_surface, (self.width, self.height))

    def _apply_scanlines(self, surface: pygame.Surface):
        """Apply scanline effect using cached surface."""
        if self._scanlines_surface is None:
            self._scanlines_surface = self._create_scanlines_surface()

        self._scanlines_surface.set_alpha(int(255 * self.scanlines_intensity))
        surface.blit(self._scanlines_surface, (0, 0))

    def _create_scanlines_surface(self) -> pygame.Surface:
        """Create a scanlines overlay surface (cached)."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        spacing = settings.SCANLINES_SPACING

        # Draw horizontal lines
        for y in range(0, self.height, spacing):
            pygame.draw.line(surface, (0, 0, 0, 100), (0, y), (self.width, y))

        return surface

    def set_preset(self, preset_name: str):
        """Apply an effect preset."""
        if preset_name not in settings.EFFECT_PRESETS:
            return

        preset = settings.EFFECT_PRESETS[preset_name]

        self.glow_enabled = preset.get("glow", False)
        self.glow_intensity = preset.get("glow_intensity", settings.GLOW_INTENSITY)
        self.vignette_enabled = preset.get("vignette", False)
        self.vignette_strength = preset.get("vignette_strength", settings.VIGNETTE_STRENGTH)
        self.scanlines_enabled = preset.get("scanlines", False)

    def toggle_glow(self):
        """Toggle glow effect."""
        self.glow_enabled = not self.glow_enabled

    def toggle_vignette(self):
        """Toggle vignette effect."""
        self.vignette_enabled = not self.vignette_enabled

    def toggle_scanlines(self):
        """Toggle scanlines effect."""
        self.scanlines_enabled = not self.scanlines_enabled

    def get_state(self) -> Dict[str, Any]:
        """Get current effect states."""
        return {
            "glow": self.glow_enabled,
            "glow_intensity": self.glow_intensity,
            "vignette": self.vignette_enabled,
            "vignette_strength": self.vignette_strength,
            "scanlines": self.scanlines_enabled,
        }
