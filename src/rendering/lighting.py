"""
Simple lighting calculations for 3D rendering.
"""

import numpy as np

from config import settings


class Lighting:
    """
    Simple directional lighting for shading 3D objects.

    Supports:
    - Ambient lighting
    - Diffuse lighting (Lambert)
    - Specular highlights (Phong)
    """

    def __init__(self):
        # Light direction (normalized)
        self.direction = np.array(settings.LIGHT_DIRECTION, dtype=np.float32)
        self.direction = self.direction / np.linalg.norm(self.direction)

        # Light intensities
        self.ambient = settings.AMBIENT_INTENSITY
        self.diffuse = settings.DIFFUSE_INTENSITY
        self.specular = settings.SPECULAR_INTENSITY
        self.specular_power = settings.SPECULAR_POWER

        # Light color (white by default)
        self.color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def set_direction(self, direction: np.ndarray):
        """Set light direction (will be normalized)."""
        self.direction = np.asarray(direction, dtype=np.float32)
        self.direction = self.direction / np.linalg.norm(self.direction)

    def shade_colors_batch(
        self,
        base_colors: np.ndarray,
        normals: np.ndarray,
        view_directions: np.ndarray = None,
    ) -> np.ndarray:
        """
        Apply lighting to multiple colors at once.

        Args:
            base_colors: Nx3 array of RGB colors (0-255)
            normals: Nx3 array of surface normals
            view_directions: Nx3 array of view directions (optional)

        Returns:
            Nx3 array of shaded colors (0-255)
        """
        # Ambient
        intensities = np.full(len(normals), self.ambient, dtype=np.float32)

        # Diffuse
        ndotl = np.sum(normals * self.direction, axis=1)
        positive_mask = ndotl > 0
        intensities[positive_mask] += self.diffuse * ndotl[positive_mask]

        # Specular (if view directions provided)
        if view_directions is not None and self.specular > 0:
            # Reflect light direction
            reflect = 2 * ndotl[:, np.newaxis] * normals - self.direction
            rdotv = np.sum(reflect * view_directions, axis=1)

            specular_mask = positive_mask & (rdotv > 0)
            intensities[specular_mask] += self.specular * np.power(
                rdotv[specular_mask], self.specular_power
            )

        # Apply to colors
        intensities = intensities[:, np.newaxis]
        shaded = base_colors.astype(np.float32) * intensities
        return np.clip(shaded, 0, 255).astype(np.uint8)


# Global lighting instance
_lighting = None


def get_lighting() -> Lighting:
    """Get or create global lighting instance."""
    global _lighting
    if _lighting is None:
        _lighting = Lighting()
    return _lighting
