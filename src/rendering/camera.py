"""
3D Camera with orbit, pan, and zoom controls.
"""

import numpy as np

from config import settings
from src.geometry.transforms import look_at_matrix, perspective_matrix


class Camera:
    """
    Orbit camera that rotates around a target point.

    Supports:
    - Orbit (rotate around target)
    - Pan (move target point)
    - Zoom (change distance to target)
    """

    def __init__(self):
        # Camera position in spherical coordinates
        self.yaw = settings.CAMERA_INITIAL_YAW  # Horizontal angle
        self.pitch = settings.CAMERA_INITIAL_PITCH  # Vertical angle
        self.distance = settings.CAMERA_DISTANCE

        # Target point (center of rotation)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Projection parameters
        self.fov = settings.CAMERA_FOV
        self.near = settings.CAMERA_NEAR
        self.far = settings.CAMERA_FAR
        self.aspect = settings.WINDOW_WIDTH / settings.WINDOW_HEIGHT

        # Control sensitivity
        self.orbit_sensitivity = settings.ORBIT_SENSITIVITY
        self.pan_sensitivity = settings.PAN_SENSITIVITY
        self.zoom_sensitivity = settings.ZOOM_SENSITIVITY

        # Computed matrices (cached)
        self._view_matrix = None
        self._projection_matrix = None
        self._dirty = True

    @property
    def position(self) -> np.ndarray:
        """Get camera position in world space."""
        # Convert spherical to Cartesian
        x = self.distance * np.cos(self.pitch) * np.sin(self.yaw)
        y = self.distance * np.sin(self.pitch)
        z = self.distance * np.cos(self.pitch) * np.cos(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    @property
    def forward(self) -> np.ndarray:
        """Get forward direction vector."""
        direction = self.target - self.position
        return direction / np.linalg.norm(direction)

    @property
    def right(self) -> np.ndarray:
        """Get right direction vector."""
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(self.forward, up)
        length = np.linalg.norm(right)
        if length < 1e-6:
            return np.array([1, 0, 0], dtype=np.float32)
        return right / length

    @property
    def up(self) -> np.ndarray:
        """Get up direction vector."""
        return np.cross(self.right, self.forward)

    @property
    def view_matrix(self) -> np.ndarray:
        """Get the view matrix."""
        if self._dirty or self._view_matrix is None:
            self._update_matrices()
        return self._view_matrix

    @property
    def projection_matrix(self) -> np.ndarray:
        """Get the projection matrix."""
        if self._dirty or self._projection_matrix is None:
            self._update_matrices()
        return self._projection_matrix

    @property
    def view_projection_matrix(self) -> np.ndarray:
        """Get combined view-projection matrix."""
        return self.projection_matrix @ self.view_matrix

    def _update_matrices(self):
        """Recalculate view and projection matrices."""
        self._view_matrix = look_at_matrix(
            self.position, self.target, np.array([0, 1, 0], dtype=np.float32)
        )
        self._projection_matrix = perspective_matrix(
            self.fov, self.aspect, self.near, self.far
        )
        self._dirty = False

    def orbit(self, delta_x: float, delta_y: float):
        """
        Rotate camera around target.

        Args:
            delta_x: Horizontal mouse delta (pixels)
            delta_y: Vertical mouse delta (pixels)
        """
        self.yaw += delta_x * self.orbit_sensitivity * 0.01
        self.pitch -= delta_y * self.orbit_sensitivity * 0.01

        # Clamp pitch to avoid flipping
        self.pitch = np.clip(self.pitch, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)

        self._dirty = True

    def pan(self, delta_x: float, delta_y: float):
        """
        Pan camera (move target point).

        Args:
            delta_x: Horizontal mouse delta (pixels)
            delta_y: Vertical mouse delta (pixels)
        """
        # Scale pan by distance for consistent feel
        scale = self.distance * self.pan_sensitivity

        # Move in camera-relative directions
        self.target -= self.right * delta_x * scale
        self.target += self.up * delta_y * scale

        self._dirty = True

    def zoom(self, delta: float):
        """
        Zoom camera (change distance to target).

        Args:
            delta: Scroll delta (positive = zoom in)
        """
        self.distance *= 1.0 - delta * self.zoom_sensitivity * 0.1
        self.distance = np.clip(
            self.distance,
            settings.CAMERA_MIN_DISTANCE,
            settings.CAMERA_MAX_DISTANCE,
        )
        self._dirty = True

    def set_aspect(self, width: int, height: int):
        """Update aspect ratio on window resize."""
        self.aspect = width / height
        self._dirty = True

    def reset(self):
        """Reset camera to initial state."""
        self.yaw = settings.CAMERA_INITIAL_YAW
        self.pitch = settings.CAMERA_INITIAL_PITCH
        self.distance = settings.CAMERA_DISTANCE
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._dirty = True

    def look_at(self, target: np.ndarray):
        """
        Set camera to look at a specific point.

        Args:
            target: Point to look at
        """
        self.target = np.asarray(target, dtype=np.float32)
        self._dirty = True

    def set_position_spherical(self, yaw: float, pitch: float, distance: float):
        """
        Set camera position in spherical coordinates.

        Args:
            yaw: Horizontal angle (radians)
            pitch: Vertical angle (radians)
            distance: Distance from target
        """
        self.yaw = yaw
        self.pitch = np.clip(pitch, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)
        self.distance = np.clip(
            distance, settings.CAMERA_MIN_DISTANCE, settings.CAMERA_MAX_DISTANCE
        )
        self._dirty = True
