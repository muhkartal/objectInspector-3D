"""
Assembly data structures for multi-part models with exploded view support.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.geometry.mesh import Mesh
from src.geometry.transforms import translation_matrix


@dataclass
class AssemblyPart:
    """
    Represents a single part in an assembly.

    Attributes:
        mesh: The 3D mesh for this part
        name: Human-readable name for the part
        base_position: Original position in the assembly (3D vector)
        explosion_direction: Direction to move when exploding (normalized 3D vector)
        explosion_distance: How far to move when fully exploded
        color: RGB color tuple for this part
        label_offset: Optional offset for the label position
    """
    mesh: Mesh
    name: str
    base_position: np.ndarray
    explosion_direction: np.ndarray
    explosion_distance: float
    color: Tuple[int, int, int]
    label_offset: Optional[np.ndarray] = None

    def __post_init__(self):
        """Ensure arrays are proper numpy arrays."""
        self.base_position = np.asarray(self.base_position, dtype=np.float32)
        self.explosion_direction = np.asarray(self.explosion_direction, dtype=np.float32)

        # Normalize explosion direction
        length = np.linalg.norm(self.explosion_direction)
        if length > 1e-8:
            self.explosion_direction = self.explosion_direction / length

        if self.label_offset is not None:
            self.label_offset = np.asarray(self.label_offset, dtype=np.float32)

    def get_exploded_position(self, factor: float) -> np.ndarray:
        """
        Get the position of this part at a given explosion factor.

        Args:
            factor: Explosion factor (0.0 = assembled, 1.0 = fully exploded)

        Returns:
            3D position vector
        """
        offset = self.explosion_direction * self.explosion_distance * factor
        return self.base_position + offset

    def get_exploded_mesh(self, factor: float) -> Mesh:
        """
        Get the mesh transformed to its exploded position.

        Args:
            factor: Explosion factor (0.0 = assembled, 1.0 = fully exploded)

        Returns:
            Transformed mesh at exploded position
        """
        position = self.get_exploded_position(factor)
        transform = translation_matrix(position[0], position[1], position[2])
        return self.mesh.transform(transform)

    def get_label_position(self, factor: float) -> np.ndarray:
        """
        Get the position for the label at a given explosion factor.

        Args:
            factor: Explosion factor

        Returns:
            3D position for the label anchor
        """
        base = self.get_exploded_position(factor)
        if self.label_offset is not None:
            return base + self.label_offset
        # Default: use mesh center
        return base


class Assembly:
    """
    Represents a multi-part assembly with exploded view capabilities.

    Manages a collection of parts that can be smoothly animated
    between assembled and exploded states.
    """

    def __init__(self, name: str = "Assembly"):
        self.name = name
        self.parts: Dict[str, AssemblyPart] = {}
        self.explosion_factor: float = 0.0  # 0.0 = assembled, 1.0 = exploded
        self._target_explosion: float = 0.0
        self._animation_speed: float = 2.0  # Units per second

    def add_part(self, part: AssemblyPart) -> None:
        """Add a part to the assembly."""
        self.parts[part.name] = part

    def remove_part(self, name: str) -> bool:
        """Remove a part from the assembly."""
        if name in self.parts:
            del self.parts[name]
            return True
        return False

    def get_part(self, name: str) -> Optional[AssemblyPart]:
        """Get a part by name."""
        return self.parts.get(name)

    @property
    def part_count(self) -> int:
        """Return the number of parts in the assembly."""
        return len(self.parts)

    @property
    def part_names(self) -> List[str]:
        """Return list of part names."""
        return list(self.parts.keys())

    def set_explosion(self, factor: float, animate: bool = False) -> None:
        """
        Set the explosion factor.

        Args:
            factor: Target explosion factor (0.0 to 1.0)
            animate: If True, smoothly animate to target; if False, set immediately
        """
        factor = max(0.0, min(1.0, factor))

        if animate:
            self._target_explosion = factor
        else:
            self.explosion_factor = factor
            self._target_explosion = factor

    def update(self, dt: float) -> None:
        """
        Update animation state.

        Args:
            dt: Delta time in seconds
        """
        if abs(self.explosion_factor - self._target_explosion) > 0.001:
            direction = 1.0 if self._target_explosion > self.explosion_factor else -1.0
            step = self._animation_speed * dt * direction

            if direction > 0:
                self.explosion_factor = min(
                    self._target_explosion,
                    self.explosion_factor + step
                )
            else:
                self.explosion_factor = max(
                    self._target_explosion,
                    self.explosion_factor + step
                )

    def get_exploded_mesh(self, part_name: str) -> Optional[Mesh]:
        """
        Get the exploded mesh for a specific part.

        Args:
            part_name: Name of the part

        Returns:
            Transformed mesh at current explosion state, or None if not found
        """
        part = self.parts.get(part_name)
        if part is None:
            return None
        return part.get_exploded_mesh(self.explosion_factor)

    def get_all_meshes(self) -> List[Tuple[str, Mesh, Tuple[int, int, int]]]:
        """
        Get all parts as exploded meshes with their colors.

        Returns:
            List of (name, mesh, color) tuples for all parts
        """
        result = []
        for name, part in self.parts.items():
            mesh = part.get_exploded_mesh(self.explosion_factor)
            result.append((name, mesh, part.color))
        return result

    def get_all_label_positions(self) -> List[Tuple[str, np.ndarray]]:
        """
        Get label positions for all parts.

        Returns:
            List of (name, position) tuples
        """
        result = []
        for name, part in self.parts.items():
            pos = part.get_label_position(self.explosion_factor)
            result.append((name, pos))
        return result

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the entire assembly at current explosion state.

        Returns:
            (min_point, max_point) tuple
        """
        if not self.parts:
            return np.zeros(3), np.zeros(3)

        all_min = np.array([float('inf'), float('inf'), float('inf')])
        all_max = np.array([float('-inf'), float('-inf'), float('-inf')])

        for part in self.parts.values():
            mesh = part.get_exploded_mesh(self.explosion_factor)
            min_pt, max_pt = mesh.bounds
            all_min = np.minimum(all_min, min_pt)
            all_max = np.maximum(all_max, max_pt)

        return all_min, all_max

    def get_center(self) -> np.ndarray:
        """Get the center of the assembly bounds."""
        min_pt, max_pt = self.get_bounds()
        return (min_pt + max_pt) / 2

    def toggle_explosion(self, animate: bool = True) -> None:
        """Toggle between assembled and exploded states."""
        if self.explosion_factor < 0.5:
            self.set_explosion(1.0, animate=animate)
        else:
            self.set_explosion(0.0, animate=animate)

    def is_animating(self) -> bool:
        """Check if currently animating."""
        return abs(self.explosion_factor - self._target_explosion) > 0.001
