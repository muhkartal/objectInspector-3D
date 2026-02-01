"""
Mesh data structure for 3D geometry.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Mesh:
    """
    Represents a 3D mesh with vertices, faces, and optional attributes.

    Attributes:
        vertices: Nx3 array of vertex positions (x, y, z)
        faces: Mx3 array of triangle face indices
        normals: Nx3 array of vertex normals (computed if not provided)
        colors: Nx3 or Nx4 array of vertex colors (RGB or RGBA, 0-255)
        uvs: Nx2 array of texture coordinates (optional)
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None

    # Computed properties
    _face_normals: Optional[np.ndarray] = field(default=None, repr=False)
    _center: Optional[np.ndarray] = field(default=None, repr=False)
    _bounds: Optional[tuple] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate and initialize mesh data."""
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)

        if self.normals is None:
            self.normals = self._compute_vertex_normals()
        else:
            self.normals = np.asarray(self.normals, dtype=np.float32)

        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.uint8)

        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float32)

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Return the number of faces."""
        return len(self.faces)

    @property
    def center(self) -> np.ndarray:
        """Return the center of the mesh bounding box."""
        if self._center is None:
            self._center = (self.vertices.min(axis=0) + self.vertices.max(axis=0)) / 2
        return self._center

    @property
    def bounds(self) -> tuple:
        """Return (min_point, max_point) bounding box."""
        if self._bounds is None:
            self._bounds = (self.vertices.min(axis=0), self.vertices.max(axis=0))
        return self._bounds

    @property
    def size(self) -> np.ndarray:
        """Return the size of the bounding box."""
        min_pt, max_pt = self.bounds
        return max_pt - min_pt

    @property
    def face_normals(self) -> np.ndarray:
        """Return Mx3 array of face normals."""
        if self._face_normals is None:
            self._face_normals = self._compute_face_normals()
        return self._face_normals

    def _compute_face_normals(self) -> np.ndarray:
        """Compute normal vectors for each face."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Cross product of two edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)  # Avoid division by zero
        return normals / lengths

    def _compute_vertex_normals(self) -> np.ndarray:
        """Compute smooth vertex normals by averaging face normals (Optimized)."""
        face_normals = self._compute_face_normals()

        # Accumulate face normals at each vertex using vectorization
        vertex_normals = np.zeros_like(self.vertices)

        # np.add.at performs unbuffered in-place addition
        # It handles duplicate indices correctly (accumulating)
        np.add.at(vertex_normals, self.faces.ravel(), face_normals.repeat(3, axis=0))

        # Normalize
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)
        return vertex_normals / lengths

    def get_edges(self) -> np.ndarray:
        """
        Extract unique edges from faces.

        Returns:
            Ex2 array of vertex index pairs forming edges
        """
        edges = set()
        for face in self.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges.add(edge)
        return np.array(list(edges), dtype=np.int32)

    def transform(self, matrix: np.ndarray) -> "Mesh":
        """
        Apply a 4x4 transformation matrix to the mesh.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            New transformed Mesh
        """
        # Convert to homogeneous coordinates
        ones = np.ones((self.num_vertices, 1), dtype=np.float32)
        vertices_h = np.hstack([self.vertices, ones])

        # Apply transformation
        transformed = (matrix @ vertices_h.T).T
        new_vertices = transformed[:, :3]

        # Transform normals (use inverse transpose of upper 3x3)
        normal_matrix = np.linalg.inv(matrix[:3, :3]).T
        new_normals = (normal_matrix @ self.normals.T).T

        # Normalize
        lengths = np.linalg.norm(new_normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)
        new_normals = new_normals / lengths

        return Mesh(
            vertices=new_vertices,
            faces=self.faces.copy(),
            normals=new_normals,
            colors=self.colors.copy() if self.colors is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
        )

    def set_color(self, color: tuple) -> None:
        """Set uniform color for all vertices."""
        self.colors = np.full((self.num_vertices, 3), color, dtype=np.uint8)

    def copy(self) -> "Mesh":
        """Create a deep copy of the mesh."""
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            colors=self.colors.copy() if self.colors is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
        )
