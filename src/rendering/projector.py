"""
3D to 2D projection and depth sorting - Optimized version.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from src.geometry.mesh import Mesh
from src.rendering.camera import Camera


@dataclass(slots=True)
class ProjectedFace:
    """A face projected to screen coordinates with depth info."""

    vertices_2d: np.ndarray  # 3x2 screen coordinates
    vertices_3d: np.ndarray  # 3x3 world coordinates
    normal: np.ndarray  # Face normal in world space
    depth: float  # Average depth (for sorting)
    face_idx: int  # Original face index
    color: Optional[tuple] = None  # Optional face color


class Projector:
    """
    Projects 3D geometry to 2D screen coordinates.
    Optimized with vectorized numpy operations.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.half_width = width * 0.5
        self.half_height = height * 0.5

        # Cache for grid projection
        self._grid_cache = None
        self._grid_cache_valid = False

    def resize(self, width: int, height: int):
        """Update viewport size."""
        self.width = width
        self.height = height
        self.half_width = width * 0.5
        self.half_height = height * 0.5
        self._grid_cache_valid = False

    def project_point(
        self, point: np.ndarray, camera: Camera
    ) -> Tuple[np.ndarray, float]:
        """Project a single 3D point to 2D screen coordinates."""
        mvp = camera.view_projection_matrix
        point_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        projected = mvp @ point_h

        if abs(projected[3]) < 1e-8:
            return np.array([0, 0]), float("inf")

        inv_w = 1.0 / projected[3]
        screen_x = (projected[0] * inv_w + 1) * self.half_width
        screen_y = (1 - projected[1] * inv_w) * self.half_height

        return np.array([screen_x, screen_y]), projected[2] * inv_w

    def project_vertices(
        self, vertices: np.ndarray, camera: Camera
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project multiple vertices using vectorized operations."""
        mvp = camera.view_projection_matrix
        n = len(vertices)

        # Pre-allocate homogeneous coordinates
        vertices_h = np.empty((n, 4), dtype=np.float32)
        vertices_h[:, :3] = vertices
        vertices_h[:, 3] = 1.0

        # Batch matrix multiplication
        projected = vertices_h @ mvp.T

        # Perspective division with safe reciprocal
        w = projected[:, 3]
        w_safe = np.where(np.abs(w) < 1e-8, 1.0, w)
        inv_w = 1.0 / w_safe

        # NDC to screen coordinates (vectorized)
        screen_x = (projected[:, 0] * inv_w + 1) * self.half_width
        screen_y = (1 - projected[:, 1] * inv_w) * self.half_height
        depths = projected[:, 2] * inv_w

        return np.column_stack([screen_x, screen_y]), depths

    def project_mesh_fast(
        self,
        mesh: Mesh,
        camera: Camera,
        cull_backfaces: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast mesh projection returning numpy arrays for batch rendering.

        Returns:
            face_verts_2d: (N, 3, 2) screen coordinates per face
            face_depths: (N,) average depth per face
            face_normals: (N, 3) normals for visible faces
            face_indices: (N,) original face indices
        """
        # Project all vertices at once
        screen_coords, depths = self.project_vertices(mesh.vertices, camera)

        faces = mesh.faces
        num_faces = len(faces)

        # Get all face vertex indices
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]

        # Get depths for all face vertices
        d0, d1, d2 = depths[v0], depths[v1], depths[v2]
        face_depths = (d0 + d1 + d2) / 3.0

        # Frustum culling: check if all vertices are behind or in front of camera
        behind = (d0 > 1) & (d1 > 1) & (d2 > 1)
        in_front = (d0 < -1) & (d1 < -1) & (d2 < -1)
        frustum_visible = ~(behind | in_front)

        # Backface culling using vectorized dot product
        if cull_backfaces:
            face_normals = mesh.face_normals
            # Compute face centers
            face_centers = (mesh.vertices[v0] + mesh.vertices[v1] + mesh.vertices[v2]) / 3.0
            # View direction from face to camera
            view_dirs = camera.position - face_centers
            # Normalize (fast approximation)
            view_lens = np.sqrt(np.sum(view_dirs * view_dirs, axis=1))
            view_lens = np.maximum(view_lens, 1e-8)
            view_dirs = view_dirs / view_lens[:, np.newaxis]
            # Dot product with normals
            dots = np.sum(face_normals * view_dirs, axis=1)
            front_facing = dots > 0
            visible = frustum_visible & front_facing
        else:
            visible = frustum_visible
            face_normals = mesh.face_normals

        # Screen bounds culling
        s0, s1, s2 = screen_coords[v0], screen_coords[v1], screen_coords[v2]
        min_x = np.minimum(np.minimum(s0[:, 0], s1[:, 0]), s2[:, 0])
        max_x = np.maximum(np.maximum(s0[:, 0], s1[:, 0]), s2[:, 0])
        min_y = np.minimum(np.minimum(s0[:, 1], s1[:, 1]), s2[:, 1])
        max_y = np.maximum(np.maximum(s0[:, 1], s1[:, 1]), s2[:, 1])

        in_bounds = (max_x > -100) & (min_x < self.width + 100) & \
                    (max_y > -100) & (min_y < self.height + 100)
        visible = visible & in_bounds

        # Get visible face indices
        visible_indices = np.where(visible)[0]

        if len(visible_indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Sort by depth (far to near)
        sorted_order = np.argsort(-face_depths[visible_indices])
        sorted_indices = visible_indices[sorted_order]

        # Gather results
        result_verts = np.stack([
            screen_coords[faces[sorted_indices, 0]],
            screen_coords[faces[sorted_indices, 1]],
            screen_coords[faces[sorted_indices, 2]]
        ], axis=1)

        return (
            result_verts,
            face_depths[sorted_indices],
            face_normals[sorted_indices],
            sorted_indices
        )

    def project_mesh(
        self,
        mesh: Mesh,
        camera: Camera,
        cull_backfaces: bool = True,
        sort_faces: bool = True,
    ) -> List[ProjectedFace]:
        """Project mesh - optimized version using batch operations."""
        screen_coords, depths = self.project_vertices(mesh.vertices, camera)
        faces = mesh.faces
        face_normals = mesh.face_normals
        camera_pos = camera.position

        # Vectorized face vertex gathering
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]

        # Face depths
        face_depths = (depths[v0] + depths[v1] + depths[v2]) / 3.0

        # Frustum culling mask
        d0, d1, d2 = depths[v0], depths[v1], depths[v2]
        visible = ~((d0 > 1) & (d1 > 1) & (d2 > 1)) & ~((d0 < -1) & (d1 < -1) & (d2 < -1))

        # Backface culling
        if cull_backfaces:
            verts_v0 = mesh.vertices[v0]
            verts_v1 = mesh.vertices[v1]
            verts_v2 = mesh.vertices[v2]
            face_centers = (verts_v0 + verts_v1 + verts_v2) / 3.0
            view_dirs = camera_pos - face_centers
            view_lens = np.linalg.norm(view_dirs, axis=1)
            view_lens = np.maximum(view_lens, 1e-8)
            view_dirs /= view_lens[:, np.newaxis]
            dots = np.einsum('ij,ij->i', face_normals, view_dirs)
            visible &= (dots > 0)

        # Screen bounds culling
        s0, s1, s2 = screen_coords[v0], screen_coords[v1], screen_coords[v2]
        min_x = np.minimum(np.minimum(s0[:, 0], s1[:, 0]), s2[:, 0])
        max_x = np.maximum(np.maximum(s0[:, 0], s1[:, 0]), s2[:, 0])
        min_y = np.minimum(np.minimum(s0[:, 1], s1[:, 1]), s2[:, 1])
        max_y = np.maximum(np.maximum(s0[:, 1], s1[:, 1]), s2[:, 1])
        visible &= (max_x > -100) & (min_x < self.width + 100)
        visible &= (max_y > -100) & (min_y < self.height + 100)

        # Get visible indices
        visible_idx = np.where(visible)[0]

        if len(visible_idx) == 0:
            return []

        # Sort by depth
        if sort_faces:
            sort_order = np.argsort(-face_depths[visible_idx])
            visible_idx = visible_idx[sort_order]

        # Build result list (optimized pre-allocation)
        projected_faces = []
        has_colors = mesh.colors is not None

        for i in visible_idx:
            face = faces[i]
            face_color = None
            if has_colors:
                face_colors = mesh.colors[face]
                face_color = (int(face_colors[0].mean()),
                             int(face_colors[1].mean()),
                             int(face_colors[2].mean()))

            projected_faces.append(
                ProjectedFace(
                    vertices_2d=screen_coords[face],
                    vertices_3d=mesh.vertices[face],
                    normal=face_normals[i],
                    depth=face_depths[i],
                    face_idx=i,
                    color=face_color,
                )
            )

        return projected_faces

    def project_edges(
        self, mesh: Mesh, camera: Camera
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Project mesh edges - optimized version."""
        screen_coords, depths = self.project_vertices(mesh.vertices, camera)
        edges = mesh.get_edges()

        if len(edges) == 0:
            return []

        v0, v1 = edges[:, 0], edges[:, 1]
        d0, d1 = depths[v0], depths[v1]

        # Visibility check
        visible = ~((d0 > 1) | (d1 > 1) | (d0 < -1) | (d1 < -1))

        # Screen bounds
        p0, p1 = screen_coords[v0], screen_coords[v1]
        both_outside_x = ((p0[:, 0] < -100) & (p1[:, 0] < -100)) | \
                         ((p0[:, 0] > self.width + 100) & (p1[:, 0] > self.width + 100))
        both_outside_y = ((p0[:, 1] < -100) & (p1[:, 1] < -100)) | \
                         ((p0[:, 1] > self.height + 100) & (p1[:, 1] > self.height + 100))
        visible &= ~(both_outside_x | both_outside_y)

        visible_idx = np.where(visible)[0]

        return [(screen_coords[edges[i, 0]], screen_coords[edges[i, 1]])
                for i in visible_idx]

    def project_points(
        self, mesh: Mesh, camera: Camera
    ) -> List[Tuple[np.ndarray, float]]:
        """Project mesh vertices as points - optimized version."""
        screen_coords, depths = self.project_vertices(mesh.vertices, camera)

        # Visibility mask
        visible = (depths <= 1) & (depths >= -1)
        visible &= (screen_coords[:, 0] >= 0) & (screen_coords[:, 0] <= self.width)
        visible &= (screen_coords[:, 1] >= 0) & (screen_coords[:, 1] <= self.height)

        visible_idx = np.where(visible)[0]

        return [(screen_coords[i], depths[i]) for i in visible_idx]
