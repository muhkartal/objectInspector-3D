"""
Procedural 3D primitive shape generators.
"""

import numpy as np

from src.geometry.mesh import Mesh


def create_cube(size: float = 1.0, center: tuple = (0, 0, 0)) -> Mesh:
    """
    Create a cube mesh.

    Args:
        size: Length of each edge
        center: Center position (x, y, z)

    Returns:
        Cube Mesh
    """
    s = size / 2
    cx, cy, cz = center

    # 8 vertices of the cube
    vertices = np.array(
        [
            [-s + cx, -s + cy, -s + cz],  # 0: back-bottom-left
            [s + cx, -s + cy, -s + cz],  # 1: back-bottom-right
            [s + cx, s + cy, -s + cz],  # 2: back-top-right
            [-s + cx, s + cy, -s + cz],  # 3: back-top-left
            [-s + cx, -s + cy, s + cz],  # 4: front-bottom-left
            [s + cx, -s + cy, s + cz],  # 5: front-bottom-right
            [s + cx, s + cy, s + cz],  # 6: front-top-right
            [-s + cx, s + cy, s + cz],  # 7: front-top-left
        ],
        dtype=np.float32,
    )

    # 12 triangles (2 per face, 6 faces)
    faces = np.array(
        [
            # Front face
            [4, 5, 6],
            [4, 6, 7],
            # Back face
            [1, 0, 3],
            [1, 3, 2],
            # Top face
            [7, 6, 2],
            [7, 2, 3],
            # Bottom face
            [0, 1, 5],
            [0, 5, 4],
            # Right face
            [5, 1, 2],
            [5, 2, 6],
            # Left face
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.int32,
    )

    return Mesh(vertices=vertices, faces=faces)


def create_sphere(
    radius: float = 1.0,
    segments: int = 24,
    rings: int = 16,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a UV sphere mesh.

    Args:
        radius: Sphere radius
        segments: Number of longitudinal segments
        rings: Number of latitudinal rings
        center: Center position (x, y, z)

    Returns:
        Sphere Mesh
    """
    cx, cy, cz = center
    vertices = []
    faces = []

    # Generate vertices
    for i in range(rings + 1):
        phi = np.pi * i / rings  # 0 to pi
        for j in range(segments):
            theta = 2 * np.pi * j / segments  # 0 to 2*pi

            x = radius * np.sin(phi) * np.cos(theta) + cx
            y = radius * np.cos(phi) + cy
            z = radius * np.sin(phi) * np.sin(theta) + cz

            vertices.append([x, y, z])

    # Generate faces
    for i in range(rings):
        for j in range(segments):
            # Current vertex indices
            curr = i * segments + j
            next_j = i * segments + (j + 1) % segments
            curr_below = (i + 1) * segments + j
            next_below = (i + 1) * segments + (j + 1) % segments

            # Two triangles per quad
            if i != 0:  # Skip degenerate triangles at top pole
                faces.append([curr, curr_below, next_j])
            if i != rings - 1:  # Skip degenerate triangles at bottom pole
                faces.append([next_j, curr_below, next_below])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_cylinder(
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 24,
    center: tuple = (0, 0, 0),
    capped: bool = True,
) -> Mesh:
    """
    Create a cylinder mesh.

    Args:
        radius: Cylinder radius
        height: Cylinder height
        segments: Number of segments around the circumference
        center: Center position (x, y, z)
        capped: Whether to include top and bottom caps

    Returns:
        Cylinder Mesh
    """
    cx, cy, cz = center
    h = height / 2
    vertices = []
    faces = []

    # Generate side vertices (top and bottom rings)
    for i in range(segments):
        theta = 2 * np.pi * i / segments

        x = radius * np.cos(theta) + cx
        z = radius * np.sin(theta) + cz

        # Bottom ring
        vertices.append([x, -h + cy, z])
        # Top ring
        vertices.append([x, h + cy, z])

    # Generate side faces
    for i in range(segments):
        # Vertex indices
        bottom_curr = i * 2
        top_curr = i * 2 + 1
        bottom_next = ((i + 1) % segments) * 2
        top_next = ((i + 1) % segments) * 2 + 1

        # Two triangles per quad
        faces.append([bottom_curr, bottom_next, top_curr])
        faces.append([top_curr, bottom_next, top_next])

    if capped:
        # Add center vertices for caps
        bottom_center_idx = len(vertices)
        vertices.append([cx, -h + cy, cz])
        top_center_idx = len(vertices)
        vertices.append([cx, h + cy, cz])

        # Generate cap faces
        for i in range(segments):
            next_i = (i + 1) % segments

            # Bottom cap (reversed winding)
            bottom_curr = i * 2
            bottom_next = next_i * 2
            faces.append([bottom_center_idx, bottom_next, bottom_curr])

            # Top cap
            top_curr = i * 2 + 1
            top_next = next_i * 2 + 1
            faces.append([top_center_idx, top_curr, top_next])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_cone(
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 24,
    center: tuple = (0, 0, 0),
    capped: bool = True,
) -> Mesh:
    """
    Create a cone mesh.

    Args:
        radius: Base radius
        height: Cone height
        segments: Number of segments around the base
        center: Center position (x, y, z)
        capped: Whether to include the base cap

    Returns:
        Cone Mesh
    """
    cx, cy, cz = center
    h = height / 2
    vertices = []
    faces = []

    # Apex vertex
    apex_idx = 0
    vertices.append([cx, h + cy, cz])

    # Base ring vertices
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = radius * np.cos(theta) + cx
        z = radius * np.sin(theta) + cz
        vertices.append([x, -h + cy, z])

    # Generate side faces
    for i in range(segments):
        curr = i + 1
        next_v = (i + 1) % segments + 1
        faces.append([apex_idx, curr, next_v])

    if capped:
        # Add center vertex for base
        base_center_idx = len(vertices)
        vertices.append([cx, -h + cy, cz])

        # Generate base faces (reversed winding)
        for i in range(segments):
            curr = i + 1
            next_v = (i + 1) % segments + 1
            faces.append([base_center_idx, next_v, curr])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_segments: int = 24,
    minor_segments: int = 12,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a torus mesh.

    Args:
        major_radius: Distance from center to tube center
        minor_radius: Radius of the tube
        major_segments: Segments around the torus
        minor_segments: Segments around the tube
        center: Center position (x, y, z)

    Returns:
        Torus Mesh
    """
    cx, cy, cz = center
    vertices = []
    faces = []

    # Generate vertices
    for i in range(major_segments):
        theta = 2 * np.pi * i / major_segments  # Around the torus

        # Center of the tube at this angle
        tube_center_x = major_radius * np.cos(theta)
        tube_center_z = major_radius * np.sin(theta)

        for j in range(minor_segments):
            phi = 2 * np.pi * j / minor_segments  # Around the tube

            # Point on the tube surface
            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta) + cx
            y = minor_radius * np.sin(phi) + cy
            z = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta) + cz

            vertices.append([x, y, z])

    # Generate faces
    for i in range(major_segments):
        for j in range(minor_segments):
            # Current vertex indices
            curr = i * minor_segments + j
            next_j = i * minor_segments + (j + 1) % minor_segments
            curr_next_i = ((i + 1) % major_segments) * minor_segments + j
            next_both = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments

            # Two triangles per quad
            faces.append([curr, curr_next_i, next_j])
            faces.append([next_j, curr_next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_plane(
    width: float = 2.0,
    height: float = 2.0,
    segments_x: int = 1,
    segments_y: int = 1,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a flat plane mesh.

    Args:
        width: Width of the plane (X direction)
        height: Height of the plane (Z direction)
        segments_x: Subdivisions in X direction
        segments_y: Subdivisions in Z direction
        center: Center position (x, y, z)

    Returns:
        Plane Mesh
    """
    cx, cy, cz = center
    vertices = []
    faces = []

    # Generate vertices
    for j in range(segments_y + 1):
        for i in range(segments_x + 1):
            x = (i / segments_x - 0.5) * width + cx
            z = (j / segments_y - 0.5) * height + cz
            vertices.append([x, cy, z])

    # Generate faces
    for j in range(segments_y):
        for i in range(segments_x):
            # Vertex indices
            curr = j * (segments_x + 1) + i
            next_i = curr + 1
            curr_next_j = (j + 1) * (segments_x + 1) + i
            next_both = curr_next_j + 1

            # Two triangles per quad
            faces.append([curr, curr_next_j, next_i])
            faces.append([next_i, curr_next_j, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_pyramid(
    base_size: float = 1.0,
    height: float = 1.5,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a square pyramid mesh.

    Args:
        base_size: Size of the square base
        height: Height of the pyramid
        center: Center position (x, y, z)

    Returns:
        Pyramid Mesh
    """
    cx, cy, cz = center
    s = base_size / 2
    h = height / 2

    vertices = np.array(
        [
            [cx, h + cy, cz],  # 0: apex
            [-s + cx, -h + cy, -s + cz],  # 1: back-left
            [s + cx, -h + cy, -s + cz],  # 2: back-right
            [s + cx, -h + cy, s + cz],  # 3: front-right
            [-s + cx, -h + cy, s + cz],  # 4: front-left
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            # Side faces
            [0, 4, 3],  # Front
            [0, 3, 2],  # Right
            [0, 2, 1],  # Back
            [0, 1, 4],  # Left
            # Base (two triangles)
            [1, 2, 3],
            [1, 3, 4],
        ],
        dtype=np.int32,
    )

    return Mesh(vertices=vertices, faces=faces)


# Factory function to create shapes by name
SHAPE_CREATORS = {
    "cube": create_cube,
    "sphere": create_sphere,
    "cylinder": create_cylinder,
    "cone": create_cone,
    "torus": create_torus,
    "plane": create_plane,
    "pyramid": create_pyramid,
}


def create_shape(name: str, **kwargs) -> Mesh:
    """
    Create a shape by name.

    Args:
        name: Shape name (cube, sphere, cylinder, cone, torus, plane, pyramid)
        **kwargs: Shape-specific parameters

    Returns:
        Mesh for the requested shape

    Raises:
        ValueError: If shape name is not recognized
    """
    if name not in SHAPE_CREATORS:
        raise ValueError(
            f"Unknown shape: {name}. Available: {list(SHAPE_CREATORS.keys())}"
        )
    return SHAPE_CREATORS[name](**kwargs)
