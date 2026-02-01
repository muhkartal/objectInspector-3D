"""
Advanced parametric surface generators inspired by PyVista.
Creates complex mathematical surfaces with high detail.
"""

import numpy as np
from typing import Tuple, Optional

from src.geometry.mesh import Mesh


def create_klein_bottle(
    radius: float = 1.0,
    u_segments: int = 60,
    v_segments: int = 30,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a Klein bottle - a non-orientable surface.

    The Klein bottle is a surface with no distinct inside or outside.

    Args:
        radius: Overall scale of the bottle
        u_segments: Segments around the main loop
        v_segments: Segments around the tube
        center: Center position

    Returns:
        Klein bottle Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = 2 * np.pi * i / u_segments

        for j in range(v_segments):
            v = 2 * np.pi * j / v_segments

            # Klein bottle parametric equations
            if u < np.pi:
                x = 6 * np.cos(u) * (1 + np.sin(u)) + 4 * (1 - np.cos(u) / 2) * np.cos(u) * np.cos(v)
                y = 16 * np.sin(u) + 4 * (1 - np.cos(u) / 2) * np.sin(u) * np.cos(v)
            else:
                x = 6 * np.cos(u) * (1 + np.sin(u)) + 4 * (1 - np.cos(u) / 2) * np.cos(v + np.pi)
                y = 16 * np.sin(u)

            z = 4 * (1 - np.cos(u) / 2) * np.sin(v)

            # Scale and translate
            vertices.append([
                x * radius * 0.05 + cx,
                y * radius * 0.05 + cy,
                z * radius * 0.05 + cz
            ])

    # Generate faces
    faces = []
    for i in range(u_segments):
        for j in range(v_segments):
            curr = i * v_segments + j
            next_j = i * v_segments + (j + 1) % v_segments
            next_i = ((i + 1) % u_segments) * v_segments + j
            next_both = ((i + 1) % u_segments) * v_segments + (j + 1) % v_segments

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_mobius_strip(
    radius: float = 1.0,
    width: float = 0.4,
    u_segments: int = 80,
    v_segments: int = 20,
    twists: int = 1,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a Möbius strip with configurable twists.

    Args:
        radius: Radius of the central circle
        width: Width of the strip
        u_segments: Segments around the strip
        v_segments: Segments across the width
        twists: Number of half-twists (1 = classic Möbius)
        center: Center position

    Returns:
        Möbius strip Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = 2 * np.pi * i / u_segments

        for j in range(v_segments):
            v = (j / (v_segments - 1) - 0.5) * width

            # Möbius strip parametric equations
            twist_angle = twists * u / 2

            x = (radius + v * np.cos(twist_angle)) * np.cos(u) + cx
            y = v * np.sin(twist_angle) + cy
            z = (radius + v * np.cos(twist_angle)) * np.sin(u) + cz

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    for i in range(u_segments):
        for j in range(v_segments - 1):
            curr = i * v_segments + j
            next_j = i * v_segments + j + 1
            next_i = ((i + 1) % u_segments) * v_segments + j
            next_both = ((i + 1) % u_segments) * v_segments + j + 1

            # For Möbius strip, need to handle the twist at the seam
            if i == u_segments - 1 and twists % 2 == 1:
                # Connect to flipped vertices for odd twists
                next_i = v_segments - 1 - j
                next_both = v_segments - 1 - (j + 1)

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_superquadric(
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    e1: float = 1.0,
    e2: float = 1.0,
    u_segments: int = 48,
    v_segments: int = 24,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a superquadric (superellipsoid).

    Superquadrics can represent many shapes by varying e1 and e2:
    - e1=e2=1: ellipsoid
    - e1=e2=0.1: box-like
    - e1=e2=2: star/cross shape
    - e1=0.1, e2=1: cylinder-like

    Args:
        a, b, c: Scale factors for x, y, z axes
        e1: East-west exponent (controls latitude shape)
        e2: North-south exponent (controls longitude shape)
        u_segments: Longitudinal segments
        v_segments: Latitudinal segments
        center: Center position

    Returns:
        Superquadric Mesh
    """
    cx, cy, cz = center
    vertices = []

    def signed_power(x, n):
        """Signed power function for superquadric."""
        return np.sign(x) * np.abs(x) ** n

    for i in range(u_segments):
        u = -np.pi + 2 * np.pi * i / u_segments  # -pi to pi (longitude)

        for j in range(v_segments + 1):
            v = -np.pi / 2 + np.pi * j / v_segments  # -pi/2 to pi/2 (latitude)

            # Superquadric parametric equations
            cos_v = np.cos(v)
            sin_v = np.sin(v)
            cos_u = np.cos(u)
            sin_u = np.sin(u)

            x = a * signed_power(cos_v, e1) * signed_power(cos_u, e2) + cx
            y = c * signed_power(sin_v, e1) + cy
            z = b * signed_power(cos_v, e1) * signed_power(sin_u, e2) + cz

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    v_count = v_segments + 1

    for i in range(u_segments):
        for j in range(v_segments):
            curr = i * v_count + j
            next_j = i * v_count + j + 1
            next_i = ((i + 1) % u_segments) * v_count + j
            next_both = ((i + 1) % u_segments) * v_count + j + 1

            # Skip degenerate triangles at poles
            if j != 0:
                faces.append([curr, next_i, next_j])
            if j != v_segments - 1:
                faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_trefoil_knot(
    radius: float = 1.0,
    tube_radius: float = 0.15,
    p: int = 2,
    q: int = 3,
    u_segments: int = 120,
    v_segments: int = 16,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a torus knot (trefoil when p=2, q=3).

    Args:
        radius: Main radius
        tube_radius: Tube thickness
        p, q: Knot parameters (winds p times around axis, q times around torus)
        u_segments: Segments along the knot
        v_segments: Segments around the tube
        center: Center position

    Returns:
        Torus knot Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        t = 2 * np.pi * i / u_segments

        # Torus knot centerline
        r = np.cos(q * t) + 2
        x0 = r * np.cos(p * t)
        y0 = r * np.sin(p * t)
        z0 = -np.sin(q * t)

        # Calculate tangent
        dx = -q * np.sin(q * t) * np.cos(p * t) - p * r * np.sin(p * t)
        dy = -q * np.sin(q * t) * np.sin(p * t) + p * r * np.cos(p * t)
        dz = -q * np.cos(q * t)

        # Normalize tangent
        tangent = np.array([dx, dy, dz])
        tangent = tangent / np.linalg.norm(tangent)

        # Create orthogonal vectors for tube
        up = np.array([0, 0, 1])
        if abs(np.dot(tangent, up)) > 0.9:
            up = np.array([1, 0, 0])

        normal = np.cross(tangent, up)
        normal = normal / np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)

        for j in range(v_segments):
            theta = 2 * np.pi * j / v_segments

            # Point on tube surface
            offset = tube_radius * (np.cos(theta) * normal + np.sin(theta) * binormal)

            x = (x0 + offset[0]) * radius * 0.3 + cx
            y = (y0 + offset[1]) * radius * 0.3 + cy
            z = (z0 + offset[2]) * radius * 0.3 + cz

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    for i in range(u_segments):
        for j in range(v_segments):
            curr = i * v_segments + j
            next_j = i * v_segments + (j + 1) % v_segments
            next_i = ((i + 1) % u_segments) * v_segments + j
            next_both = ((i + 1) % u_segments) * v_segments + (j + 1) % v_segments

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_seashell(
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.1,
    n: float = 2.0,
    spirals: float = 3.0,
    u_segments: int = 100,
    v_segments: int = 24,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a mathematical seashell (spiral shell).

    Args:
        a: Spiral growth rate
        b: Opening size
        c: Shell thickness
        n: Number of ribs
        spirals: Number of spiral turns
        u_segments: Segments along spiral
        v_segments: Segments around opening
        center: Center position

    Returns:
        Seashell Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = spirals * 2 * np.pi * i / u_segments

        # Spiral parameters
        exp_term = np.exp(u / (6 * np.pi))

        for j in range(v_segments):
            v = 2 * np.pi * j / v_segments

            # Seashell parametric equations
            cos_v = np.cos(v)
            sin_v = np.sin(v)

            # Radius with ribs
            r = b * (1 + c * np.cos(n * v))

            x = a * exp_term * (1 + cos_v) * np.cos(u)
            y = a * exp_term * (1 + cos_v) * np.sin(u)
            z = a * exp_term * sin_v + a * exp_term * u / (2 * np.pi)

            vertices.append([x * 0.15 + cx, z * 0.15 + cy, y * 0.15 + cz])

    # Generate faces
    faces = []
    for i in range(u_segments - 1):
        for j in range(v_segments):
            curr = i * v_segments + j
            next_j = i * v_segments + (j + 1) % v_segments
            next_i = (i + 1) * v_segments + j
            next_both = (i + 1) * v_segments + (j + 1) % v_segments

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_boy_surface(
    radius: float = 1.0,
    u_segments: int = 60,
    v_segments: int = 60,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create Boy's surface - an immersion of the real projective plane.

    Args:
        radius: Scale of the surface
        u_segments: Segments in u direction
        v_segments: Segments in v direction
        center: Center position

    Returns:
        Boy surface Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = np.pi * i / u_segments

        for j in range(v_segments):
            v = np.pi * j / v_segments

            # Boy surface parametric equations (Bryant-Kusner parametrization)
            cos_u = np.cos(u)
            sin_u = np.sin(u)
            cos_v = np.cos(v)
            sin_v = np.sin(v)

            sqrt2 = np.sqrt(2)

            denom = 2 - sqrt2 * np.sin(3 * u) * np.sin(2 * v)

            x = (sqrt2 * cos_v * cos_v * np.cos(2 * u) + cos_u * np.sin(2 * v)) / denom
            y = (sqrt2 * cos_v * cos_v * np.sin(2 * u) - sin_u * np.sin(2 * v)) / denom
            z = 3 * cos_v * cos_v / denom

            vertices.append([
                x * radius + cx,
                y * radius + cy,
                z * radius - radius + cz
            ])

    # Generate faces
    faces = []
    for i in range(u_segments - 1):
        for j in range(v_segments - 1):
            curr = i * v_segments + j
            next_j = i * v_segments + j + 1
            next_i = (i + 1) * v_segments + j
            next_both = (i + 1) * v_segments + j + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_enneper_surface(
    radius: float = 1.0,
    u_segments: int = 50,
    v_segments: int = 50,
    u_range: float = 1.5,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create Enneper's minimal surface.

    Args:
        radius: Scale of the surface
        u_segments: Segments in u direction
        v_segments: Segments in v direction
        u_range: Parameter range
        center: Center position

    Returns:
        Enneper surface Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = -u_range + 2 * u_range * i / (u_segments - 1)

        for j in range(v_segments):
            v = -u_range + 2 * u_range * j / (v_segments - 1)

            # Enneper surface parametric equations
            x = u - u**3 / 3 + u * v**2
            y = v - v**3 / 3 + v * u**2
            z = u**2 - v**2

            vertices.append([
                x * radius * 0.3 + cx,
                z * radius * 0.3 + cy,
                y * radius * 0.3 + cz
            ])

    # Generate faces
    faces = []
    for i in range(u_segments - 1):
        for j in range(v_segments - 1):
            curr = i * v_segments + j
            next_j = i * v_segments + j + 1
            next_i = (i + 1) * v_segments + j
            next_both = (i + 1) * v_segments + j + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_dini_surface(
    a: float = 1.0,
    b: float = 0.2,
    u_segments: int = 80,
    v_segments: int = 20,
    u_max: float = 4 * np.pi,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create Dini's surface - a surface of constant negative curvature.

    Args:
        a: Scale parameter
        b: Twist parameter
        u_segments: Segments in u direction
        v_segments: Segments in v direction
        u_max: Maximum u value (controls length)
        center: Center position

    Returns:
        Dini surface Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = 0.01 + u_max * i / u_segments

        for j in range(v_segments):
            v = 0.01 + (1 - 0.02) * j / (v_segments - 1)

            # Dini surface parametric equations
            x = a * np.cos(u) * np.sin(v)
            y = a * np.sin(u) * np.sin(v)
            z = a * (np.cos(v) + np.log(np.tan(v / 2))) + b * u

            vertices.append([
                x * 0.3 + cx,
                z * 0.1 + cy,
                y * 0.3 + cz
            ])

    # Generate faces
    faces = []
    for i in range(u_segments - 1):
        for j in range(v_segments - 1):
            curr = i * v_segments + j
            next_j = i * v_segments + j + 1
            next_i = (i + 1) * v_segments + j
            next_both = (i + 1) * v_segments + j + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_roman_surface(
    radius: float = 1.0,
    u_segments: int = 50,
    v_segments: int = 50,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create Steiner's Roman surface.

    Args:
        radius: Scale of the surface
        u_segments: Segments in u direction
        v_segments: Segments in v direction
        center: Center position

    Returns:
        Roman surface Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments):
        u = np.pi * i / u_segments

        for j in range(v_segments):
            v = np.pi * j / v_segments

            # Roman surface parametric equations
            cos_u = np.cos(u)
            sin_u = np.sin(u)
            cos_v = np.cos(v)
            sin_v = np.sin(v)
            sin_2u = np.sin(2 * u)
            sin_2v = np.sin(2 * v)

            x = sin_2u * cos_v * cos_v
            y = sin_u * sin_2v
            z = cos_u * sin_2v

            vertices.append([
                x * radius + cx,
                y * radius + cy,
                z * radius + cz
            ])

    # Generate faces
    faces = []
    for i in range(u_segments - 1):
        for j in range(v_segments - 1):
            curr = i * v_segments + j
            next_j = i * v_segments + j + 1
            next_i = (i + 1) * v_segments + j
            next_both = (i + 1) * v_segments + j + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_catenoid(
    radius: float = 0.3,
    height: float = 2.0,
    u_segments: int = 48,
    v_segments: int = 24,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a catenoid - a minimal surface of revolution.

    Args:
        radius: Waist radius
        height: Total height
        u_segments: Segments around
        v_segments: Segments along height
        center: Center position

    Returns:
        Catenoid Mesh
    """
    cx, cy, cz = center
    vertices = []

    c = radius
    h = height / 2

    for i in range(u_segments):
        u = 2 * np.pi * i / u_segments

        for j in range(v_segments + 1):
            v = -h + 2 * h * j / v_segments

            # Catenoid parametric equations
            r = c * np.cosh(v / c)

            x = r * np.cos(u) + cx
            y = v + cy
            z = r * np.sin(u) + cz

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    v_count = v_segments + 1

    for i in range(u_segments):
        for j in range(v_segments):
            curr = i * v_count + j
            next_j = i * v_count + j + 1
            next_i = ((i + 1) % u_segments) * v_count + j
            next_both = ((i + 1) % u_segments) * v_count + j + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_helicoid(
    radius: float = 1.0,
    pitch: float = 0.5,
    turns: float = 2.0,
    u_segments: int = 80,
    v_segments: int = 20,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a helicoid - a ruled minimal surface.

    Args:
        radius: Outer radius
        pitch: Height per turn
        turns: Number of turns
        u_segments: Segments along the helix
        v_segments: Segments across radius
        center: Center position

    Returns:
        Helicoid Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments + 1):
        u = turns * 2 * np.pi * i / u_segments

        for j in range(v_segments + 1):
            v = -radius + 2 * radius * j / v_segments

            # Helicoid parametric equations
            x = v * np.cos(u) + cx
            y = pitch * u / (2 * np.pi) + cy
            z = v * np.sin(u) + cz

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    v_count = v_segments + 1

    for i in range(u_segments):
        for j in range(v_segments):
            curr = i * v_count + j
            next_j = i * v_count + j + 1
            next_i = (i + 1) * v_count + j
            next_both = (i + 1) * v_count + j + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_spring(
    radius: float = 0.5,
    tube_radius: float = 0.08,
    pitch: float = 0.3,
    turns: float = 5.0,
    u_segments: int = 150,
    v_segments: int = 12,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a spring (helical tube).

    Args:
        radius: Coil radius
        tube_radius: Tube thickness
        pitch: Height per turn
        turns: Number of turns
        u_segments: Segments along the helix
        v_segments: Segments around the tube
        center: Center position

    Returns:
        Spring Mesh
    """
    cx, cy, cz = center
    vertices = []

    for i in range(u_segments + 1):
        t = turns * 2 * np.pi * i / u_segments

        # Helix centerline
        hx = radius * np.cos(t)
        hy = pitch * t / (2 * np.pi)
        hz = radius * np.sin(t)

        # Tangent vector
        tx = -radius * np.sin(t)
        ty = pitch / (2 * np.pi)
        tz = radius * np.cos(t)
        t_len = np.sqrt(tx*tx + ty*ty + tz*tz)
        tx, ty, tz = tx/t_len, ty/t_len, tz/t_len

        # Normal and binormal
        nx, ny, nz = -np.cos(t), 0, -np.sin(t)
        bx = ty * nz - tz * ny
        by = tz * nx - tx * nz
        bz = tx * ny - ty * nx

        for j in range(v_segments):
            theta = 2 * np.pi * j / v_segments

            # Point on tube surface
            x = hx + tube_radius * (np.cos(theta) * nx + np.sin(theta) * bx) + cx
            y = hy + tube_radius * (np.cos(theta) * ny + np.sin(theta) * by) + cy
            z = hz + tube_radius * (np.cos(theta) * nz + np.sin(theta) * bz) + cz

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    for i in range(u_segments):
        for j in range(v_segments):
            curr = i * v_segments + j
            next_j = i * v_segments + (j + 1) % v_segments
            next_i = (i + 1) * v_segments + j
            next_both = (i + 1) * v_segments + (j + 1) % v_segments

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_gyroid(
    size: float = 2.0,
    resolution: int = 40,
    threshold: float = 0.0,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a gyroid surface using marching cubes approximation.

    The gyroid is a triply periodic minimal surface.

    Args:
        size: Overall size
        resolution: Grid resolution
        threshold: Isosurface threshold
        center: Center position

    Returns:
        Gyroid Mesh (approximated)
    """
    cx, cy, cz = center

    # Create grid
    lin = np.linspace(-np.pi, np.pi, resolution)
    X, Y, Z = np.meshgrid(lin, lin, lin)

    # Gyroid equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    F = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)

    # Simple marching cubes approximation
    vertices = []
    faces = []

    scale = size / (2 * np.pi)

    for i in range(resolution - 1):
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                # Get cube corner values
                cube_vals = [
                    F[i, j, k], F[i+1, j, k], F[i+1, j+1, k], F[i, j+1, k],
                    F[i, j, k+1], F[i+1, j, k+1], F[i+1, j+1, k+1], F[i, j+1, k+1]
                ]

                # Check if isosurface crosses this cube
                above = sum(1 for v in cube_vals if v > threshold)
                if above > 0 and above < 8:
                    # Add vertex at cube center (simplified)
                    x = (lin[i] + lin[i+1]) / 2 * scale + cx
                    y = (lin[j] + lin[j+1]) / 2 * scale + cy
                    z = (lin[k] + lin[k+1]) / 2 * scale + cz
                    vertices.append([x, y, z])

    # Create faces by connecting nearby vertices (simplified approach)
    if len(vertices) < 4:
        # Fallback to simple surface if not enough vertices
        return create_superquadric(size/2, size/2, size/2, 0.5, 0.5, center=center)

    vertices = np.array(vertices, dtype=np.float32)

    # Use Delaunay-like approach for face generation
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(vertices)
        faces = []
        for simplex in tri.simplices:
            # Only keep surface triangles (those with reasonable edge lengths)
            v0, v1, v2, v3 = vertices[simplex]
            edges = [
                np.linalg.norm(v1 - v0),
                np.linalg.norm(v2 - v0),
                np.linalg.norm(v3 - v0),
            ]
            if max(edges) < size * 0.3:  # Filter long edges
                faces.append([simplex[0], simplex[1], simplex[2]])
                faces.append([simplex[0], simplex[2], simplex[3]])
                faces.append([simplex[0], simplex[1], simplex[3]])
                faces.append([simplex[1], simplex[2], simplex[3]])
    except:
        # Fallback
        return create_superquadric(size/2, size/2, size/2, 0.5, 0.5, center=center)

    if len(faces) == 0:
        return create_superquadric(size/2, size/2, size/2, 0.5, 0.5, center=center)

    return Mesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int32),
    )


# Platonic solids with high detail

def create_icosahedron(radius: float = 1.0, center: tuple = (0, 0, 0)) -> Mesh:
    """Create an icosahedron (20 triangular faces)."""
    cx, cy, cz = center
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # Icosahedron vertices
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float32)

    # Normalize to radius and translate
    vertices = vertices / np.linalg.norm(vertices[0]) * radius
    vertices[:, 0] += cx
    vertices[:, 1] += cy
    vertices[:, 2] += cz

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces)


def create_dodecahedron(radius: float = 1.0, center: tuple = (0, 0, 0)) -> Mesh:
    """Create a dodecahedron (12 pentagonal faces, triangulated)."""
    cx, cy, cz = center
    phi = (1 + np.sqrt(5)) / 2

    # Dodecahedron vertices
    vertices = []

    # Cube vertices
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append([i, j, k])

    # Rectangle vertices
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i * phi, j / phi])
            vertices.append([i / phi, 0, j * phi])
            vertices.append([i * phi, j / phi, 0])

    vertices = np.array(vertices, dtype=np.float32)
    vertices = vertices / np.linalg.norm(vertices[0]) * radius
    vertices[:, 0] += cx
    vertices[:, 1] += cy
    vertices[:, 2] += cz

    # Pentagon face definitions (triangulated)
    faces = [
        [0, 8, 4, 16, 12], [0, 12, 2, 10, 14], [0, 14, 6, 18, 8],
        [1, 9, 5, 17, 13], [1, 13, 3, 11, 15], [1, 15, 7, 19, 9],
        [2, 12, 16, 17, 3], [4, 8, 18, 19, 5], [6, 14, 10, 11, 7],
        [16, 4, 5, 17, 3], [18, 6, 7, 19, 5], [10, 2, 3, 11, 7],
    ]

    # Triangulate pentagons
    tri_faces = []
    for face in faces:
        # Fan triangulation from first vertex
        for i in range(1, len(face) - 1):
            tri_faces.append([face[0], face[i], face[i + 1]])

    return Mesh(vertices=vertices, faces=np.array(tri_faces, dtype=np.int32))


def create_octahedron(radius: float = 1.0, center: tuple = (0, 0, 0)) -> Mesh:
    """Create an octahedron (8 triangular faces)."""
    cx, cy, cz = center

    vertices = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=np.float32) * radius

    vertices[:, 0] += cx
    vertices[:, 1] += cy
    vertices[:, 2] += cz

    faces = np.array([
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ], dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces)


def create_tetrahedron(radius: float = 1.0, center: tuple = (0, 0, 0)) -> Mesh:
    """Create a tetrahedron (4 triangular faces)."""
    cx, cy, cz = center

    # Regular tetrahedron vertices
    vertices = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=np.float32)

    vertices = vertices / np.linalg.norm(vertices[0]) * radius
    vertices[:, 0] += cx
    vertices[:, 1] += cy
    vertices[:, 2] += cz

    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2],
    ], dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces)


def subdivide_mesh(mesh: Mesh, iterations: int = 1) -> Mesh:
    """
    Subdivide a mesh using Loop subdivision (simplified).

    Args:
        mesh: Input mesh
        iterations: Number of subdivision iterations

    Returns:
        Subdivided mesh
    """
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    for _ in range(iterations):
        edge_midpoints = {}
        new_faces = []

        for face in faces:
            v0, v1, v2 = face

            # Create or get edge midpoints
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            midpoint_indices = []

            for e in edges:
                edge_key = tuple(sorted(e))
                if edge_key not in edge_midpoints:
                    # Create new midpoint vertex
                    p0 = np.array(vertices[e[0]])
                    p1 = np.array(vertices[e[1]])
                    midpoint = ((p0 + p1) / 2).tolist()
                    edge_midpoints[edge_key] = len(vertices)
                    vertices.append(midpoint)
                midpoint_indices.append(edge_midpoints[edge_key])

            m01, m12, m20 = midpoint_indices

            # Create 4 new triangles
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])

        faces = new_faces

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_geodesic_sphere(
    radius: float = 1.0,
    subdivisions: int = 3,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a geodesic sphere by subdividing an icosahedron.

    Args:
        radius: Sphere radius
        subdivisions: Number of subdivision iterations
        center: Center position

    Returns:
        Geodesic sphere Mesh
    """
    # Start with icosahedron at origin
    mesh = create_icosahedron(radius=1.0, center=(0, 0, 0))

    # Subdivide
    mesh = subdivide_mesh(mesh, subdivisions)

    # Project vertices onto sphere and scale
    vertices = mesh.vertices
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms * radius

    # Translate to center
    cx, cy, cz = center
    vertices[:, 0] += cx
    vertices[:, 1] += cy
    vertices[:, 2] += cz

    return Mesh(vertices=vertices, faces=mesh.faces)


# Factory for advanced surfaces
ADVANCED_SURFACE_CREATORS = {
    "klein_bottle": create_klein_bottle,
    "mobius_strip": create_mobius_strip,
    "superquadric": create_superquadric,
    "trefoil_knot": create_trefoil_knot,
    "seashell": create_seashell,
    "boy_surface": create_boy_surface,
    "enneper": create_enneper_surface,
    "dini": create_dini_surface,
    "roman_surface": create_roman_surface,
    "catenoid": create_catenoid,
    "helicoid": create_helicoid,
    "spring": create_spring,
    "icosahedron": create_icosahedron,
    "dodecahedron": create_dodecahedron,
    "octahedron": create_octahedron,
    "tetrahedron": create_tetrahedron,
    "geodesic_sphere": create_geodesic_sphere,
}


def create_advanced_surface(name: str, **kwargs) -> Mesh:
    """Create an advanced surface by name."""
    if name not in ADVANCED_SURFACE_CREATORS:
        raise ValueError(
            f"Unknown surface: {name}. Available: {list(ADVANCED_SURFACE_CREATORS.keys())}"
        )
    return ADVANCED_SURFACE_CREATORS[name](**kwargs)
