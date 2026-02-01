"""
High-detail mesh generators for realistic 3D parts.
These create smooth, detailed geometry instead of blocky primitives.
"""

import numpy as np
from typing import Tuple, List, Optional

from src.geometry.mesh import Mesh


def create_smooth_cylinder(
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 32,
    rings: int = 1,
    center: tuple = (0, 0, 0),
    capped: bool = True,
) -> Mesh:
    """Create a high-detail smooth cylinder."""
    cx, cy, cz = center
    h = height / 2
    vertices = []

    # Side vertices with multiple rings for smoothness
    ring_count = max(2, rings + 2)
    for ring in range(ring_count):
        y = -h + (ring / (ring_count - 1)) * height + cy
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta) + cx
            z = radius * np.sin(theta) + cz
            vertices.append([x, y, z])

    faces = []
    # Side faces
    for ring in range(ring_count - 1):
        for i in range(segments):
            curr = ring * segments + i
            next_i = ring * segments + (i + 1) % segments
            curr_up = (ring + 1) * segments + i
            next_up = (ring + 1) * segments + (i + 1) % segments

            faces.append([curr, curr_up, next_i])
            faces.append([next_i, curr_up, next_up])

    if capped:
        # Bottom cap
        bottom_center = len(vertices)
        vertices.append([cx, -h + cy, cz])
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta) + cx
            z = radius * np.sin(theta) + cz
            vertices.append([x, -h + cy, z])

        for i in range(segments):
            curr = bottom_center + 1 + i
            next_i = bottom_center + 1 + (i + 1) % segments
            faces.append([bottom_center, next_i, curr])

        # Top cap
        top_center = len(vertices)
        vertices.append([cx, h + cy, cz])
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta) + cx
            z = radius * np.sin(theta) + cz
            vertices.append([x, h + cy, z])

        for i in range(segments):
            curr = top_center + 1 + i
            next_i = top_center + 1 + (i + 1) % segments
            faces.append([top_center, curr, next_i])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_turbine_blade(
    length: float = 1.0,
    chord: float = 0.3,
    thickness: float = 0.05,
    twist: float = 0.5,
    taper: float = 0.4,
    segments_span: int = 20,
    segments_chord: int = 16,
) -> Mesh:
    """
    Create a realistic turbine/propeller blade with airfoil profile.

    Args:
        length: Blade length (span)
        chord: Chord length at root
        thickness: Maximum thickness ratio
        twist: Total twist angle in radians
        taper: Taper ratio (tip chord / root chord)
        segments_span: Segments along span
        segments_chord: Segments around chord

    Returns:
        Detailed blade Mesh
    """
    vertices = []

    def naca_profile(x_c: float, t: float = 0.12) -> float:
        """NACA 4-digit thickness distribution."""
        return 5 * t * (
            0.2969 * np.sqrt(x_c + 0.001) -
            0.1260 * x_c -
            0.3516 * x_c**2 +
            0.2843 * x_c**3 -
            0.1015 * x_c**4
        )

    for i in range(segments_span + 1):
        span_t = i / segments_span
        y = span_t * length

        # Local chord with taper
        local_chord = chord * (1 - span_t * (1 - taper))

        # Local twist
        local_twist = twist * span_t

        for j in range(segments_chord):
            # Position around airfoil (0 to 1 for upper, 1 to 2 for lower)
            t = j / segments_chord * 2
            if t <= 1:
                x_c = 1 - t  # Leading edge to trailing edge (upper)
                z_sign = 1
            else:
                x_c = t - 1  # Trailing edge to leading edge (lower)
                z_sign = -1

            # Airfoil coordinates
            x_local = (x_c - 0.5) * local_chord
            z_local = z_sign * naca_profile(x_c, thickness) * local_chord

            # Apply twist
            x_twisted = x_local * np.cos(local_twist) - z_local * np.sin(local_twist)
            z_twisted = x_local * np.sin(local_twist) + z_local * np.cos(local_twist)

            vertices.append([x_twisted, y, z_twisted])

    # Generate faces
    faces = []
    for i in range(segments_span):
        for j in range(segments_chord):
            curr = i * segments_chord + j
            next_j = i * segments_chord + (j + 1) % segments_chord
            curr_up = (i + 1) * segments_chord + j
            next_up = (i + 1) * segments_chord + (j + 1) % segments_chord

            faces.append([curr, curr_up, next_j])
            faces.append([next_j, curr_up, next_up])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_gear(
    outer_radius: float = 1.0,
    inner_radius: float = 0.3,
    thickness: float = 0.2,
    num_teeth: int = 20,
    tooth_height: float = 0.15,
    tooth_width_ratio: float = 0.4,
    segments: int = 4,
) -> Mesh:
    """
    Create a detailed gear with proper tooth profile.

    Args:
        outer_radius: Radius to tooth tips
        inner_radius: Bore radius
        thickness: Gear thickness
        num_teeth: Number of teeth
        tooth_height: Height of teeth
        tooth_width_ratio: Tooth width as ratio of spacing
        segments: Segments per tooth

    Returns:
        Detailed gear Mesh
    """
    vertices = []
    h = thickness / 2

    root_radius = outer_radius - tooth_height

    # Generate gear profile points
    profile_points = []
    points_per_tooth = segments * 4

    for tooth in range(num_teeth):
        base_angle = 2 * np.pi * tooth / num_teeth
        tooth_angle = 2 * np.pi / num_teeth

        for seg in range(points_per_tooth):
            t = seg / points_per_tooth
            angle = base_angle + t * tooth_angle

            # Tooth profile (trapezoidal approximation)
            if t < 0.2:
                # Rising edge
                r = root_radius + (tooth_height * t / 0.2)
            elif t < 0.3:
                # Top flat
                r = outer_radius
            elif t < 0.5:
                # Falling edge
                r = outer_radius - (tooth_height * (t - 0.3) / 0.2)
            else:
                # Root
                r = root_radius

            profile_points.append((angle, r))

    # Create top and bottom faces
    for side in [-1, 1]:
        y = side * h
        center_idx = len(vertices)
        vertices.append([0, y, 0])

        # Inner ring
        inner_start = len(vertices)
        for angle, _ in profile_points:
            x = inner_radius * np.cos(angle)
            z = inner_radius * np.sin(angle)
            vertices.append([x, y, z])

        # Outer profile
        outer_start = len(vertices)
        for angle, r in profile_points:
            x = r * np.cos(angle)
            z = r * np.sin(angle)
            vertices.append([x, y, z])

    # Generate faces
    faces = []
    n_profile = len(profile_points)

    # Top face (side = 1, indices in first half)
    top_center = 0
    top_inner_start = 1
    top_outer_start = 1 + n_profile

    # Bottom face (side = -1, indices in second half)
    bottom_center = 1 + 2 * n_profile
    bottom_inner_start = bottom_center + 1
    bottom_outer_start = bottom_inner_start + n_profile

    # Top face triangles
    for i in range(n_profile):
        next_i = (i + 1) % n_profile

        # Inner ring to center
        faces.append([top_center, top_inner_start + next_i, top_inner_start + i])

        # Inner to outer ring
        faces.append([top_inner_start + i, top_inner_start + next_i, top_outer_start + i])
        faces.append([top_outer_start + i, top_inner_start + next_i, top_outer_start + next_i])

    # Bottom face triangles (reversed winding)
    for i in range(n_profile):
        next_i = (i + 1) % n_profile

        faces.append([bottom_center, bottom_inner_start + i, bottom_inner_start + next_i])

        faces.append([bottom_inner_start + i, bottom_outer_start + i, bottom_inner_start + next_i])
        faces.append([bottom_outer_start + i, bottom_outer_start + next_i, bottom_inner_start + next_i])

    # Side faces (outer profile)
    for i in range(n_profile):
        next_i = (i + 1) % n_profile

        top_curr = top_outer_start + i
        top_next = top_outer_start + next_i
        bot_curr = bottom_outer_start + i
        bot_next = bottom_outer_start + next_i

        faces.append([top_curr, bot_curr, top_next])
        faces.append([top_next, bot_curr, bot_next])

    # Inner bore faces
    for i in range(n_profile):
        next_i = (i + 1) % n_profile

        top_curr = top_inner_start + i
        top_next = top_inner_start + next_i
        bot_curr = bottom_inner_start + i
        bot_next = bottom_inner_start + next_i

        faces.append([top_curr, top_next, bot_curr])
        faces.append([top_next, bot_next, bot_curr])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_piston(
    diameter: float = 1.0,
    height: float = 1.2,
    skirt_height: float = 0.6,
    crown_shape: str = "flat",
    ring_grooves: int = 3,
    segments: int = 32,
) -> Mesh:
    """
    Create a detailed piston with ring grooves and crown.

    Args:
        diameter: Piston diameter
        height: Total height
        skirt_height: Height of skirt section
        crown_shape: "flat", "dome", or "dish"
        ring_grooves: Number of ring grooves
        segments: Circumference segments

    Returns:
        Detailed piston Mesh
    """
    radius = diameter / 2
    vertices = []

    # Crown profile
    crown_height = height - skirt_height
    groove_depth = 0.03 * diameter
    groove_height = 0.05 * diameter
    groove_spacing = crown_height * 0.7 / max(1, ring_grooves)

    # Build profile from bottom to top
    profile = []

    # Skirt bottom
    profile.append((radius, 0))

    # Skirt (slight taper)
    profile.append((radius * 0.98, skirt_height * 0.3))
    profile.append((radius, skirt_height * 0.7))
    profile.append((radius, skirt_height))

    # Ring grooves
    for i in range(ring_grooves):
        y_base = skirt_height + i * groove_spacing + 0.02
        profile.append((radius, y_base))
        profile.append((radius - groove_depth, y_base + 0.01))
        profile.append((radius - groove_depth, y_base + groove_height - 0.01))
        profile.append((radius, y_base + groove_height))

    # Crown land
    profile.append((radius, height - 0.05))

    # Crown edge
    profile.append((radius * 0.95, height))

    # Create vertices from profile
    for r, y in profile:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, y, z])

    # Crown surface
    if crown_shape == "dome":
        # Domed crown
        for ring in range(5):
            t = ring / 4
            r = radius * 0.95 * (1 - t)
            y = height + 0.1 * diameter * np.sin(t * np.pi / 2)

            if ring == 4:
                vertices.append([0, y, 0])
            else:
                for i in range(segments):
                    theta = 2 * np.pi * i / segments
                    x = r * np.cos(theta)
                    z = r * np.sin(theta)
                    vertices.append([x, y, z])

    elif crown_shape == "dish":
        # Dished crown
        for ring in range(5):
            t = ring / 4
            r = radius * 0.9 * (1 - t)
            y = height - 0.08 * diameter * np.sin(t * np.pi / 2)

            if ring == 4:
                vertices.append([0, y, 0])
            else:
                for i in range(segments):
                    theta = 2 * np.pi * i / segments
                    x = r * np.cos(theta)
                    z = r * np.sin(theta)
                    vertices.append([x, y, z])
    else:
        # Flat crown
        vertices.append([0, height, 0])

    # Bottom cap center
    bottom_center_idx = len(vertices)
    vertices.append([0, 0, 0])

    # Generate faces
    faces = []
    n_profile = len(profile)

    # Side faces
    for layer in range(n_profile - 1):
        for i in range(segments):
            curr = layer * segments + i
            next_i = layer * segments + (i + 1) % segments
            curr_up = (layer + 1) * segments + i
            next_up = (layer + 1) * segments + (i + 1) % segments

            faces.append([curr, curr_up, next_i])
            faces.append([next_i, curr_up, next_up])

    # Bottom cap
    for i in range(segments):
        curr = i
        next_i = (i + 1) % segments
        faces.append([bottom_center_idx, next_i, curr])

    # Crown faces (simplified)
    crown_start = (n_profile - 1) * segments
    if crown_shape in ["dome", "dish"]:
        # Multiple rings for domed/dished
        for ring in range(4):
            ring_start = crown_start + ring * segments
            if ring < 3:
                next_ring = ring_start + segments
                for i in range(segments):
                    curr = ring_start + i
                    next_i = ring_start + (i + 1) % segments
                    curr_up = next_ring + i
                    next_up = next_ring + (i + 1) % segments

                    faces.append([curr, next_i, curr_up])
                    faces.append([next_i, next_up, curr_up])
            else:
                # Connect to center
                center = crown_start + 4 * segments
                for i in range(segments):
                    curr = ring_start + i
                    next_i = ring_start + (i + 1) % segments
                    faces.append([curr, next_i, center])
    else:
        # Flat crown
        crown_center = n_profile * segments
        crown_edge_start = (n_profile - 1) * segments
        for i in range(segments):
            curr = crown_edge_start + i
            next_i = crown_edge_start + (i + 1) % segments
            faces.append([crown_center, curr, next_i])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_screw_thread(
    outer_radius: float = 0.1,
    inner_radius: float = 0.08,
    length: float = 1.0,
    pitch: float = 0.1,
    segments: int = 24,
) -> Mesh:
    """Create a screw with helical threads."""
    vertices = []

    num_turns = length / pitch
    points_per_turn = segments
    total_points = int(num_turns * points_per_turn) + 1

    # Thread profile (triangular)
    thread_height = (outer_radius - inner_radius)

    for i in range(total_points):
        t = i / points_per_turn
        y = t * pitch
        theta = 2 * np.pi * (i % points_per_turn) / points_per_turn

        # Outer thread point
        x_out = outer_radius * np.cos(theta)
        z_out = outer_radius * np.sin(theta)
        vertices.append([x_out, y, z_out])

        # Inner thread point (offset by half pitch)
        y_inner = y + pitch / 2
        x_in = inner_radius * np.cos(theta)
        z_in = inner_radius * np.sin(theta)
        vertices.append([x_in, y_inner, z_in])

    # Generate faces
    faces = []
    for i in range(total_points - 1):
        curr_out = i * 2
        curr_in = i * 2 + 1
        next_out = (i + 1) * 2
        next_in = (i + 1) * 2 + 1

        # Thread surface
        faces.append([curr_out, next_out, curr_in])
        faces.append([curr_in, next_out, next_in])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_bearing(
    outer_radius: float = 0.5,
    inner_radius: float = 0.2,
    width: float = 0.15,
    num_balls: int = 8,
    ball_radius: float = 0.08,
    segments: int = 24,
) -> Mesh:
    """Create a ball bearing with visible balls."""
    vertices = []
    faces = []

    # Outer race
    h = width / 2
    race_thickness = 0.03

    # Outer race profile
    for side in [-1, 1]:
        for ring in range(2):
            y = side * h
            r = outer_radius if ring == 0 else outer_radius - race_thickness

            for i in range(segments):
                theta = 2 * np.pi * i / segments
                x = r * np.cos(theta)
                z = r * np.sin(theta)
                vertices.append([x, y, z])

    n_seg = segments
    # Outer race faces
    for side_idx in range(2):
        base = side_idx * 2 * n_seg
        for i in range(n_seg):
            next_i = (i + 1) % n_seg

            # Top/bottom face
            outer_curr = base + i
            outer_next = base + next_i
            inner_curr = base + n_seg + i
            inner_next = base + n_seg + next_i

            if side_idx == 0:
                faces.append([outer_curr, inner_curr, outer_next])
                faces.append([outer_next, inner_curr, inner_next])
            else:
                faces.append([outer_curr, outer_next, inner_curr])
                faces.append([outer_next, inner_next, inner_curr])

    # Side faces
    for i in range(n_seg):
        next_i = (i + 1) % n_seg

        # Outer surface
        top_out = i
        bot_out = 2 * n_seg + i
        faces.append([top_out, bot_out, (i + 1) % n_seg])
        faces.append([(i + 1) % n_seg, bot_out, 2 * n_seg + (i + 1) % n_seg])

    # Inner race
    inner_base = len(vertices)
    for side in [-1, 1]:
        for ring in range(2):
            y = side * h
            r = inner_radius if ring == 1 else inner_radius + race_thickness

            for i in range(segments):
                theta = 2 * np.pi * i / segments
                x = r * np.cos(theta)
                z = r * np.sin(theta)
                vertices.append([x, y, z])

    # Inner race faces (similar to outer, reversed)
    for side_idx in range(2):
        base = inner_base + side_idx * 2 * n_seg
        for i in range(n_seg):
            next_i = (i + 1) % n_seg

            outer_curr = base + i
            outer_next = base + next_i
            inner_curr = base + n_seg + i
            inner_next = base + n_seg + next_i

            if side_idx == 1:
                faces.append([outer_curr, inner_curr, outer_next])
                faces.append([outer_next, inner_curr, inner_next])
            else:
                faces.append([outer_curr, outer_next, inner_curr])
                faces.append([outer_next, inner_next, inner_curr])

    # Balls
    ball_center_radius = (outer_radius - race_thickness + inner_radius + race_thickness) / 2
    ball_segments = 12
    ball_rings = 8

    for ball_idx in range(num_balls):
        angle = 2 * np.pi * ball_idx / num_balls

        cx = ball_center_radius * np.cos(angle)
        cz = ball_center_radius * np.sin(angle)
        cy = 0

        ball_base = len(vertices)

        # Generate ball vertices
        for ring in range(ball_rings + 1):
            phi = np.pi * ring / ball_rings
            for seg in range(ball_segments):
                theta = 2 * np.pi * seg / ball_segments

                x = cx + ball_radius * np.sin(phi) * np.cos(theta)
                y = cy + ball_radius * np.cos(phi)
                z = cz + ball_radius * np.sin(phi) * np.sin(theta)

                vertices.append([x, y, z])

        # Generate ball faces
        for ring in range(ball_rings):
            for seg in range(ball_segments):
                curr = ball_base + ring * ball_segments + seg
                next_seg = ball_base + ring * ball_segments + (seg + 1) % ball_segments
                curr_up = ball_base + (ring + 1) * ball_segments + seg
                next_up = ball_base + (ring + 1) * ball_segments + (seg + 1) % ball_segments

                if ring != 0:
                    faces.append([curr, curr_up, next_seg])
                if ring != ball_rings - 1:
                    faces.append([next_seg, curr_up, next_up])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_spring_coil(
    coil_radius: float = 0.3,
    wire_radius: float = 0.03,
    pitch: float = 0.08,
    num_coils: float = 6,
    segments_coil: int = 60,
    segments_wire: int = 12,
) -> Mesh:
    """Create a detailed coil spring."""
    vertices = []

    total_angle = num_coils * 2 * np.pi
    total_points = int(num_coils * segments_coil)

    for i in range(total_points + 1):
        t = i / segments_coil
        theta = 2 * np.pi * t

        # Helix center point
        hx = coil_radius * np.cos(theta)
        hz = coil_radius * np.sin(theta)
        hy = t * pitch

        # Tangent direction
        tx = -coil_radius * np.sin(theta)
        tz = coil_radius * np.cos(theta)
        ty = pitch / (2 * np.pi)

        t_len = np.sqrt(tx*tx + ty*ty + tz*tz)
        tx, ty, tz = tx/t_len, ty/t_len, tz/t_len

        # Normal (toward helix axis)
        nx, ny, nz = -np.cos(theta), 0, -np.sin(theta)

        # Binormal
        bx = ty * nz - tz * ny
        by = tz * nx - tx * nz
        bz = tx * ny - ty * nx

        # Wire cross-section
        for j in range(segments_wire):
            phi = 2 * np.pi * j / segments_wire

            x = hx + wire_radius * (np.cos(phi) * nx + np.sin(phi) * bx)
            y = hy + wire_radius * (np.cos(phi) * ny + np.sin(phi) * by)
            z = hz + wire_radius * (np.cos(phi) * nz + np.sin(phi) * bz)

            vertices.append([x, y, z])

    # Generate faces
    faces = []
    for i in range(total_points):
        for j in range(segments_wire):
            curr = i * segments_wire + j
            next_j = i * segments_wire + (j + 1) % segments_wire
            curr_up = (i + 1) * segments_wire + j
            next_up = (i + 1) * segments_wire + (j + 1) % segments_wire

            faces.append([curr, curr_up, next_j])
            faces.append([next_j, curr_up, next_up])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_hex_bolt(
    head_radius: float = 0.15,
    head_height: float = 0.1,
    shaft_radius: float = 0.08,
    shaft_length: float = 0.5,
    thread_length: float = 0.3,
    segments: int = 24,
) -> Mesh:
    """Create a hex bolt with head and threaded shaft."""
    vertices = []
    faces = []

    # Hex head
    h = head_height
    for side in [-1, 1]:
        y = (side + 1) / 2 * h  # 0 or h

        # Hex points
        center_idx = len(vertices)
        vertices.append([0, y, 0])

        for i in range(6):
            angle = np.pi / 6 + i * np.pi / 3
            x = head_radius * np.cos(angle)
            z = head_radius * np.sin(angle)
            vertices.append([x, y, z])

    # Hex faces
    for i in range(6):
        next_i = (i % 6) + 1
        next_next = ((i + 1) % 6) + 1

        # Top
        faces.append([0, i + 1, next_next if next_i == 6 else next_i + 1])
        # Bottom
        faces.append([7, 7 + next_next if next_i == 6 else 7 + next_i + 1, 7 + i + 1])

    # Hex side faces
    for i in range(6):
        next_i = (i % 6) + 1

        top_curr = i + 1
        top_next = next_i + 1 if next_i < 6 else 1
        bot_curr = 7 + i + 1
        bot_next = 7 + next_i + 1 if next_i < 6 else 8

        faces.append([top_curr, top_next, bot_curr])
        faces.append([top_next, bot_next, bot_curr])

    # Shaft
    shaft_base = len(vertices)
    for ring in range(3):  # Top, middle, thread start
        if ring == 0:
            y = head_height
        elif ring == 1:
            y = head_height + (shaft_length - thread_length)
        else:
            y = head_height + shaft_length

        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = shaft_radius * np.cos(theta)
            z = shaft_radius * np.sin(theta)
            vertices.append([x, y, z])

    # Shaft faces
    for ring in range(2):
        for i in range(segments):
            curr = shaft_base + ring * segments + i
            next_i = shaft_base + ring * segments + (i + 1) % segments
            curr_up = shaft_base + (ring + 1) * segments + i
            next_up = shaft_base + (ring + 1) * segments + (i + 1) % segments

            faces.append([curr, curr_up, next_i])
            faces.append([next_i, curr_up, next_up])

    # Bottom cap
    bottom_center = len(vertices)
    vertices.append([0, head_height + shaft_length, 0])

    bottom_ring = shaft_base + 2 * segments
    for i in range(segments):
        curr = bottom_ring + i
        next_i = bottom_ring + (i + 1) % segments
        faces.append([bottom_center, curr, next_i])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


# Factory
DETAILED_PART_CREATORS = {
    "smooth_cylinder": create_smooth_cylinder,
    "turbine_blade": create_turbine_blade,
    "gear": create_gear,
    "piston": create_piston,
    "screw_thread": create_screw_thread,
    "bearing": create_bearing,
    "spring_coil": create_spring_coil,
    "hex_bolt": create_hex_bolt,
}


def create_detailed_part(name: str, **kwargs) -> Mesh:
    """Create a detailed part by name."""
    if name not in DETAILED_PART_CREATORS:
        raise ValueError(f"Unknown part: {name}. Available: {list(DETAILED_PART_CREATORS.keys())}")
    return DETAILED_PART_CREATORS[name](**kwargs)
