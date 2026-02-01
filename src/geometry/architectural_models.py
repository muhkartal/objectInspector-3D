"""
Architectural and engineering structural models.
Bridges, space stations, buildings, and terrain.
"""

import numpy as np
from typing import Tuple, List

from src.geometry.mesh import Mesh
from src.geometry.assembly import Assembly, AssemblyPart
from src.geometry.primitives import (
    create_cylinder,
    create_cube,
    create_sphere,
    create_torus,
    create_cone,
)
from src.geometry.transforms import (
    translation_matrix,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    scale_matrix,
)
from config import settings


def get_color(index: int) -> Tuple[int, int, int]:
    """Get a color from the assembly color palette."""
    colors = settings.ASSEMBLY_COLORS
    return colors[index % len(colors)]


def merge_meshes(meshes: List[Mesh]) -> Mesh:
    """Merge multiple meshes into one."""
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        faces = mesh.faces + vertex_offset
        all_faces.append(faces)
        vertex_offset += len(mesh.vertices)

    return Mesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
    )


def create_suspension_bridge_assembly() -> Assembly:
    """
    Create a detailed suspension bridge assembly (~50 parts).

    Parts include:
    - Main towers (2)
    - Tower bases
    - Main cables
    - Suspender cables
    - Deck sections
    - Anchorages
    - Tower cross-beams
    """
    assembly = Assembly("Suspension Bridge")

    bridge_length = 3.0
    deck_width = 0.4
    tower_height = 0.8
    cable_sag = 0.3

    # === TOWERS ===
    for tower_side in [-1, 1]:
        tower_x = tower_side * bridge_length * 0.3

        # Tower legs
        for leg_side in [-1, 1]:
            leg = create_cube(size=0.08)
            leg = leg.transform(scale_matrix(1.0, tower_height / 0.08, 1.0))
            leg = leg.transform(translation_matrix(
                tower_x,
                tower_height / 2 - 0.1,
                leg_side * deck_width * 0.4
            ))

            assembly.add_part(AssemblyPart(
                mesh=leg,
                name=f"Tower {'A' if tower_side < 0 else 'B'} Leg {'L' if leg_side < 0 else 'R'}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([tower_side * 0.3, 0.5, leg_side * 0.3]),
                explosion_distance=0.4,
                color=(100, 100, 110),
            ))

        # Tower cross-beams
        beam_heights = [0.2, 0.5, tower_height - 0.05]
        for i, beam_y in enumerate(beam_heights):
            beam = create_cylinder(radius=0.02, height=deck_width * 0.7, segments=12)
            beam = beam.transform(rotation_matrix_x(np.pi / 2))
            beam = beam.transform(translation_matrix(tower_x, beam_y, 0))

            assembly.add_part(AssemblyPart(
                mesh=beam,
                name=f"Tower {'A' if tower_side < 0 else 'B'} Beam {i+1}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([tower_side * 0.2, 0.3, 0.0]),
                explosion_distance=0.25 + i * 0.05,
                color=(120, 120, 130),
            ))

        # Tower top saddle
        saddle = create_cylinder(radius=0.04, height=deck_width * 0.5, segments=12)
        saddle = saddle.transform(rotation_matrix_x(np.pi / 2))
        saddle = saddle.transform(translation_matrix(tower_x, tower_height, 0))

        assembly.add_part(AssemblyPart(
            mesh=saddle,
            name=f"Tower {'A' if tower_side < 0 else 'B'} Saddle",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([tower_side * 0.2, 0.8, 0.0]),
            explosion_distance=0.35,
            color=get_color(2),
        ))

    # === MAIN CABLES ===
    cable_points = 40

    for cable_side in [-1, 1]:
        vertices = []

        for i in range(cable_points + 1):
            t = i / cable_points
            x = -bridge_length / 2 + t * bridge_length

            # Parabolic shape for main cable
            # Lowest at center, highest at towers
            tower_x = bridge_length * 0.3
            if abs(x) < tower_x:
                # Between towers - parabolic
                y = tower_height - cable_sag * (1 - (x / tower_x) ** 2)
            else:
                # Outside towers - straight down to anchorage
                outside_t = (abs(x) - tower_x) / (bridge_length / 2 - tower_x)
                y = tower_height - outside_t * (tower_height - 0.1)

            z = cable_side * deck_width * 0.3
            vertices.append([x, y, z])

        # Create tube along cable
        cable_mesh = _create_cable_tube(vertices, 0.015)

        assembly.add_part(AssemblyPart(
            mesh=cable_mesh,
            name=f"Main Cable {'L' if cable_side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, 0.6, cable_side * 0.4]),
            explosion_distance=0.35,
            color=(80, 80, 90),
        ))

    # === DECK SECTIONS ===
    num_deck_sections = 16
    section_length = bridge_length / num_deck_sections

    for i in range(num_deck_sections):
        x = -bridge_length / 2 + (i + 0.5) * section_length

        deck = create_cube(size=section_length * 0.95)
        deck = deck.transform(scale_matrix(1.0, 0.08 / section_length, deck_width / section_length))
        deck = deck.transform(translation_matrix(x, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=deck,
            name=f"Deck Section {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, -0.5, 0.0]),
            explosion_distance=0.3,
            color=(140, 140, 150),
        ))

    # === SUSPENDER CABLES ===
    num_suspenders = 24

    for i in range(num_suspenders):
        x = -bridge_length * 0.28 + (i / (num_suspenders - 1)) * bridge_length * 0.56

        # Calculate main cable height at this x
        tower_x = bridge_length * 0.3
        cable_y = tower_height - cable_sag * (1 - (x / tower_x) ** 2)

        for side in [-1, 1]:
            suspender_height = cable_y
            suspender = create_cylinder(radius=0.005, height=suspender_height, segments=6)
            suspender = suspender.transform(translation_matrix(x, suspender_height / 2, side * deck_width * 0.3))

            assembly.add_part(AssemblyPart(
                mesh=suspender,
                name=f"Suspender {i+1}{'L' if side < 0 else 'R'}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([0.0, 0.3, side * 0.2]),
                explosion_distance=0.15,
                color=(100, 100, 110),
            ))

    # === ANCHORAGES ===
    for anchor_side in [-1, 1]:
        anchor = create_cube(size=0.2)
        anchor = anchor.transform(scale_matrix(1.2, 0.8, 1.5))
        anchor = anchor.transform(translation_matrix(anchor_side * bridge_length * 0.48, -0.02, 0))

        assembly.add_part(AssemblyPart(
            mesh=anchor,
            name=f"Anchorage {'A' if anchor_side < 0 else 'B'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([anchor_side * 0.8, -0.3, 0.0]),
            explosion_distance=0.4,
            color=(80, 80, 90),
        ))

    return assembly


def _create_cable_tube(points: List, radius: float, segments: int = 8) -> Mesh:
    """Create a tube mesh along a series of points."""
    points = np.array(points, dtype=np.float32)
    vertices = []

    for i in range(len(points)):
        p = points[i]

        # Calculate tangent
        if i == 0:
            tangent = points[1] - points[0]
        elif i == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]

        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

        # Create orthogonal vectors
        up = np.array([0, 1, 0])
        if abs(np.dot(tangent, up)) > 0.9:
            up = np.array([1, 0, 0])

        normal = np.cross(tangent, up)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        binormal = np.cross(tangent, normal)

        # Create circle
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            offset = radius * (np.cos(theta) * normal + np.sin(theta) * binormal)
            vertices.append(p + offset)

    # Generate faces
    faces = []
    for i in range(len(points) - 1):
        for j in range(segments):
            curr = i * segments + j
            next_j = i * segments + (j + 1) % segments
            next_i = (i + 1) * segments + j
            next_both = (i + 1) * segments + (j + 1) % segments

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_space_station_assembly() -> Assembly:
    """
    Create a modular space station assembly (~45 parts).

    Parts include:
    - Central hub
    - Laboratory modules
    - Living quarters
    - Solar array wings
    - Docking ports
    - Truss segments
    - Communication arrays
    - Thermal radiators
    """
    assembly = Assembly("Space Station")

    # === CENTRAL HUB ===
    hub = create_cylinder(radius=0.2, height=0.4, segments=24)
    hub = hub.transform(rotation_matrix_z(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=hub,
        name="Central Hub",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=(200, 200, 210),
    ))

    # === LABORATORY MODULES ===
    module_configs = [
        ("Lab Module A", (0.4, 0, 0), (1, 0, 0)),
        ("Lab Module B", (-0.4, 0, 0), (-1, 0, 0)),
        ("Lab Module C", (0, 0, 0.35), (0, 0, 1)),
        ("Lab Module D", (0, 0, -0.35), (0, 0, -1)),
    ]

    for name, pos, direction in module_configs:
        module = create_cylinder(radius=0.12, height=0.35, segments=20)

        # Orient based on direction
        if direction[0] != 0:
            module = module.transform(rotation_matrix_z(np.pi / 2 * direction[0]))
        elif direction[2] != 0:
            module = module.transform(rotation_matrix_x(np.pi / 2 * direction[2]))

        module = module.transform(translation_matrix(*pos))

        assembly.add_part(AssemblyPart(
            mesh=module,
            name=name,
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array(direction),
            explosion_distance=0.4,
            color=(180, 180, 190),
        ))

        # Module docking ring
        ring = create_torus(major_radius=0.11, minor_radius=0.015, major_segments=20, minor_segments=10)

        ring_pos = np.array(pos) + np.array(direction) * 0.18

        if direction[0] != 0:
            ring = ring.transform(rotation_matrix_y(np.pi / 2))
        elif direction[2] != 0:
            ring = ring.transform(rotation_matrix_x(np.pi / 2))

        ring = ring.transform(translation_matrix(*ring_pos))

        assembly.add_part(AssemblyPart(
            mesh=ring,
            name=f"{name} Ring",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array(direction) * 1.2,
            explosion_distance=0.35,
            color=(150, 150, 160),
        ))

    # === TRUSS STRUCTURE ===
    truss_length = 1.2

    # Main truss beam
    truss = create_cube(size=truss_length)
    truss = truss.transform(scale_matrix(1.0, 0.06 / truss_length, 0.06 / truss_length))
    truss = truss.transform(translation_matrix(0, 0.25, 0))

    assembly.add_part(AssemblyPart(
        mesh=truss,
        name="Main Truss",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.6, 0.0]),
        explosion_distance=0.3,
        color=(100, 100, 110),
    ))

    # Truss diagonal bracing
    for side in [-1, 1]:
        for i in range(4):
            x = -truss_length / 2 + 0.15 + i * 0.3

            brace = create_cylinder(radius=0.008, height=0.15, segments=6)
            brace = brace.transform(rotation_matrix_z(side * np.pi / 4))
            brace = brace.transform(translation_matrix(x, 0.25, 0))

            assembly.add_part(AssemblyPart(
                mesh=brace,
                name=f"Truss Brace {i+1}{'A' if side < 0 else 'B'}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([0.0, 0.4, side * 0.2]),
                explosion_distance=0.2,
                color=(80, 80, 90),
            ))

    # === SOLAR ARRAYS ===
    for wing in [-1, 1]:
        wing_x = wing * truss_length * 0.45

        # Solar panel mounting
        mount = create_cylinder(radius=0.03, height=0.08, segments=10)
        mount = mount.transform(rotation_matrix_x(np.pi / 2))
        mount = mount.transform(translation_matrix(wing_x, 0.25, 0))

        assembly.add_part(AssemblyPart(
            mesh=mount,
            name=f"Solar Mount {'L' if wing < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([wing * 0.3, 0.3, 0.0]),
            explosion_distance=0.2,
            color=(100, 100, 110),
        ))

        # Solar panels (4 per wing)
        for panel_idx in range(4):
            panel = create_cube(size=0.2)
            panel = panel.transform(scale_matrix(1.0, 0.02, 2.5))

            pz = (panel_idx - 1.5) * 0.22

            panel = panel.transform(translation_matrix(wing_x, 0.25, pz))

            assembly.add_part(AssemblyPart(
                mesh=panel,
                name=f"Solar Panel {'L' if wing < 0 else 'R'}{panel_idx + 1}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([wing * 0.5, 0.2, 0.0]),
                explosion_distance=0.3 + panel_idx * 0.08,
                color=(30, 30, 60),
            ))

    # === DOCKING PORTS ===
    dock_positions = [
        (0.0, 0.22, 0.0),  # Top
        (0.0, -0.22, 0.0),  # Bottom (nadir)
    ]

    for i, (dx, dy, dz) in enumerate(dock_positions):
        dock = create_cylinder(radius=0.08, height=0.08, segments=16)
        dock = dock.transform(translation_matrix(dx, dy, dz))

        assembly.add_part(AssemblyPart(
            mesh=dock,
            name=f"Docking Port {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, np.sign(dy), 0.0]),
            explosion_distance=0.25,
            color=get_color(3),
        ))

    # === COMMUNICATION ARRAYS ===
    comm_positions = [(0.3, 0.18, 0.2), (-0.3, 0.18, -0.2)]

    for i, (cx, cy, cz) in enumerate(comm_positions):
        # Dish
        dish = create_cone(radius=0.06, height=0.03, segments=16)
        dish = dish.transform(rotation_matrix_x(-np.pi / 2))
        dish = dish.transform(translation_matrix(cx, cy, cz))

        assembly.add_part(AssemblyPart(
            mesh=dish,
            name=f"Comm Dish {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([cx, 0.5, cz]),
            explosion_distance=0.25,
            color=(220, 220, 230),
        ))

        # Antenna boom
        boom = create_cylinder(radius=0.008, height=0.12, segments=6)
        boom = boom.transform(translation_matrix(cx, cy - 0.08, cz))

        assembly.add_part(AssemblyPart(
            mesh=boom,
            name=f"Comm Boom {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([cx * 0.5, 0.3, cz * 0.5]),
            explosion_distance=0.2,
            color=(80, 80, 90),
        ))

    # === THERMAL RADIATORS ===
    for side in [-1, 1]:
        radiator = create_cube(size=0.3)
        radiator = radiator.transform(scale_matrix(0.05, 1.0, 0.6))
        radiator = radiator.transform(translation_matrix(side * 0.2, 0.25, 0))

        assembly.add_part(AssemblyPart(
            mesh=radiator,
            name=f"Radiator {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 0.5, 0.3, 0.0]),
            explosion_distance=0.25,
            color=(240, 240, 250),
        ))

    return assembly


def create_wind_turbine_assembly() -> Assembly:
    """
    Create a wind turbine assembly (~20 parts).
    """
    assembly = Assembly("Wind Turbine")

    # Tower
    tower_height = 1.5
    tower = create_cone(radius=0.08, height=tower_height, segments=20)
    tower = tower.transform(scale_matrix(1.0, 1.0, 1.0))
    tower = tower.transform(translation_matrix(0, tower_height / 2 - 0.5, 0))

    # Modify to be a tapered cylinder (frustum)
    verts = tower.vertices.copy()
    for i in range(len(verts)):
        y_ratio = (verts[i, 1] + 0.5) / tower_height
        verts[i, 0] *= (0.4 + 0.6 * (1 - y_ratio))
        verts[i, 2] *= (0.4 + 0.6 * (1 - y_ratio))

    tower = Mesh(vertices=verts, faces=tower.faces)

    assembly.add_part(AssemblyPart(
        mesh=tower,
        name="Tower",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.3, 0.0]),
        explosion_distance=0.3,
        color=(200, 200, 210),
    ))

    # Nacelle
    nacelle = create_cube(size=0.2)
    nacelle = nacelle.transform(scale_matrix(1.5, 0.6, 0.8))
    nacelle = nacelle.transform(translation_matrix(0, tower_height - 0.45, 0))

    assembly.add_part(AssemblyPart(
        mesh=nacelle,
        name="Nacelle",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.5, 0.0]),
        explosion_distance=0.35,
        color=(220, 220, 230),
    ))

    # Hub
    hub = create_sphere(radius=0.08, segments=16, rings=12)
    hub = hub.transform(translation_matrix(0.18, tower_height - 0.45, 0))

    assembly.add_part(AssemblyPart(
        mesh=hub,
        name="Hub",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.5, 0.3, 0.0]),
        explosion_distance=0.3,
        color=(180, 180, 190),
    ))

    # Blades
    blade_length = 0.7
    num_blades = 3

    for i in range(num_blades):
        angle = i * 2 * np.pi / num_blades

        blade = create_cube(size=blade_length)
        blade = blade.transform(scale_matrix(0.03, 1.0, 0.08))

        # Position at hub
        blade = blade.transform(translation_matrix(0, blade_length / 2, 0))
        blade = blade.transform(rotation_matrix_x(angle))
        blade = blade.transform(translation_matrix(0.2, tower_height - 0.45, 0))

        assembly.add_part(AssemblyPart(
            mesh=blade,
            name=f"Blade {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                0.5,
                np.cos(angle) * 0.5,
                np.sin(angle) * 0.5
            ]),
            explosion_distance=0.4,
            color=(230, 230, 240),
        ))

    # Foundation
    foundation = create_cylinder(radius=0.15, height=0.08, segments=20)
    foundation = foundation.transform(translation_matrix(0, -0.5, 0))

    assembly.add_part(AssemblyPart(
        mesh=foundation,
        name="Foundation",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.3,
        color=(100, 100, 110),
    ))

    return assembly


def create_terrain_mesh(
    width: float = 2.0,
    depth: float = 2.0,
    resolution: int = 50,
    height_scale: float = 0.3,
    noise_octaves: int = 4,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a procedural terrain mesh with fractal noise.

    Args:
        width: Terrain width
        depth: Terrain depth
        resolution: Grid resolution
        height_scale: Maximum height variation
        noise_octaves: Fractal noise detail
        center: Center position

    Returns:
        Terrain Mesh with height-based coloring
    """
    cx, cy, cz = center

    # Generate height map using simplex-like noise
    def noise2d(x, y, octaves=4):
        """Multi-octave noise approximation."""
        value = 0
        amplitude = 1
        frequency = 1
        max_value = 0

        for _ in range(octaves):
            # Simplified noise using sine combinations
            value += amplitude * (
                np.sin(x * frequency * 3.14) * np.cos(y * frequency * 2.72) +
                np.sin(x * frequency * 1.41 + y * frequency * 1.73) * 0.5
            )
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2

        return value / max_value

    # Generate vertices
    vertices = []
    colors = []

    for i in range(resolution + 1):
        for j in range(resolution + 1):
            x = (i / resolution - 0.5) * width + cx
            z = (j / resolution - 0.5) * depth + cz

            # Calculate height
            h = noise2d(i / resolution * 4, j / resolution * 4, noise_octaves)
            h = (h + 1) * 0.5  # Normalize to 0-1
            y = h * height_scale + cy

            vertices.append([x, y, z])

            # Height-based coloring
            if h < 0.3:
                color = (50, 100, 50)  # Low - dark green (valley)
            elif h < 0.5:
                color = (80, 140, 60)  # Medium-low - green
            elif h < 0.7:
                color = (120, 100, 80)  # Medium - brown (hills)
            elif h < 0.85:
                color = (160, 140, 120)  # High - light brown (mountains)
            else:
                color = (220, 220, 230)  # Very high - snow

            colors.append(color)

    # Generate faces
    faces = []

    for i in range(resolution):
        for j in range(resolution):
            curr = i * (resolution + 1) + j
            next_j = curr + 1
            next_i = (i + 1) * (resolution + 1) + j
            next_both = next_i + 1

            faces.append([curr, next_i, next_j])
            faces.append([next_j, next_i, next_both])

    mesh = Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        colors=np.array(colors, dtype=np.uint8),
    )

    return mesh


def create_skyscraper(
    floors: int = 40,
    base_width: float = 0.4,
    base_depth: float = 0.3,
    floor_height: float = 0.025,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a detailed skyscraper model with setbacks.

    Args:
        floors: Number of floors
        base_width: Building width at base
        base_depth: Building depth at base
        floor_height: Height per floor
        center: Center position

    Returns:
        Skyscraper Mesh
    """
    cx, cy, cz = center
    meshes = []

    total_height = floors * floor_height
    setback_floors = [int(floors * 0.6), int(floors * 0.8)]

    current_width = base_width
    current_depth = base_depth

    for floor in range(floors):
        # Check for setbacks
        if floor == setback_floors[0]:
            current_width *= 0.8
            current_depth *= 0.8
        elif floor == setback_floors[1]:
            current_width *= 0.85
            current_depth *= 0.85

        y = floor * floor_height + floor_height / 2

        # Floor slab
        slab = create_cube(size=1.0)
        slab = slab.transform(scale_matrix(current_width, floor_height * 0.9, current_depth))
        slab = slab.transform(translation_matrix(cx, y + cy, cz))

        meshes.append(slab)

    # Spire on top
    spire = create_cone(radius=base_width * 0.15, height=total_height * 0.15, segments=12)
    spire = spire.transform(translation_matrix(cx, total_height + total_height * 0.075 + cy, cz))
    meshes.append(spire)

    mesh = merge_meshes(meshes)
    mesh.set_color((180, 190, 200))
    return mesh


def create_geodesic_dome(
    radius: float = 1.0,
    frequency: int = 3,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a geodesic dome (half sphere with triangular panels).

    Args:
        radius: Dome radius
        frequency: Subdivision frequency (higher = more detail)
        center: Center position

    Returns:
        Geodesic dome Mesh
    """
    cx, cy, cz = center

    # Start with icosahedron vertices (top half)
    phi = (1 + np.sqrt(5)) / 2

    ico_verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
    ]

    # Normalize to sphere
    ico_verts = [np.array(v) / np.linalg.norm(v) for v in ico_verts]

    # Icosahedron faces
    ico_faces = [
        (0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1),
        (1, 6, 8), (8, 6, 10), (8, 10, 4), (4, 10, 2), (4, 2, 5),
        (5, 2, 11), (5, 11, 9), (9, 11, 7), (9, 7, 1), (1, 7, 6),
        (3, 6, 7), (3, 10, 6), (3, 2, 10), (3, 11, 2), (3, 7, 11),
    ]

    # Subdivide faces
    def subdivide_face(v0, v1, v2, depth):
        if depth == 0:
            return [(v0, v1, v2)]

        # Find midpoints
        m01 = (v0 + v1) / 2
        m12 = (v1 + v2) / 2
        m20 = (v2 + v0) / 2

        # Normalize to sphere
        m01 = m01 / np.linalg.norm(m01)
        m12 = m12 / np.linalg.norm(m12)
        m20 = m20 / np.linalg.norm(m20)

        # Recurse
        faces = []
        faces.extend(subdivide_face(v0, m01, m20, depth - 1))
        faces.extend(subdivide_face(m01, v1, m12, depth - 1))
        faces.extend(subdivide_face(m20, m12, v2, depth - 1))
        faces.extend(subdivide_face(m01, m12, m20, depth - 1))
        return faces

    all_faces = []
    for f in ico_faces:
        v0 = np.array(ico_verts[f[0]])
        v1 = np.array(ico_verts[f[1]])
        v2 = np.array(ico_verts[f[2]])
        all_faces.extend(subdivide_face(v0, v1, v2, frequency))

    # Convert to mesh format (only top half - y > -0.1)
    vertex_map = {}
    vertices = []
    faces = []

    for face_verts in all_faces:
        # Check if face is mostly in upper hemisphere
        avg_y = sum(v[2] for v in face_verts) / 3  # z is up in ico coords
        if avg_y < -0.1:
            continue

        face_indices = []
        for v in face_verts:
            v_tuple = tuple(np.round(v, 6))
            if v_tuple not in vertex_map:
                vertex_map[v_tuple] = len(vertices)
                # Transform: ico z -> mesh y
                vertices.append([
                    v[0] * radius + cx,
                    v[2] * radius + cy,
                    v[1] * radius + cz
                ])
            face_indices.append(vertex_map[v_tuple])

        if len(face_indices) == 3:
            faces.append(face_indices)

    mesh = Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )
    mesh.set_color((150, 180, 200))
    return mesh


# Factory for architectural models
ARCHITECTURAL_ASSEMBLY_CREATORS = {
    "suspension_bridge": create_suspension_bridge_assembly,
    "space_station": create_space_station_assembly,
    "wind_turbine": create_wind_turbine_assembly,
}

ARCHITECTURAL_MESH_CREATORS = {
    "terrain": create_terrain_mesh,
    "skyscraper": create_skyscraper,
    "geodesic_dome": create_geodesic_dome,
}


def create_architectural_assembly(name: str) -> Assembly:
    """Create an architectural assembly by name."""
    if name not in ARCHITECTURAL_ASSEMBLY_CREATORS:
        raise ValueError(
            f"Unknown assembly: {name}. Available: {list(ARCHITECTURAL_ASSEMBLY_CREATORS.keys())}"
        )
    return ARCHITECTURAL_ASSEMBLY_CREATORS[name]()


def create_architectural_mesh(name: str, **kwargs) -> Mesh:
    """Create an architectural mesh by name."""
    if name not in ARCHITECTURAL_MESH_CREATORS:
        raise ValueError(
            f"Unknown mesh: {name}. Available: {list(ARCHITECTURAL_MESH_CREATORS.keys())}"
        )
    return ARCHITECTURAL_MESH_CREATORS[name](**kwargs)


def get_available_architectural_assemblies() -> list:
    """Get list of available architectural assembly names."""
    return list(ARCHITECTURAL_ASSEMBLY_CREATORS.keys())


def get_available_architectural_meshes() -> list:
    """Get list of available architectural mesh names."""
    return list(ARCHITECTURAL_MESH_CREATORS.keys())
