"""
Scientific visualization models inspired by PyVista.
DNA, molecules, atomic orbitals, crystal structures, field visualizations.
"""

import numpy as np
from typing import List, Tuple, Optional

from src.geometry.mesh import Mesh
from src.geometry.primitives import create_sphere, create_cylinder
from src.geometry.transforms import (
    translation_matrix,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    scale_matrix,
)


def merge_meshes(meshes: List[Mesh]) -> Mesh:
    """
    Merge multiple meshes into a single mesh.

    Args:
        meshes: List of meshes to merge

    Returns:
        Combined Mesh
    """
    if not meshes:
        raise ValueError("No meshes to merge")

    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)

        # Offset face indices
        faces = mesh.faces + vertex_offset
        all_faces.append(faces)

        if mesh.colors is not None:
            all_colors.append(mesh.colors)
        else:
            # Default to white if no colors
            all_colors.append(np.full((len(mesh.vertices), 3), 200, dtype=np.uint8))

        vertex_offset += len(mesh.vertices)

    return Mesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        colors=np.vstack(all_colors) if all_colors else None,
    )


def create_dna_helix(
    radius: float = 0.5,
    pitch: float = 0.7,
    turns: float = 3.0,
    base_pairs: int = 30,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a detailed DNA double helix.

    Args:
        radius: Helix radius
        pitch: Height per turn
        turns: Number of complete turns
        base_pairs: Number of base pairs
        center: Center position

    Returns:
        DNA helix Mesh
    """
    cx, cy, cz = center
    meshes = []

    total_height = turns * pitch
    backbone_radius = 0.04
    base_radius = 0.03

    for strand in [0, 1]:
        # Each strand is offset by 180 degrees
        phase = strand * np.pi

        for i in range(base_pairs * 3):  # More points for smooth backbone
            t = i / (base_pairs * 3 - 1)
            theta = turns * 2 * np.pi * t + phase
            y = t * total_height - total_height / 2 + cy

            # Backbone position
            x = radius * np.cos(theta) + cx
            z = radius * np.sin(theta) + cz

            # Backbone sphere
            backbone = create_sphere(radius=backbone_radius, segments=8, rings=6)
            backbone = backbone.transform(translation_matrix(x, y, z))
            if strand == 0:
                backbone.set_color((70, 130, 180))  # Steel blue
            else:
                backbone.set_color((178, 102, 255))  # Purple
            meshes.append(backbone)

    # Base pairs connecting the strands
    for i in range(base_pairs):
        t = (i + 0.5) / base_pairs
        theta = turns * 2 * np.pi * t
        y = t * total_height - total_height / 2 + cy

        # Positions on both strands
        x1 = radius * np.cos(theta) + cx
        z1 = radius * np.sin(theta) + cz
        x2 = radius * np.cos(theta + np.pi) + cx
        z2 = radius * np.sin(theta + np.pi) + cz

        # Create base pair (two half-rungs with different colors)
        mid_x = (x1 + x2) / 2
        mid_z = (z1 + z2) / 2

        # First half (A-T or G-C)
        length1 = np.sqrt((mid_x - x1)**2 + (mid_z - z1)**2)
        angle1 = np.arctan2(mid_z - z1, mid_x - x1)

        base1 = create_cylinder(radius=base_radius, height=length1, segments=8)
        base1 = base1.transform(rotation_matrix_x(np.pi / 2))
        base1 = base1.transform(rotation_matrix_y(-angle1))
        base1 = base1.transform(translation_matrix((x1 + mid_x) / 2, y, (z1 + mid_z) / 2))
        base1.set_color((255, 100, 100) if i % 2 == 0 else (100, 255, 100))  # Red or Green
        meshes.append(base1)

        # Second half
        length2 = np.sqrt((x2 - mid_x)**2 + (z2 - mid_z)**2)
        angle2 = np.arctan2(z2 - mid_z, x2 - mid_x)

        base2 = create_cylinder(radius=base_radius, height=length2, segments=8)
        base2 = base2.transform(rotation_matrix_x(np.pi / 2))
        base2 = base2.transform(rotation_matrix_y(-angle2))
        base2 = base2.transform(translation_matrix((mid_x + x2) / 2, y, (mid_z + z2) / 2))
        base2.set_color((255, 255, 100) if i % 2 == 0 else (100, 200, 255))  # Yellow or Cyan
        meshes.append(base2)

    return merge_meshes(meshes)


def create_molecule(
    molecule_type: str = "caffeine",
    scale: float = 1.0,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a ball-and-stick molecular model.

    Args:
        molecule_type: Type of molecule (water, methane, benzene, caffeine, buckyball)
        scale: Overall scale
        center: Center position

    Returns:
        Molecule Mesh
    """
    cx, cy, cz = center

    # Atom definitions with positions and types
    molecules = {
        "water": {
            "atoms": [
                ("O", (0, 0, 0), 0.35),
                ("H", (0.76, 0.59, 0), 0.25),
                ("H", (-0.76, 0.59, 0), 0.25),
            ],
            "bonds": [(0, 1), (0, 2)],
        },
        "methane": {
            "atoms": [
                ("C", (0, 0, 0), 0.35),
                ("H", (0.63, 0.63, 0.63), 0.25),
                ("H", (-0.63, -0.63, 0.63), 0.25),
                ("H", (-0.63, 0.63, -0.63), 0.25),
                ("H", (0.63, -0.63, -0.63), 0.25),
            ],
            "bonds": [(0, 1), (0, 2), (0, 3), (0, 4)],
        },
        "benzene": {
            "atoms": [
                ("C", (1.4, 0, 0), 0.3),
                ("C", (0.7, 1.21, 0), 0.3),
                ("C", (-0.7, 1.21, 0), 0.3),
                ("C", (-1.4, 0, 0), 0.3),
                ("C", (-0.7, -1.21, 0), 0.3),
                ("C", (0.7, -1.21, 0), 0.3),
                ("H", (2.48, 0, 0), 0.2),
                ("H", (1.24, 2.15, 0), 0.2),
                ("H", (-1.24, 2.15, 0), 0.2),
                ("H", (-2.48, 0, 0), 0.2),
                ("H", (-1.24, -2.15, 0), 0.2),
                ("H", (1.24, -2.15, 0), 0.2),
            ],
            "bonds": [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Ring
                (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),  # H bonds
            ],
        },
        "caffeine": {
            "atoms": [
                # Purine ring system
                ("N", (0, 0, 0), 0.28),
                ("C", (1.2, 0.4, 0), 0.3),
                ("N", (2.1, -0.5, 0), 0.28),
                ("C", (1.8, -1.8, 0), 0.3),
                ("C", (0.4, -2.1, 0), 0.3),
                ("C", (-0.3, -0.9, 0), 0.3),
                ("N", (-1.6, -0.8, 0), 0.28),
                ("C", (-2.0, 0.5, 0), 0.3),
                ("N", (-1.0, 1.3, 0), 0.28),
                # Methyl groups
                ("C", (0.3, 1.4, 0), 0.3),
                ("C", (3.5, -0.3, 0), 0.3),
                ("C", (-2.3, -1.9, 0), 0.3),
                # Oxygen
                ("O", (1.5, 1.6, 0), 0.28),
                ("O", (2.6, -2.6, 0), 0.28),
            ],
            "bonds": [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                (5, 6), (6, 7), (7, 8), (8, 0),
                (0, 9), (2, 10), (6, 11),
                (1, 12), (3, 13),
            ],
        },
        "buckyball": _generate_buckyball_structure(),
    }

    if molecule_type not in molecules:
        molecule_type = "benzene"

    mol_data = molecules[molecule_type]
    meshes = []

    # Atom colors
    atom_colors = {
        "H": (255, 255, 255),  # White
        "C": (80, 80, 80),     # Dark gray
        "N": (50, 50, 200),    # Blue
        "O": (200, 50, 50),    # Red
        "S": (255, 200, 50),   # Yellow
        "P": (255, 150, 0),    # Orange
    }

    # Create atoms
    atom_positions = []
    for atom_type, pos, radius in mol_data["atoms"]:
        x, y, z = pos
        atom = create_sphere(
            radius=radius * scale * 0.2,
            segments=16,
            rings=12,
        )
        atom = atom.transform(translation_matrix(
            x * scale * 0.3 + cx,
            y * scale * 0.3 + cy,
            z * scale * 0.3 + cz
        ))
        color = atom_colors.get(atom_type, (150, 150, 150))
        atom.set_color(color)
        meshes.append(atom)
        atom_positions.append((x * scale * 0.3 + cx, y * scale * 0.3 + cy, z * scale * 0.3 + cz))

    # Create bonds
    for i1, i2 in mol_data["bonds"]:
        p1 = np.array(atom_positions[i1])
        p2 = np.array(atom_positions[i2])

        # Bond direction and length
        direction = p2 - p1
        length = np.linalg.norm(direction)
        midpoint = (p1 + p2) / 2

        if length < 0.001:
            continue

        # Calculate rotation angles
        direction_norm = direction / length
        pitch = np.arcsin(-direction_norm[1])
        yaw = np.arctan2(direction_norm[0], direction_norm[2])

        bond = create_cylinder(radius=0.02 * scale, height=length, segments=8)
        bond = bond.transform(rotation_matrix_x(pitch))
        bond = bond.transform(rotation_matrix_y(yaw))
        bond = bond.transform(translation_matrix(midpoint[0], midpoint[1], midpoint[2]))
        bond.set_color((180, 180, 180))
        meshes.append(bond)

    return merge_meshes(meshes)


def _generate_buckyball_structure():
    """Generate C60 buckyball structure."""
    # Buckyball (C60) vertices using truncated icosahedron coordinates
    phi = (1 + np.sqrt(5)) / 2

    vertices = []

    # Generate vertices from cyclic permutations
    coords = [
        (0, 1, 3 * phi),
        (1, 2 + phi, 2 * phi),
        (phi, 2, 2 * phi + 1),
    ]

    for c in coords:
        for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
            x, y, z = c[0] * signs[0], c[1] * signs[1], c[2] * signs[2]
            # Cyclic permutations
            for perm in [(x, y, z), (y, z, x), (z, x, y)]:
                if perm not in vertices:
                    vertices.append(perm)

    # Remove duplicates and create atoms
    unique_verts = list(set(vertices))[:60]  # Should be exactly 60

    atoms = [("C", v, 0.25) for v in unique_verts]

    # Generate bonds (connect vertices within bond distance)
    bonds = []
    for i in range(len(unique_verts)):
        for j in range(i + 1, len(unique_verts)):
            p1 = np.array(unique_verts[i])
            p2 = np.array(unique_verts[j])
            dist = np.linalg.norm(p2 - p1)
            if 1.8 < dist < 2.5:  # Bond distance threshold
                bonds.append((i, j))

    return {"atoms": atoms, "bonds": bonds}


def create_crystal_lattice(
    lattice_type: str = "fcc",
    size: int = 3,
    atom_radius: float = 0.15,
    spacing: float = 0.5,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a crystal lattice structure.

    Args:
        lattice_type: Type of lattice (sc, bcc, fcc, diamond, hcp)
        size: Number of unit cells in each direction
        atom_radius: Radius of atoms
        spacing: Unit cell spacing
        center: Center position

    Returns:
        Crystal lattice Mesh
    """
    cx, cy, cz = center
    meshes = []

    # Unit cell atom positions (relative)
    lattices = {
        "sc": [(0, 0, 0)],  # Simple cubic
        "bcc": [(0, 0, 0), (0.5, 0.5, 0.5)],  # Body-centered cubic
        "fcc": [  # Face-centered cubic
            (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        ],
        "diamond": [  # Diamond structure
            (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
            (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
        ],
        "hcp": [  # Hexagonal close-packed
            (0, 0, 0), (0.5, 0.5 * np.sqrt(3) / 3, 0),
            (0.25, 0.5 * np.sqrt(3) / 6, 0.5),
        ],
    }

    positions = lattices.get(lattice_type, lattices["fcc"])

    # Calculate offset to center the lattice
    offset = -spacing * (size - 1) / 2

    # Generate lattice
    atom_positions = set()

    for i in range(size):
        for j in range(size):
            for k in range(size):
                for px, py, pz in positions:
                    x = (i + px) * spacing + offset
                    y = (j + py) * spacing + offset
                    z = (k + pz) * spacing + offset
                    # Round to avoid floating point duplicates
                    pos = (round(x * 1000), round(y * 1000), round(z * 1000))
                    if pos not in atom_positions:
                        atom_positions.add(pos)

                        atom = create_sphere(radius=atom_radius, segments=12, rings=8)
                        atom = atom.transform(translation_matrix(
                            x + cx, y + cy, z + cz
                        ))

                        # Color based on position in unit cell
                        if (px, py, pz) == (0, 0, 0):
                            atom.set_color((100, 150, 200))
                        elif px == 0.5 and py == 0.5 and pz == 0.5:
                            atom.set_color((200, 100, 100))
                        else:
                            atom.set_color((150, 200, 100))

                        meshes.append(atom)

    return merge_meshes(meshes)


def create_atomic_orbital(
    orbital_type: str = "p",
    n: int = 2,
    l: int = 1,
    m: int = 0,
    resolution: int = 30,
    scale: float = 1.0,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a visualization of atomic orbital shapes.

    Args:
        orbital_type: s, p, d, f, or sp3 (hybrid)
        n: Principal quantum number
        l: Angular momentum quantum number
        m: Magnetic quantum number
        resolution: Surface resolution
        scale: Overall scale
        center: Center position

    Returns:
        Atomic orbital Mesh
    """
    cx, cy, cz = center
    meshes = []

    if orbital_type == "s":
        # S orbital - sphere
        orbital = create_sphere(radius=0.4 * scale, segments=24, rings=16)
        orbital = orbital.transform(translation_matrix(cx, cy, cz))
        orbital.set_color((100, 150, 255))
        return orbital

    elif orbital_type == "p":
        # P orbital - two lobes (dumbbell shape)
        for sign in [1, -1]:
            lobe = _create_orbital_lobe(scale * 0.5, resolution)
            lobe = lobe.transform(scale_matrix(0.6, 1.0, 0.6))
            lobe = lobe.transform(translation_matrix(cx, cy + sign * 0.3 * scale, cz))
            lobe.set_color((255, 100, 100) if sign > 0 else (100, 100, 255))
            meshes.append(lobe)

    elif orbital_type == "d":
        # D orbital - four lobes
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            lobe = _create_orbital_lobe(scale * 0.35, resolution)
            lobe = lobe.transform(rotation_matrix_z(angle))
            lobe = lobe.transform(translation_matrix(
                cx + 0.3 * scale * np.cos(angle),
                cy,
                cz + 0.3 * scale * np.sin(angle)
            ))
            lobe.set_color((255, 200, 100) if angle < np.pi else (100, 200, 255))
            meshes.append(lobe)

    elif orbital_type == "sp3":
        # SP3 hybrid - tetrahedral arrangement
        angles = [
            (np.arctan(np.sqrt(2)), 0),
            (np.arctan(np.sqrt(2)), 2 * np.pi / 3),
            (np.arctan(np.sqrt(2)), 4 * np.pi / 3),
            (np.pi, 0),
        ]
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]

        for (theta, phi), color in zip(angles, colors):
            lobe = _create_orbital_lobe(scale * 0.4, resolution)

            # Direction
            dx = np.sin(theta) * np.cos(phi)
            dy = np.cos(theta)
            dz = np.sin(theta) * np.sin(phi)

            # Rotate to align
            lobe = lobe.transform(rotation_matrix_x(theta))
            lobe = lobe.transform(rotation_matrix_y(phi))
            lobe = lobe.transform(translation_matrix(
                cx + dx * 0.3 * scale,
                cy + dy * 0.3 * scale,
                cz + dz * 0.3 * scale
            ))
            lobe.set_color(color)
            meshes.append(lobe)

    elif orbital_type == "f":
        # F orbital - complex multi-lobe
        for i in range(8):
            theta = np.pi / 4 if i < 4 else 3 * np.pi / 4
            phi = i * np.pi / 2 if i < 4 else (i - 4) * np.pi / 2 + np.pi / 4

            lobe = _create_orbital_lobe(scale * 0.3, resolution)

            dx = np.sin(theta) * np.cos(phi)
            dy = np.cos(theta)
            dz = np.sin(theta) * np.sin(phi)

            lobe = lobe.transform(translation_matrix(
                cx + dx * 0.35 * scale,
                cy + dy * 0.35 * scale,
                cz + dz * 0.35 * scale
            ))
            lobe.set_color((200, 100, 255) if i < 4 else (100, 255, 200))
            meshes.append(lobe)

    # Add nucleus
    nucleus = create_sphere(radius=0.08 * scale, segments=12, rings=8)
    nucleus = nucleus.transform(translation_matrix(cx, cy, cz))
    nucleus.set_color((255, 255, 0))
    meshes.append(nucleus)

    return merge_meshes(meshes)


def _create_orbital_lobe(scale: float, resolution: int) -> Mesh:
    """Create a single orbital lobe (teardrop shape)."""
    vertices = []

    for i in range(resolution):
        u = np.pi * i / (resolution - 1)  # 0 to pi

        # Teardrop profile
        r = scale * np.sin(u) * (1 + 0.5 * np.cos(u))

        for j in range(resolution):
            v = 2 * np.pi * j / resolution

            x = r * np.cos(v)
            y = scale * (np.cos(u) + 0.3)  # Offset center
            z = r * np.sin(v)

            vertices.append([x, y, z])

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution):
            curr = i * resolution + j
            next_j = i * resolution + (j + 1) % resolution
            next_i = (i + 1) * resolution + j
            next_both = (i + 1) * resolution + (j + 1) % resolution

            if i != 0:
                faces.append([curr, next_i, next_j])
            if i != resolution - 2:
                faces.append([next_j, next_i, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_magnetic_field(
    num_lines: int = 12,
    strength: float = 1.0,
    turns: float = 1.5,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create magnetic field lines visualization (dipole field).

    Args:
        num_lines: Number of field lines
        strength: Field strength (affects size)
        turns: Number of turns for each line
        center: Center position

    Returns:
        Field lines Mesh
    """
    cx, cy, cz = center
    meshes = []

    tube_radius = 0.02 * strength

    for i in range(num_lines):
        phi = 2 * np.pi * i / num_lines

        # Create field line as parametric curve
        vertices = []
        num_points = 50

        for j in range(num_points + 1):
            t = np.pi * j / num_points  # 0 to pi

            # Dipole field line equation: r = sin^2(theta)
            r = strength * np.sin(t) ** 2

            x = r * np.sin(t) * np.cos(phi) + cx
            y = r * np.cos(t) + cy
            z = r * np.sin(t) * np.sin(phi) + cz

            vertices.append([x, y, z])

        # Create tube along the curve
        meshes.append(_create_tube_along_curve(vertices, tube_radius))

    # Create central magnet (north and south poles)
    north = create_sphere(radius=0.15 * strength, segments=12, rings=8)
    north = north.transform(translation_matrix(cx, cy + 0.1, cz))
    north.set_color((255, 50, 50))
    meshes.append(north)

    south = create_sphere(radius=0.15 * strength, segments=12, rings=8)
    south = south.transform(translation_matrix(cx, cy - 0.1, cz))
    south.set_color((50, 50, 255))
    meshes.append(south)

    return merge_meshes(meshes)


def _create_tube_along_curve(
    points: List,
    radius: float,
    segments: int = 8,
) -> Mesh:
    """Create a tube mesh along a curve defined by points."""
    if len(points) < 2:
        return create_sphere(radius=radius)

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

        # Create circle at this point
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

    mesh = Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )
    mesh.set_color((100, 200, 255))
    return mesh


def create_virus_model(
    radius: float = 1.0,
    spike_count: int = 60,
    spike_length: float = 0.3,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a coronavirus-like virus model.

    Args:
        radius: Capsid radius
        spike_count: Number of spike proteins
        spike_length: Length of spikes
        center: Center position

    Returns:
        Virus Mesh
    """
    cx, cy, cz = center
    meshes = []

    # Viral capsid (bumpy sphere)
    capsid = create_sphere(radius=radius, segments=32, rings=24)
    capsid = capsid.transform(translation_matrix(cx, cy, cz))

    # Add surface texture (modify vertices for bumpy effect)
    vertices = capsid.vertices.copy()
    for i in range(len(vertices)):
        v = vertices[i] - np.array([cx, cy, cz])
        noise = 0.05 * radius * np.sin(20 * v[0]) * np.cos(20 * v[1]) * np.sin(20 * v[2])
        norm = np.linalg.norm(v)
        if norm > 0:
            vertices[i] = (v / norm * (norm + noise)) + np.array([cx, cy, cz])

    capsid.vertices = vertices
    capsid.set_color((200, 150, 150))
    meshes.append(capsid)

    # Generate spike positions (golden spiral distribution)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    for i in range(spike_count):
        y = 1 - (i / (spike_count - 1)) * 2  # -1 to 1
        rad = np.sqrt(1 - y * y)
        theta = 2 * np.pi * i / phi

        # Spike direction
        dx = rad * np.cos(theta)
        dy = y
        dz = rad * np.sin(theta)

        # Spike position (on surface)
        sx = dx * radius + cx
        sy = dy * radius + cy
        sz = dz * radius + cz

        # Create spike (crown-shaped spike protein)
        spike = _create_spike_protein(spike_length * radius, 0.08 * radius)

        # Rotate spike to point outward
        pitch = -np.arcsin(dy)
        yaw = np.arctan2(dx, dz)

        spike = spike.transform(rotation_matrix_x(pitch))
        spike = spike.transform(rotation_matrix_y(yaw))
        spike = spike.transform(translation_matrix(
            sx + dx * spike_length * radius * 0.5,
            sy + dy * spike_length * radius * 0.5,
            sz + dz * spike_length * radius * 0.5
        ))

        spike.set_color((255, 100, 100))
        meshes.append(spike)

    return merge_meshes(meshes)


def _create_spike_protein(length: float, radius: float) -> Mesh:
    """Create a single spike protein (club-shaped)."""
    meshes = []

    # Stem
    stem = create_cylinder(radius=radius * 0.5, height=length * 0.7, segments=8)
    meshes.append(stem)

    # Crown (receptor binding domain)
    crown = create_sphere(radius=radius * 1.5, segments=10, rings=8)
    crown = crown.transform(scale_matrix(1.0, 0.6, 1.0))
    crown = crown.transform(translation_matrix(0, length * 0.5, 0))
    meshes.append(crown)

    return merge_meshes(meshes)


def create_protein_structure(
    structure_type: str = "helix",
    length: int = 20,
    scale: float = 1.0,
    center: tuple = (0, 0, 0),
) -> Mesh:
    """
    Create a simplified protein secondary structure visualization.

    Args:
        structure_type: helix (alpha helix) or sheet (beta sheet)
        length: Number of residues
        scale: Overall scale
        center: Center position

    Returns:
        Protein structure Mesh
    """
    cx, cy, cz = center
    meshes = []

    if structure_type == "helix":
        # Alpha helix - ribbon representation
        helix_radius = 0.25 * scale
        pitch = 0.15 * scale
        residue_angle = 100 * np.pi / 180  # 100 degrees per residue

        # Create ribbon along helix
        points = []
        for i in range(length * 3):
            t = i / 3
            theta = t * residue_angle
            y = t * pitch

            x = helix_radius * np.cos(theta) + cx
            z = helix_radius * np.sin(theta) + cz

            points.append([x, y + cy - length * pitch / 6, z])

        # Create ribbon mesh
        ribbon = _create_ribbon(points, 0.15 * scale, 0.03 * scale)
        ribbon.set_color((255, 100, 150))
        meshes.append(ribbon)

        # Add backbone atoms
        for i in range(length):
            t = i
            theta = t * residue_angle
            y = t * pitch

            x = helix_radius * np.cos(theta) + cx
            z = helix_radius * np.sin(theta) + cz

            atom = create_sphere(radius=0.05 * scale, segments=8, rings=6)
            atom = atom.transform(translation_matrix(x, y + cy - length * pitch / 6, z))
            atom.set_color((100, 100, 255))
            meshes.append(atom)

    else:  # Beta sheet
        # Create pleated sheet
        sheet_width = length * 0.1 * scale
        sheet_height = 0.5 * scale
        pleat_depth = 0.05 * scale

        for strand in range(3):
            strand_offset = (strand - 1) * sheet_height * 0.4

            for i in range(length):
                x = (i - length / 2) * 0.1 * scale + cx
                y = pleat_depth * ((-1) ** i) + strand_offset + cy
                z = cz

                atom = create_sphere(radius=0.04 * scale, segments=8, rings=6)
                atom = atom.transform(translation_matrix(x, y, z))
                atom.set_color((100, 200, 100) if strand == 1 else (255, 200, 100))
                meshes.append(atom)

                # Connect to next
                if i < length - 1:
                    next_x = x + 0.1 * scale
                    next_y = pleat_depth * ((-1) ** (i + 1)) + strand_offset + cy

                    bond = create_cylinder(
                        radius=0.015 * scale,
                        height=np.sqrt((0.1 * scale) ** 2 + (next_y - y) ** 2),
                        segments=6
                    )
                    angle = np.arctan2(next_y - y, 0.1 * scale)
                    bond = bond.transform(rotation_matrix_z(np.pi / 2 - angle))
                    bond = bond.transform(translation_matrix((x + next_x) / 2, (y + next_y) / 2, z))
                    bond.set_color((180, 180, 180))
                    meshes.append(bond)

    return merge_meshes(meshes)


def _create_ribbon(
    points: List,
    width: float,
    thickness: float,
) -> Mesh:
    """Create a ribbon mesh along a curve."""
    points = np.array(points, dtype=np.float32)
    vertices = []
    faces = []

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
        normal = np.cross(tangent, up)
        norm_len = np.linalg.norm(normal)
        if norm_len < 0.1:
            up = np.array([1, 0, 0])
            normal = np.cross(tangent, up)

        normal = normal / (np.linalg.norm(normal) + 1e-8)
        binormal = np.cross(tangent, normal)

        # Create ribbon cross-section (4 vertices)
        hw = width / 2
        ht = thickness / 2

        vertices.append(p + hw * normal + ht * binormal)
        vertices.append(p + hw * normal - ht * binormal)
        vertices.append(p - hw * normal - ht * binormal)
        vertices.append(p - hw * normal + ht * binormal)

    # Generate faces
    for i in range(len(points) - 1):
        base = i * 4
        for j in range(4):
            curr = base + j
            next_j = base + (j + 1) % 4
            curr_next = base + 4 + j
            next_both = base + 4 + (j + 1) % 4

            faces.append([curr, curr_next, next_j])
            faces.append([next_j, curr_next, next_both])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


# Factory for scientific models
SCIENTIFIC_MODEL_CREATORS = {
    "dna_helix": create_dna_helix,
    "molecule": create_molecule,
    "crystal_lattice": create_crystal_lattice,
    "atomic_orbital": create_atomic_orbital,
    "magnetic_field": create_magnetic_field,
    "virus": create_virus_model,
    "protein": create_protein_structure,
}


def create_scientific_model(name: str, **kwargs) -> Mesh:
    """Create a scientific model by name."""
    if name not in SCIENTIFIC_MODEL_CREATORS:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(SCIENTIFIC_MODEL_CREATORS.keys())}"
        )
    return SCIENTIFIC_MODEL_CREATORS[name](**kwargs)
