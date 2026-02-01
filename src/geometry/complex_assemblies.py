"""
Complex mechanical assemblies with many detailed parts.
Turbine engines, robotic arms, satellites, and more.
"""

import numpy as np
from typing import Tuple

from config import settings
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


def get_color(index: int) -> Tuple[int, int, int]:
    """Get a color from the assembly color palette."""
    colors = settings.ASSEMBLY_COLORS
    return colors[index % len(colors)]


def merge_meshes_simple(meshes: list) -> Mesh:
    """Simple mesh merge without color handling."""
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


def create_turbine_blade(length: float, chord: float, twist: float) -> Mesh:
    """Create a single turbine blade with airfoil profile and twist."""
    vertices = []
    segments = 20
    chord_points = 12

    for i in range(segments + 1):
        t = i / segments
        span_pos = t * length

        # Twist increases along span
        blade_twist = twist * t

        # Chord tapers toward tip
        local_chord = chord * (1 - 0.4 * t)

        for j in range(chord_points):
            # NACA-like airfoil profile
            x_c = j / (chord_points - 1)

            # Thickness distribution (NACA 4-digit style)
            thick = 0.15 * local_chord * (
                2.969 * np.sqrt(x_c + 0.001) -
                1.26 * x_c -
                3.516 * x_c**2 +
                2.843 * x_c**3 -
                1.015 * x_c**4
            )

            # Upper and lower surfaces
            for surface in [1, -1]:
                x = (x_c - 0.5) * local_chord
                y = span_pos
                z = surface * thick * 0.5

                # Apply twist
                x_rot = x * np.cos(blade_twist) - z * np.sin(blade_twist)
                z_rot = x * np.sin(blade_twist) + z * np.cos(blade_twist)

                vertices.append([x_rot, y, z_rot])

    # Generate faces
    faces = []
    points_per_section = chord_points * 2

    for i in range(segments):
        for j in range(chord_points - 1):
            # Upper surface
            for surface_offset in [0, chord_points]:
                idx1 = i * points_per_section + j * 2 + (0 if surface_offset == 0 else 1)
                idx2 = i * points_per_section + (j + 1) * 2 + (0 if surface_offset == 0 else 1)
                idx3 = (i + 1) * points_per_section + j * 2 + (0 if surface_offset == 0 else 1)
                idx4 = (i + 1) * points_per_section + (j + 1) * 2 + (0 if surface_offset == 0 else 1)

                if surface_offset == 0:
                    faces.append([idx1, idx3, idx2])
                    faces.append([idx2, idx3, idx4])
                else:
                    faces.append([idx1, idx2, idx3])
                    faces.append([idx2, idx4, idx3])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def create_jet_engine_assembly() -> Assembly:
    """
    Create a detailed jet turbine engine assembly (~40 parts).

    Parts include:
    - Nacelle (outer casing)
    - Fan blades (multiple)
    - Compressor stages
    - Combustion chamber
    - Turbine stages
    - Exhaust nozzle
    - Core shaft
    """
    assembly = Assembly("Jet Turbine Engine")

    # === NACELLE (outer casing) ===
    nacelle_outer = create_cylinder(radius=0.5, height=1.8, segments=32, capped=False)
    nacelle_outer = nacelle_outer.transform(rotation_matrix_z(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=nacelle_outer,
        name="Nacelle Outer",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.3,
        color=(180, 180, 190),
    ))

    # Inlet ring
    inlet = create_torus(major_radius=0.48, minor_radius=0.03, major_segments=32, minor_segments=12)
    inlet = inlet.transform(rotation_matrix_y(np.pi / 2))
    inlet = inlet.transform(translation_matrix(-0.9, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=inlet,
        name="Inlet Ring",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-1.0, 0.0, 0.0]),
        explosion_distance=0.4,
        color=(100, 100, 110),
    ))

    # === FAN SECTION ===
    # Fan hub
    fan_hub = create_cylinder(radius=0.15, height=0.15, segments=24)
    fan_hub = fan_hub.transform(rotation_matrix_z(np.pi / 2))
    fan_hub = fan_hub.transform(translation_matrix(-0.75, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=fan_hub,
        name="Fan Hub",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-1.0, 0.0, 0.0]),
        explosion_distance=0.6,
        color=get_color(2),
    ))

    # Fan blades (24 blades)
    num_fan_blades = 24
    for i in range(num_fan_blades):
        angle = 2 * np.pi * i / num_fan_blades

        # Create blade
        blade = create_cube(size=0.3)
        blade = blade.transform(scale_matrix(0.08, 1.0, 0.02))
        blade = blade.transform(rotation_matrix_x(0.3))  # Blade pitch
        blade = blade.transform(translation_matrix(0, 0.25, 0))
        blade = blade.transform(rotation_matrix_y(angle))
        blade = blade.transform(translation_matrix(-0.75, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=blade,
            name=f"Fan Blade {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                -0.5,
                np.cos(angle) * 0.8,
                np.sin(angle) * 0.8
            ]),
            explosion_distance=0.5,
            color=(200, 200, 210),
        ))

    # Fan case
    fan_case = create_cylinder(radius=0.46, height=0.2, segments=32, capped=False)
    fan_case = fan_case.transform(rotation_matrix_z(np.pi / 2))
    fan_case = fan_case.transform(translation_matrix(-0.75, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=fan_case,
        name="Fan Case",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-0.5, 0.5, 0.0]),
        explosion_distance=0.35,
        color=(150, 150, 160),
    ))

    # === COMPRESSOR STAGES ===
    compressor_positions = [-0.5, -0.35, -0.2, -0.05]

    for stage, x_pos in enumerate(compressor_positions):
        # Compressor disk
        disk_radius = 0.35 - stage * 0.03

        disk = create_cylinder(radius=disk_radius, height=0.04, segments=24)
        disk = disk.transform(rotation_matrix_z(np.pi / 2))
        disk = disk.transform(translation_matrix(x_pos, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=disk,
            name=f"Compressor Stage {stage+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([-0.3, 0.3 + stage * 0.15, 0.0]),
            explosion_distance=0.4 + stage * 0.1,
            color=get_color(0),
        ))

        # Compressor blades (simplified as ring)
        blade_ring = create_torus(
            major_radius=disk_radius - 0.05,
            minor_radius=0.015,
            major_segments=24,
            minor_segments=8
        )
        blade_ring = blade_ring.transform(rotation_matrix_y(np.pi / 2))
        blade_ring = blade_ring.transform(translation_matrix(x_pos + 0.03, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=blade_ring,
            name=f"Compressor Blades {stage+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([-0.2, 0.4 + stage * 0.1, 0.1]),
            explosion_distance=0.35 + stage * 0.08,
            color=(220, 220, 230),
        ))

    # === COMBUSTION CHAMBER ===
    combustor_outer = create_cylinder(radius=0.25, height=0.3, segments=24, capped=False)
    combustor_outer = combustor_outer.transform(rotation_matrix_z(np.pi / 2))
    combustor_outer = combustor_outer.transform(translation_matrix(0.15, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=combustor_outer,
        name="Combustion Chamber",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=get_color(1),
    ))

    # Fuel injectors
    for i in range(6):
        angle = 2 * np.pi * i / 6

        injector = create_cylinder(radius=0.02, height=0.08, segments=8)
        injector = injector.transform(translation_matrix(
            0.05,
            0.2 * np.cos(angle),
            0.2 * np.sin(angle)
        ))

        assembly.add_part(AssemblyPart(
            mesh=injector,
            name=f"Fuel Injector {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, np.cos(angle), np.sin(angle)]),
            explosion_distance=0.25,
            color=get_color(3),
        ))

    # === TURBINE STAGES ===
    turbine_positions = [0.35, 0.5, 0.65]

    for stage, x_pos in enumerate(turbine_positions):
        # Turbine disk
        disk = create_cylinder(radius=0.2 + stage * 0.02, height=0.03, segments=24)
        disk = disk.transform(rotation_matrix_z(np.pi / 2))
        disk = disk.transform(translation_matrix(x_pos, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=disk,
            name=f"Turbine Stage {stage+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.5, 0.3 - stage * 0.1, 0.0]),
            explosion_distance=0.4 + stage * 0.1,
            color=get_color(1),
        ))

    # === EXHAUST NOZZLE ===
    nozzle = create_cone(radius=0.35, height=0.25, segments=24)
    nozzle = nozzle.transform(rotation_matrix_z(-np.pi / 2))
    nozzle = nozzle.transform(translation_matrix(0.85, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=nozzle,
        name="Exhaust Nozzle",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([1.0, 0.0, 0.0]),
        explosion_distance=0.5,
        color=(120, 120, 130),
    ))

    # Core nozzle
    core_nozzle = create_cone(radius=0.15, height=0.2, segments=16)
    core_nozzle = core_nozzle.transform(rotation_matrix_z(-np.pi / 2))
    core_nozzle = core_nozzle.transform(translation_matrix(0.9, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=core_nozzle,
        name="Core Nozzle",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([1.0, 0.0, 0.0]),
        explosion_distance=0.6,
        color=get_color(2),
    ))

    # === CENTRAL SHAFT ===
    shaft = create_cylinder(radius=0.03, height=1.6, segments=12)
    shaft = shaft.transform(rotation_matrix_z(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=shaft,
        name="Core Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.4,
        color=(80, 80, 90),
    ))

    return assembly


def create_robotic_arm_assembly() -> Assembly:
    """
    Create a detailed industrial robotic arm assembly (~30 parts).

    Parts include:
    - Base with motor
    - Shoulder joint
    - Upper arm
    - Elbow joint
    - Forearm
    - Wrist assembly (3-axis)
    - End effector (gripper)
    """
    assembly = Assembly("Industrial Robotic Arm")

    # === BASE ===
    base_plate = create_cylinder(radius=0.4, height=0.08, segments=32)
    base_plate = base_plate.transform(translation_matrix(0, -0.6, 0))

    assembly.add_part(AssemblyPart(
        mesh=base_plate,
        name="Base Plate",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.3,
        color=(60, 60, 70),
    ))

    # Base motor housing
    base_housing = create_cylinder(radius=0.2, height=0.25, segments=24)
    base_housing = base_housing.transform(translation_matrix(0, -0.45, 0))

    assembly.add_part(AssemblyPart(
        mesh=base_housing,
        name="Base Housing",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.5, 0.0]),
        explosion_distance=0.2,
        color=get_color(0),
    ))

    # Base rotation ring
    base_ring = create_torus(major_radius=0.18, minor_radius=0.025, major_segments=24, minor_segments=12)
    base_ring = base_ring.transform(rotation_matrix_x(np.pi / 2))
    base_ring = base_ring.transform(translation_matrix(0, -0.3, 0))

    assembly.add_part(AssemblyPart(
        mesh=base_ring,
        name="Base Bearing",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.3, 0.3]),
        explosion_distance=0.15,
        color=(200, 180, 50),
    ))

    # === SHOULDER JOINT ===
    shoulder_housing = create_sphere(radius=0.15, segments=20, rings=16)
    shoulder_housing = shoulder_housing.transform(scale_matrix(1.2, 0.8, 1.0))
    shoulder_housing = shoulder_housing.transform(translation_matrix(0, -0.2, 0))

    assembly.add_part(AssemblyPart(
        mesh=shoulder_housing,
        name="Shoulder Joint",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.3, 0.0, 0.3]),
        explosion_distance=0.25,
        color=get_color(1),
    ))

    # Shoulder motor
    shoulder_motor = create_cylinder(radius=0.08, height=0.12, segments=16)
    shoulder_motor = shoulder_motor.transform(rotation_matrix_x(np.pi / 2))
    shoulder_motor = shoulder_motor.transform(translation_matrix(0.15, -0.2, 0))

    assembly.add_part(AssemblyPart(
        mesh=shoulder_motor,
        name="Shoulder Motor",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([1.0, 0.0, 0.0]),
        explosion_distance=0.2,
        color=(50, 50, 60),
    ))

    # === UPPER ARM ===
    upper_arm = create_cube(size=0.5)
    upper_arm = upper_arm.transform(scale_matrix(0.2, 1.0, 0.15))
    upper_arm = upper_arm.transform(translation_matrix(0, 0.15, 0))

    assembly.add_part(AssemblyPart(
        mesh=upper_arm,
        name="Upper Arm",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.3, 0.3]),
        explosion_distance=0.3,
        color=get_color(0),
    ))

    # Upper arm cable channel
    cable_channel = create_cylinder(radius=0.025, height=0.45, segments=8)
    cable_channel = cable_channel.transform(translation_matrix(-0.06, 0.15, 0))

    assembly.add_part(AssemblyPart(
        mesh=cable_channel,
        name="Upper Cable Channel",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-0.5, 0.2, 0.0]),
        explosion_distance=0.15,
        color=(40, 40, 45),
    ))

    # === ELBOW JOINT ===
    elbow = create_sphere(radius=0.1, segments=16, rings=12)
    elbow = elbow.transform(translation_matrix(0, 0.42, 0))

    assembly.add_part(AssemblyPart(
        mesh=elbow,
        name="Elbow Joint",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.3, 0.5, 0.0]),
        explosion_distance=0.25,
        color=get_color(1),
    ))

    # Elbow motor
    elbow_motor = create_cylinder(radius=0.06, height=0.1, segments=12)
    elbow_motor = elbow_motor.transform(rotation_matrix_x(np.pi / 2))
    elbow_motor = elbow_motor.transform(translation_matrix(0.1, 0.42, 0))

    assembly.add_part(AssemblyPart(
        mesh=elbow_motor,
        name="Elbow Motor",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([1.0, 0.3, 0.0]),
        explosion_distance=0.2,
        color=(50, 50, 60),
    ))

    # === FOREARM ===
    forearm = create_cube(size=0.4)
    forearm = forearm.transform(scale_matrix(0.15, 1.0, 0.12))
    forearm = forearm.transform(translation_matrix(0, 0.65, 0))

    assembly.add_part(AssemblyPart(
        mesh=forearm,
        name="Forearm",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.5, 0.3]),
        explosion_distance=0.35,
        color=get_color(0),
    ))

    # === WRIST ASSEMBLY (3-axis) ===
    # Wrist pitch
    wrist_pitch = create_cylinder(radius=0.06, height=0.08, segments=16)
    wrist_pitch = wrist_pitch.transform(rotation_matrix_x(np.pi / 2))
    wrist_pitch = wrist_pitch.transform(translation_matrix(0, 0.88, 0))

    assembly.add_part(AssemblyPart(
        mesh=wrist_pitch,
        name="Wrist Pitch",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.6, 0.3]),
        explosion_distance=0.25,
        color=get_color(2),
    ))

    # Wrist roll
    wrist_roll = create_cylinder(radius=0.05, height=0.06, segments=16)
    wrist_roll = wrist_roll.transform(translation_matrix(0, 0.95, 0))

    assembly.add_part(AssemblyPart(
        mesh=wrist_roll,
        name="Wrist Roll",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.7, 0.0]),
        explosion_distance=0.2,
        color=get_color(2),
    ))

    # Tool flange
    tool_flange = create_cylinder(radius=0.04, height=0.03, segments=16)
    tool_flange = tool_flange.transform(translation_matrix(0, 1.0, 0))

    assembly.add_part(AssemblyPart(
        mesh=tool_flange,
        name="Tool Flange",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.8, 0.0]),
        explosion_distance=0.15,
        color=(100, 100, 110),
    ))

    # === GRIPPER END EFFECTOR ===
    gripper_base = create_cube(size=0.06)
    gripper_base = gripper_base.transform(scale_matrix(1.5, 0.5, 1.0))
    gripper_base = gripper_base.transform(translation_matrix(0, 1.06, 0))

    assembly.add_part(AssemblyPart(
        mesh=gripper_base,
        name="Gripper Base",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.2,
        color=get_color(3),
    ))

    # Gripper fingers
    for side in [-1, 1]:
        finger = create_cube(size=0.08)
        finger = finger.transform(scale_matrix(0.3, 1.2, 0.4))
        finger = finger.transform(translation_matrix(side * 0.035, 1.12, 0))

        assembly.add_part(AssemblyPart(
            mesh=finger,
            name=f"Gripper Finger {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 0.5, 0.8, 0.0]),
            explosion_distance=0.15,
            color=(80, 80, 90),
        ))

    # Gripper tips
    for side in [-1, 1]:
        tip = create_cube(size=0.04)
        tip = tip.transform(scale_matrix(0.4, 0.8, 0.6))
        tip = tip.transform(translation_matrix(side * 0.03, 1.2, 0))

        assembly.add_part(AssemblyPart(
            mesh=tip,
            name=f"Gripper Tip {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 0.6, 1.0, 0.0]),
            explosion_distance=0.12,
            color=(255, 150, 50),
        ))

    return assembly


def create_satellite_assembly() -> Assembly:
    """
    Create a detailed communication satellite assembly (~35 parts).

    Parts include:
    - Main bus (central body)
    - Solar panels (2 wings, multiple segments)
    - Communication dish
    - Antenna arrays
    - Thrusters
    - Sensor package
    - Thermal radiators
    """
    assembly = Assembly("Communication Satellite")

    # === MAIN BUS ===
    bus_body = create_cube(size=0.6)
    bus_body = bus_body.transform(scale_matrix(1.0, 1.2, 0.8))

    assembly.add_part(AssemblyPart(
        mesh=bus_body,
        name="Main Bus",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=(200, 180, 100),  # Gold foil
    ))

    # === SOLAR PANELS ===
    panel_width = 0.4
    panel_length = 0.8

    for wing in [-1, 1]:
        for segment in range(3):
            panel = create_cube(size=panel_width)
            panel = panel.transform(scale_matrix(1.0, 0.02, 2.0))

            x_pos = wing * (0.35 + segment * panel_width)
            panel = panel.transform(translation_matrix(x_pos, 0.1, 0))

            assembly.add_part(AssemblyPart(
                mesh=panel,
                name=f"Solar Panel {'L' if wing < 0 else 'R'}{segment+1}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([wing * 0.8, 0.2, 0.0]),
                explosion_distance=0.3 + segment * 0.15,
                color=(40, 40, 80),  # Dark blue solar cells
            ))

            # Panel support structure
            support = create_cylinder(radius=0.015, height=panel_width * 0.9, segments=8)
            support = support.transform(rotation_matrix_z(np.pi / 2))
            support = support.transform(translation_matrix(x_pos, 0.05, 0))

            assembly.add_part(AssemblyPart(
                mesh=support,
                name=f"Panel Support {'L' if wing < 0 else 'R'}{segment+1}",
                base_position=np.array([0.0, 0.0, 0.0]),
                explosion_direction=np.array([wing * 0.6, -0.3, 0.0]),
                explosion_distance=0.25 + segment * 0.1,
                color=(180, 180, 190),
            ))

    # Solar panel hinges
    for wing in [-1, 1]:
        hinge = create_cylinder(radius=0.025, height=0.1, segments=12)
        hinge = hinge.transform(rotation_matrix_x(np.pi / 2))
        hinge = hinge.transform(translation_matrix(wing * 0.32, 0.1, 0))

        assembly.add_part(AssemblyPart(
            mesh=hinge,
            name=f"Panel Hinge {'L' if wing < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([wing * 0.4, 0.0, 0.3]),
            explosion_distance=0.2,
            color=(100, 100, 110),
        ))

    # === COMMUNICATION DISH ===
    # Dish reflector
    dish_verts = []
    dish_faces = []
    dish_segments = 24
    dish_rings = 8
    dish_radius = 0.35
    dish_depth = 0.08

    for i in range(dish_rings + 1):
        r = dish_radius * i / dish_rings
        z = -dish_depth * (r / dish_radius) ** 2

        for j in range(dish_segments):
            theta = 2 * np.pi * j / dish_segments
            dish_verts.append([r * np.cos(theta), 0.45 + z, r * np.sin(theta)])

    for i in range(dish_rings):
        for j in range(dish_segments):
            curr = i * dish_segments + j
            next_j = i * dish_segments + (j + 1) % dish_segments
            next_i = (i + 1) * dish_segments + j
            next_both = (i + 1) * dish_segments + (j + 1) % dish_segments

            if i != 0:
                dish_faces.append([curr, next_i, next_j])
            dish_faces.append([next_j, next_i, next_both])

    dish = Mesh(
        vertices=np.array(dish_verts, dtype=np.float32),
        faces=np.array(dish_faces, dtype=np.int32),
    )

    assembly.add_part(AssemblyPart(
        mesh=dish,
        name="Comm Dish Reflector",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.5,
        color=(220, 220, 230),
    ))

    # Dish feed horn
    feed = create_cone(radius=0.04, height=0.1, segments=12)
    feed = feed.transform(translation_matrix(0, 0.42, 0))

    assembly.add_part(AssemblyPart(
        mesh=feed,
        name="Feed Horn",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.8, 0.2]),
        explosion_distance=0.35,
        color=get_color(3),
    ))

    # Dish support struts
    for i in range(3):
        angle = 2 * np.pi * i / 3

        strut = create_cylinder(radius=0.008, height=0.35, segments=6)
        strut = strut.transform(rotation_matrix_x(0.4))
        strut = strut.transform(rotation_matrix_y(angle))
        strut = strut.transform(translation_matrix(0, 0.35, 0))

        assembly.add_part(AssemblyPart(
            mesh=strut,
            name=f"Dish Strut {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                0.3 * np.cos(angle),
                0.6,
                0.3 * np.sin(angle)
            ]),
            explosion_distance=0.3,
            color=(150, 150, 160),
        ))

    # === ANTENNA ARRAYS ===
    for i in range(4):
        angle = np.pi / 4 + i * np.pi / 2

        antenna = create_cylinder(radius=0.015, height=0.2, segments=8)
        antenna = antenna.transform(translation_matrix(
            0.32 * np.cos(angle),
            -0.3,
            0.25 * np.sin(angle)
        ))

        assembly.add_part(AssemblyPart(
            mesh=antenna,
            name=f"Antenna {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                0.3 * np.cos(angle),
                -0.5,
                0.3 * np.sin(angle)
            ]),
            explosion_distance=0.25,
            color=(200, 200, 210),
        ))

    # === THRUSTERS ===
    thruster_positions = [
        (0.25, -0.35, 0.2), (-0.25, -0.35, 0.2),
        (0.25, -0.35, -0.2), (-0.25, -0.35, -0.2),
    ]

    for i, (tx, ty, tz) in enumerate(thruster_positions):
        thruster = create_cone(radius=0.03, height=0.06, segments=10)
        thruster = thruster.transform(rotation_matrix_x(np.pi))
        thruster = thruster.transform(translation_matrix(tx, ty, tz))

        assembly.add_part(AssemblyPart(
            mesh=thruster,
            name=f"Thruster {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([tx, -0.8, tz]),
            explosion_distance=0.3,
            color=(100, 100, 110),
        ))

    # === SENSOR PACKAGE ===
    sensor_body = create_cube(size=0.12)
    sensor_body = sensor_body.transform(translation_matrix(0, -0.42, 0))

    assembly.add_part(AssemblyPart(
        mesh=sensor_body,
        name="Sensor Package",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.35,
        color=get_color(2),
    ))

    # Star trackers
    for side in [-1, 1]:
        tracker = create_cylinder(radius=0.025, height=0.04, segments=10)
        tracker = tracker.transform(rotation_matrix_x(np.pi / 4 * side))
        tracker = tracker.transform(translation_matrix(side * 0.15, -0.38, 0))

        assembly.add_part(AssemblyPart(
            mesh=tracker,
            name=f"Star Tracker {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 0.5, -0.5, 0.0]),
            explosion_distance=0.2,
            color=(50, 50, 60),
        ))

    # === THERMAL RADIATORS ===
    for side in [-1, 1]:
        radiator = create_cube(size=0.25)
        radiator = radiator.transform(scale_matrix(0.1, 1.0, 0.8))
        radiator = radiator.transform(translation_matrix(side * 0.32, 0, 0.25))

        assembly.add_part(AssemblyPart(
            mesh=radiator,
            name=f"Radiator {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 0.6, 0.0, 0.4]),
            explosion_distance=0.25,
            color=(240, 240, 250),
        ))

    return assembly


def create_microscope_assembly() -> Assembly:
    """
    Create a laboratory microscope assembly (~25 parts).
    """
    assembly = Assembly("Laboratory Microscope")

    # Base
    base = create_cube(size=0.5)
    base = base.transform(scale_matrix(1.5, 0.15, 1.0))
    base = base.transform(translation_matrix(0, -0.5, 0))

    assembly.add_part(AssemblyPart(
        mesh=base,
        name="Base",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.3,
        color=(40, 40, 45),
    ))

    # Arm (C-shaped support)
    arm_vertical = create_cube(size=0.7)
    arm_vertical = arm_vertical.transform(scale_matrix(0.15, 1.0, 0.2))
    arm_vertical = arm_vertical.transform(translation_matrix(-0.2, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=arm_vertical,
        name="Arm Vertical",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-0.5, 0.0, 0.0]),
        explosion_distance=0.25,
        color=(50, 50, 55),
    ))

    arm_horizontal = create_cube(size=0.3)
    arm_horizontal = arm_horizontal.transform(scale_matrix(1.0, 0.15, 0.2))
    arm_horizontal = arm_horizontal.transform(translation_matrix(0, 0.35, 0))

    assembly.add_part(AssemblyPart(
        mesh=arm_horizontal,
        name="Arm Horizontal",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.5, 0.0]),
        explosion_distance=0.2,
        color=(50, 50, 55),
    ))

    # Stage
    stage = create_cylinder(radius=0.2, height=0.03, segments=24)
    stage = stage.transform(translation_matrix(0.1, -0.25, 0))

    assembly.add_part(AssemblyPart(
        mesh=stage,
        name="Stage",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.3, 0.0, 0.3]),
        explosion_distance=0.25,
        color=(30, 30, 35),
    ))

    # Stage clips
    for side in [-1, 1]:
        clip = create_cube(size=0.05)
        clip = clip.transform(scale_matrix(0.3, 0.2, 1.0))
        clip = clip.transform(translation_matrix(0.1, -0.22, side * 0.15))

        assembly.add_part(AssemblyPart(
            mesh=clip,
            name=f"Stage Clip {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, 0.2, side * 0.5]),
            explosion_distance=0.15,
            color=(180, 180, 190),
        ))

    # Objective turret
    turret = create_cylinder(radius=0.08, height=0.05, segments=20)
    turret = turret.transform(translation_matrix(0.1, -0.1, 0))

    assembly.add_part(AssemblyPart(
        mesh=turret,
        name="Objective Turret",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.3, -0.3, 0.0]),
        explosion_distance=0.2,
        color=(60, 60, 65),
    ))

    # Objectives (4x, 10x, 40x, 100x)
    objective_mags = ["4x", "10x", "40x", "100x"]
    objective_heights = [0.06, 0.08, 0.1, 0.12]

    for i, (mag, height) in enumerate(zip(objective_mags, objective_heights)):
        angle = i * np.pi / 2

        obj = create_cylinder(radius=0.02, height=height, segments=12)
        obj = obj.transform(translation_matrix(
            0.1 + 0.05 * np.cos(angle),
            -0.15 - height / 2,
            0.05 * np.sin(angle)
        ))

        assembly.add_part(AssemblyPart(
            mesh=obj,
            name=f"Objective {mag}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                0.2 * np.cos(angle),
                -0.5,
                0.2 * np.sin(angle)
            ]),
            explosion_distance=0.2 + i * 0.05,
            color=get_color(i),
        ))

    # Body tube
    tube = create_cylinder(radius=0.04, height=0.25, segments=16)
    tube = tube.transform(translation_matrix(0.1, 0.1, 0))

    assembly.add_part(AssemblyPart(
        mesh=tube,
        name="Body Tube",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.2, 0.5, 0.0]),
        explosion_distance=0.25,
        color=(40, 40, 45),
    ))

    # Eyepiece
    eyepiece = create_cylinder(radius=0.025, height=0.08, segments=12)
    eyepiece = eyepiece.transform(translation_matrix(0.1, 0.3, 0))

    assembly.add_part(AssemblyPart(
        mesh=eyepiece,
        name="Eyepiece",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.3,
        color=(30, 30, 35),
    ))

    # Focus knobs
    for side in [-1, 1]:
        knob = create_cylinder(radius=0.04, height=0.03, segments=16)
        knob = knob.transform(rotation_matrix_x(np.pi / 2))
        knob = knob.transform(translation_matrix(-0.28, -0.1, side * 0.1))

        assembly.add_part(AssemblyPart(
            mesh=knob,
            name=f"Focus Knob {'Coarse' if side < 0 else 'Fine'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([-0.5, 0.0, side * 0.3]),
            explosion_distance=0.15,
            color=(180, 180, 190),
        ))

    # Light source
    light = create_cylinder(radius=0.05, height=0.06, segments=16)
    light = light.transform(translation_matrix(0.1, -0.4, 0))

    assembly.add_part(AssemblyPart(
        mesh=light,
        name="Light Source",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.8, 0.0]),
        explosion_distance=0.2,
        color=(255, 255, 200),
    ))

    return assembly


def create_differential_assembly() -> Assembly:
    """
    Create an automotive differential assembly (~20 parts).
    """
    assembly = Assembly("Automotive Differential")

    # Housing (two halves)
    for side in [-1, 1]:
        housing = create_sphere(radius=0.25, segments=20, rings=16)
        housing = housing.transform(scale_matrix(0.8, 1.0, 1.0))

        # Cut in half (approximate by scaling)
        verts = housing.vertices.copy()
        mask = verts[:, 0] * side > 0
        verts[~mask, 0] = 0

        housing = Mesh(vertices=verts, faces=housing.faces)
        housing = housing.transform(translation_matrix(side * 0.02, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=housing,
            name=f"Housing {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 1.0, 0.0, 0.0]),
            explosion_distance=0.3,
            color=(100, 100, 110),
        ))

    # Ring gear
    ring = create_torus(major_radius=0.2, minor_radius=0.025, major_segments=32, minor_segments=12)
    ring = ring.transform(rotation_matrix_y(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=ring,
        name="Ring Gear",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.5]),
        explosion_distance=0.25,
        color=get_color(3),
    ))

    # Pinion gear
    pinion = create_cylinder(radius=0.04, height=0.15, segments=16)
    pinion = pinion.transform(rotation_matrix_x(np.pi / 2))
    pinion = pinion.transform(translation_matrix(0, -0.2, 0))

    assembly.add_part(AssemblyPart(
        mesh=pinion,
        name="Pinion Gear",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.35,
        color=get_color(2),
    ))

    # Pinion shaft
    pinion_shaft = create_cylinder(radius=0.02, height=0.25, segments=10)
    pinion_shaft = pinion_shaft.transform(rotation_matrix_x(np.pi / 2))
    pinion_shaft = pinion_shaft.transform(translation_matrix(0, -0.25, 0))

    assembly.add_part(AssemblyPart(
        mesh=pinion_shaft,
        name="Pinion Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.8, 0.3]),
        explosion_distance=0.3,
        color=(80, 80, 90),
    ))

    # Spider gears (4)
    for i in range(4):
        angle = i * np.pi / 2

        spider = create_cylinder(radius=0.03, height=0.04, segments=12)
        spider = spider.transform(rotation_matrix_x(np.pi / 2))
        spider = spider.transform(rotation_matrix_y(angle))
        spider = spider.transform(translation_matrix(
            0.08 * np.cos(angle),
            0,
            0.08 * np.sin(angle)
        ))

        assembly.add_part(AssemblyPart(
            mesh=spider,
            name=f"Spider Gear {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                0.5 * np.cos(angle),
                0.3,
                0.5 * np.sin(angle)
            ]),
            explosion_distance=0.2,
            color=get_color(0),
        ))

    # Side gears (2)
    for side in [-1, 1]:
        side_gear = create_cylinder(radius=0.05, height=0.03, segments=16)
        side_gear = side_gear.transform(rotation_matrix_z(np.pi / 2))
        side_gear = side_gear.transform(translation_matrix(side * 0.06, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=side_gear,
            name=f"Side Gear {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 0.6, 0.2, 0.0]),
            explosion_distance=0.2,
            color=get_color(1),
        ))

    # Axle shafts
    for side in [-1, 1]:
        axle = create_cylinder(radius=0.015, height=0.3, segments=10)
        axle = axle.transform(rotation_matrix_z(np.pi / 2))
        axle = axle.transform(translation_matrix(side * 0.2, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=axle,
            name=f"Axle {'L' if side < 0 else 'R'}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([side * 1.0, 0.0, 0.0]),
            explosion_distance=0.4,
            color=(60, 60, 70),
        ))

    # Bearings
    bearing_positions = [(-0.08, 0, 0), (0.08, 0, 0), (0, -0.15, 0)]

    for i, (bx, by, bz) in enumerate(bearing_positions):
        bearing = create_torus(major_radius=0.025, minor_radius=0.008, major_segments=16, minor_segments=8)

        if i < 2:
            bearing = bearing.transform(rotation_matrix_y(np.pi / 2))
        else:
            bearing = bearing.transform(rotation_matrix_x(np.pi / 2))

        bearing = bearing.transform(translation_matrix(bx, by, bz))

        assembly.add_part(AssemblyPart(
            mesh=bearing,
            name=f"Bearing {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([bx * 3, by * 2 if by != 0 else 0.3, bz]),
            explosion_distance=0.15,
            color=(200, 180, 50),
        ))

    return assembly


# Factory for complex assemblies
COMPLEX_ASSEMBLY_CREATORS = {
    "jet_engine": create_jet_engine_assembly,
    "robotic_arm": create_robotic_arm_assembly,
    "satellite": create_satellite_assembly,
    "microscope": create_microscope_assembly,
    "differential": create_differential_assembly,
}


def create_complex_assembly(name: str) -> Assembly:
    """Create a complex assembly by name."""
    if name not in COMPLEX_ASSEMBLY_CREATORS:
        raise ValueError(
            f"Unknown assembly: {name}. Available: {list(COMPLEX_ASSEMBLY_CREATORS.keys())}"
        )
    return COMPLEX_ASSEMBLY_CREATORS[name]()


def get_available_complex_assemblies() -> list:
    """Get list of available complex assembly names."""
    return list(COMPLEX_ASSEMBLY_CREATORS.keys())
