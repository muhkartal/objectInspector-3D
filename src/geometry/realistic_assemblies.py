"""
Realistic high-detail assemblies using proper mesh geometry.
These look like actual objects, not blocky primitives.
"""

import numpy as np
from typing import Tuple

from config import settings
from src.geometry.mesh import Mesh
from src.geometry.assembly import Assembly, AssemblyPart
from src.geometry.detailed_parts import (
    create_smooth_cylinder,
    create_turbine_blade,
    create_gear,
    create_piston,
    create_bearing,
    create_spring_coil,
    create_hex_bolt,
)
from src.geometry.advanced_surfaces import (
    create_geodesic_sphere,
    create_spring,
)
from src.geometry.primitives import (
    create_sphere,
    create_cylinder,
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


def create_realistic_engine_assembly() -> Assembly:
    """
    Create a highly detailed 4-cylinder engine.
    Uses proper detailed parts instead of basic primitives.
    """
    assembly = Assembly("Detailed 4-Cylinder Engine")

    # Engine block with proper detail
    block = create_smooth_cylinder(radius=0.35, height=0.6, segments=8, rings=4)
    block = block.transform(scale_matrix(1.2, 1.0, 0.8))

    assembly.add_part(AssemblyPart(
        mesh=block,
        name="Engine Block",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=(120, 125, 130),
    ))

    # Cylinder head with cooling fins
    head_parts = []
    head_base = create_smooth_cylinder(radius=0.34, height=0.15, segments=8, rings=2)
    head_base = head_base.transform(translation_matrix(0, 0.38, 0))

    # Add cooling fins
    for i in range(8):
        fin = create_smooth_cylinder(radius=0.36, height=0.01, segments=8)
        fin = fin.transform(translation_matrix(0, 0.32 + i * 0.015, 0))

    assembly.add_part(AssemblyPart(
        mesh=head_base,
        name="Cylinder Head",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.6,
        color=(100, 105, 110),
    ))

    # Detailed pistons
    for i in range(4):
        px = (i - 1.5) * 0.15

        piston = create_piston(
            diameter=0.1,
            height=0.12,
            skirt_height=0.06,
            crown_shape="dome",
            ring_grooves=3,
            segments=24,
        )
        piston = piston.transform(translation_matrix(px, 0.1, 0))

        assembly.add_part(AssemblyPart(
            mesh=piston,
            name=f"Piston {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, 1.0, 0.0]),
            explosion_distance=0.4 + i * 0.08,
            color=(180, 170, 160),
        ))

        # Connecting rod
        rod = create_smooth_cylinder(radius=0.015, height=0.15, segments=16)
        rod = rod.transform(translation_matrix(px, 0.0, 0))

        assembly.add_part(AssemblyPart(
            mesh=rod,
            name=f"Connecting Rod {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.1 * (i - 1.5), 0.5, 0.0]),
            explosion_distance=0.3 + i * 0.05,
            color=(150, 150, 155),
        ))

    # Crankshaft with proper detail
    crankshaft = create_smooth_cylinder(radius=0.03, height=0.7, segments=24, rings=8)
    crankshaft = crankshaft.transform(rotation_matrix_z(np.pi / 2))
    crankshaft = crankshaft.transform(translation_matrix(0, -0.2, 0))

    assembly.add_part(AssemblyPart(
        mesh=crankshaft,
        name="Crankshaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.5,
        color=(80, 80, 85),
    ))

    # Crankshaft counterweights
    for i in range(4):
        px = (i - 1.5) * 0.15

        weight = create_smooth_cylinder(radius=0.06, height=0.02, segments=16)
        weight = weight.transform(scale_matrix(1.5, 1.0, 0.5))
        weight = weight.transform(rotation_matrix_z(np.pi / 2))
        weight = weight.transform(translation_matrix(px, -0.2, 0.04))

        assembly.add_part(AssemblyPart(
            mesh=weight,
            name=f"Counterweight {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, -0.5, 0.5]),
            explosion_distance=0.35,
            color=(70, 70, 75),
        ))

    # Main bearings
    for i in range(5):
        px = (i - 2) * 0.14

        bearing = create_bearing(
            outer_radius=0.045,
            inner_radius=0.025,
            width=0.015,
            num_balls=8,
            ball_radius=0.006,
            segments=20,
        )
        bearing = bearing.transform(rotation_matrix_z(np.pi / 2))
        bearing = bearing.transform(translation_matrix(px, -0.2, 0))

        assembly.add_part(AssemblyPart(
            mesh=bearing,
            name=f"Main Bearing {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.2 * (i - 2), -0.6, 0.3]),
            explosion_distance=0.3,
            color=(200, 190, 50),
        ))

    # Camshaft
    camshaft = create_smooth_cylinder(radius=0.02, height=0.65, segments=16, rings=10)
    camshaft = camshaft.transform(rotation_matrix_z(np.pi / 2))
    camshaft = camshaft.transform(translation_matrix(0, 0.25, 0.12))

    assembly.add_part(AssemblyPart(
        mesh=camshaft,
        name="Camshaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.5, 0.5]),
        explosion_distance=0.45,
        color=(90, 90, 95),
    ))

    # Cam lobes
    for i in range(8):
        px = (i - 3.5) * 0.08

        lobe = create_smooth_cylinder(radius=0.025, height=0.015, segments=16)
        lobe = lobe.transform(scale_matrix(1.0, 1.0, 1.5))
        lobe = lobe.transform(rotation_matrix_z(np.pi / 2))
        lobe = lobe.transform(rotation_matrix_x(i * np.pi / 4))
        lobe = lobe.transform(translation_matrix(px, 0.25, 0.12))

        assembly.add_part(AssemblyPart(
            mesh=lobe,
            name=f"Cam Lobe {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.1 * (i - 3.5), 0.4, 0.4]),
            explosion_distance=0.35,
            color=(85, 85, 90),
        ))

    # Timing gears
    timing_gear_crank = create_gear(
        outer_radius=0.06,
        inner_radius=0.02,
        thickness=0.015,
        num_teeth=20,
        tooth_height=0.008,
        segments=3,
    )
    timing_gear_crank = timing_gear_crank.transform(rotation_matrix_z(np.pi / 2))
    timing_gear_crank = timing_gear_crank.transform(translation_matrix(-0.33, -0.2, 0))

    assembly.add_part(AssemblyPart(
        mesh=timing_gear_crank,
        name="Crank Timing Gear",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-1.0, -0.3, 0.0]),
        explosion_distance=0.35,
        color=get_color(2),
    ))

    timing_gear_cam = create_gear(
        outer_radius=0.12,
        inner_radius=0.015,
        thickness=0.015,
        num_teeth=40,
        tooth_height=0.008,
        segments=3,
    )
    timing_gear_cam = timing_gear_cam.transform(rotation_matrix_z(np.pi / 2))
    timing_gear_cam = timing_gear_cam.transform(translation_matrix(-0.33, 0.25, 0.12))

    assembly.add_part(AssemblyPart(
        mesh=timing_gear_cam,
        name="Cam Timing Gear",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-1.0, 0.3, 0.3]),
        explosion_distance=0.4,
        color=get_color(2),
    ))

    # Valve springs
    for i in range(8):
        px = (i - 3.5) * 0.08

        spring = create_spring_coil(
            coil_radius=0.015,
            wire_radius=0.002,
            pitch=0.008,
            num_coils=5,
            segments_coil=30,
            segments_wire=8,
        )
        spring = spring.transform(translation_matrix(px, 0.35, 0.08))

        assembly.add_part(AssemblyPart(
            mesh=spring,
            name=f"Valve Spring {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.05 * (i - 3.5), 0.8, 0.2]),
            explosion_distance=0.3,
            color=(100, 180, 100),
        ))

    # Oil pan
    oil_pan = create_smooth_cylinder(radius=0.32, height=0.1, segments=8)
    oil_pan = oil_pan.transform(scale_matrix(1.1, 1.0, 0.7))
    oil_pan = oil_pan.transform(translation_matrix(0, -0.35, 0))

    assembly.add_part(AssemblyPart(
        mesh=oil_pan,
        name="Oil Pan",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.6,
        color=(60, 60, 65),
    ))

    # Intake manifold
    intake = create_smooth_cylinder(radius=0.15, height=0.08, segments=20)
    intake = intake.transform(scale_matrix(1.5, 1.0, 0.5))
    intake = intake.transform(translation_matrix(0, 0.35, 0.2))

    assembly.add_part(AssemblyPart(
        mesh=intake,
        name="Intake Manifold",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.3, 1.0]),
        explosion_distance=0.5,
        color=get_color(0),
    ))

    # Exhaust manifold
    for i in range(4):
        px = (i - 1.5) * 0.15

        exhaust_runner = create_smooth_cylinder(radius=0.025, height=0.15, segments=16)
        exhaust_runner = exhaust_runner.transform(rotation_matrix_x(np.pi / 3))
        exhaust_runner = exhaust_runner.transform(translation_matrix(px, 0.2, -0.2))

        assembly.add_part(AssemblyPart(
            mesh=exhaust_runner,
            name=f"Exhaust Runner {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, 0.2, -0.8]),
            explosion_distance=0.4 + i * 0.05,
            color=(150, 80, 50),
        ))

    # Spark plugs
    for i in range(4):
        px = (i - 1.5) * 0.15

        plug = create_smooth_cylinder(radius=0.012, height=0.08, segments=12)
        plug = plug.transform(translation_matrix(px, 0.42, 0))

        assembly.add_part(AssemblyPart(
            mesh=plug,
            name=f"Spark Plug {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, 1.0, 0.0]),
            explosion_distance=0.5 + i * 0.03,
            color=(220, 220, 225),
        ))

    # Head bolts
    bolt_positions = [
        (-0.25, 0.38, 0.15), (0.25, 0.38, 0.15),
        (-0.25, 0.38, -0.15), (0.25, 0.38, -0.15),
        (-0.1, 0.38, 0.18), (0.1, 0.38, 0.18),
        (-0.1, 0.38, -0.18), (0.1, 0.38, -0.18),
    ]

    for i, (bx, by, bz) in enumerate(bolt_positions):
        bolt = create_hex_bolt(
            head_radius=0.015,
            head_height=0.008,
            shaft_radius=0.006,
            shaft_length=0.05,
            segments=16,
        )
        bolt = bolt.transform(rotation_matrix_x(np.pi))
        bolt = bolt.transform(translation_matrix(bx, by, bz))

        assembly.add_part(AssemblyPart(
            mesh=bolt,
            name=f"Head Bolt {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([bx, 0.8, bz]),
            explosion_distance=0.25,
            color=(50, 50, 55),
        ))

    return assembly


def create_realistic_turbofan_assembly() -> Assembly:
    """
    Create a highly detailed turbofan engine with realistic blades.
    """
    assembly = Assembly("Detailed Turbofan Engine")

    # Nacelle outer casing
    nacelle = create_smooth_cylinder(radius=0.45, height=1.6, segments=48, rings=8, capped=False)
    nacelle = nacelle.transform(rotation_matrix_z(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=nacelle,
        name="Nacelle",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.5, 0.0]),
        explosion_distance=0.25,
        color=(180, 185, 190),
    ))

    # Fan hub (spinner)
    spinner = create_cone(radius=0.12, height=0.2, segments=32)
    spinner = spinner.transform(rotation_matrix_z(np.pi / 2))
    spinner = spinner.transform(translation_matrix(-0.7, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=spinner,
        name="Fan Spinner",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-1.0, 0.0, 0.0]),
        explosion_distance=0.4,
        color=(200, 200, 205),
    ))

    # Fan blades (detailed airfoil blades)
    num_fan_blades = 18

    for i in range(num_fan_blades):
        angle = 2 * np.pi * i / num_fan_blades

        blade = create_turbine_blade(
            length=0.32,
            chord=0.08,
            thickness=0.04,
            twist=0.4,
            taper=0.5,
            segments_span=15,
            segments_chord=12,
        )

        # Position and orient blade
        blade = blade.transform(rotation_matrix_x(0.3))  # Blade pitch
        blade = blade.transform(translation_matrix(0, 0.11, 0))  # Move to hub edge
        blade = blade.transform(rotation_matrix_y(angle))  # Rotate around hub
        blade = blade.transform(translation_matrix(-0.65, 0, 0))  # Move to fan position

        assembly.add_part(AssemblyPart(
            mesh=blade,
            name=f"Fan Blade {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([
                -0.3,
                0.7 * np.cos(angle),
                0.7 * np.sin(angle)
            ]),
            explosion_distance=0.45,
            color=(210, 210, 215),
        ))

    # Compressor stages
    compressor_x = [-0.4, -0.25, -0.1, 0.05]
    compressor_radii = [0.32, 0.28, 0.24, 0.20]

    for stage, (x_pos, radius) in enumerate(zip(compressor_x, compressor_radii)):
        # Compressor disk
        disk = create_smooth_cylinder(radius=radius, height=0.025, segments=32)
        disk = disk.transform(rotation_matrix_z(np.pi / 2))
        disk = disk.transform(translation_matrix(x_pos, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=disk,
            name=f"Compressor Disk {stage+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([-0.3, 0.2 + stage * 0.1, 0.0]),
            explosion_distance=0.35 + stage * 0.08,
            color=get_color(0),
        ))

        # Compressor blades (smaller)
        num_blades = 30 - stage * 4

        for i in range(num_blades):
            angle = 2 * np.pi * i / num_blades

            c_blade = create_turbine_blade(
                length=0.06 - stage * 0.01,
                chord=0.02,
                thickness=0.03,
                twist=0.2,
                taper=0.6,
                segments_span=8,
                segments_chord=8,
            )
            c_blade = c_blade.transform(rotation_matrix_x(0.4))
            c_blade = c_blade.transform(translation_matrix(0, radius - 0.02, 0))
            c_blade = c_blade.transform(rotation_matrix_y(angle))
            c_blade = c_blade.transform(translation_matrix(x_pos + 0.015, 0, 0))

            if i % 3 == 0:  # Only add every 3rd blade to assembly to keep part count reasonable
                assembly.add_part(AssemblyPart(
                    mesh=c_blade,
                    name=f"Comp{stage+1} Blade {i//3 + 1}",
                    base_position=np.array([0.0, 0.0, 0.0]),
                    explosion_direction=np.array([
                        -0.2,
                        0.4 * np.cos(angle),
                        0.4 * np.sin(angle)
                    ]),
                    explosion_distance=0.3 + stage * 0.05,
                    color=(190, 190, 195),
                ))

    # Combustion chamber
    combustor = create_smooth_cylinder(radius=0.18, height=0.2, segments=32, capped=False)
    combustor = combustor.transform(rotation_matrix_z(np.pi / 2))
    combustor = combustor.transform(translation_matrix(0.2, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=combustor,
        name="Combustion Chamber",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=(200, 120, 80),
    ))

    # Fuel nozzles
    for i in range(8):
        angle = 2 * np.pi * i / 8

        nozzle = create_smooth_cylinder(radius=0.015, height=0.06, segments=12)
        nozzle = nozzle.transform(translation_matrix(
            0.12,
            0.15 * np.cos(angle),
            0.15 * np.sin(angle)
        ))

        assembly.add_part(AssemblyPart(
            mesh=nozzle,
            name=f"Fuel Nozzle {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, np.cos(angle), np.sin(angle)]),
            explosion_distance=0.2,
            color=get_color(3),
        ))

    # Turbine stages
    turbine_x = [0.35, 0.48, 0.6]
    turbine_radii = [0.15, 0.18, 0.20]

    for stage, (x_pos, radius) in enumerate(zip(turbine_x, turbine_radii)):
        # Turbine disk
        disk = create_smooth_cylinder(radius=radius, height=0.02, segments=32)
        disk = disk.transform(rotation_matrix_z(np.pi / 2))
        disk = disk.transform(translation_matrix(x_pos, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=disk,
            name=f"Turbine Disk {stage+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.5, 0.2 - stage * 0.1, 0.0]),
            explosion_distance=0.4 + stage * 0.1,
            color=get_color(1),
        ))

    # Exhaust cone
    exhaust = create_cone(radius=0.28, height=0.2, segments=32)
    exhaust = exhaust.transform(rotation_matrix_z(-np.pi / 2))
    exhaust = exhaust.transform(translation_matrix(0.75, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=exhaust,
        name="Exhaust Cone",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([1.0, 0.0, 0.0]),
        explosion_distance=0.5,
        color=(150, 150, 155),
    ))

    # Core shaft
    shaft = create_smooth_cylinder(radius=0.025, height=1.3, segments=20)
    shaft = shaft.transform(rotation_matrix_z(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=shaft,
        name="Core Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.8, 0.0]),
        explosion_distance=0.35,
        color=(80, 80, 85),
    ))

    # Bearings
    bearing_x = [-0.5, 0.0, 0.5]

    for i, bx in enumerate(bearing_x):
        bearing = create_bearing(
            outer_radius=0.05,
            inner_radius=0.028,
            width=0.02,
            num_balls=10,
            ball_radius=0.008,
            segments=24,
        )
        bearing = bearing.transform(rotation_matrix_z(np.pi / 2))
        bearing = bearing.transform(translation_matrix(bx, 0, 0))

        assembly.add_part(AssemblyPart(
            mesh=bearing,
            name=f"Shaft Bearing {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([bx * 0.3, -0.5, 0.3]),
            explosion_distance=0.25,
            color=(200, 180, 50),
        ))

    return assembly


def create_realistic_gearbox_assembly() -> Assembly:
    """
    Create a detailed gearbox with proper gear meshes.
    """
    assembly = Assembly("Detailed 6-Speed Gearbox")

    # Housing bottom
    housing_bot = create_smooth_cylinder(radius=0.25, height=0.2, segments=32)
    housing_bot = housing_bot.transform(scale_matrix(1.4, 0.8, 1.0))
    housing_bot = housing_bot.transform(translation_matrix(0, -0.1, 0))

    assembly.add_part(AssemblyPart(
        mesh=housing_bot,
        name="Housing Bottom",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.4,
        color=(100, 100, 105),
    ))

    # Housing top
    housing_top = create_smooth_cylinder(radius=0.24, height=0.15, segments=32)
    housing_top = housing_top.transform(scale_matrix(1.35, 0.7, 0.95))
    housing_top = housing_top.transform(translation_matrix(0, 0.1, 0))

    assembly.add_part(AssemblyPart(
        mesh=housing_top,
        name="Housing Top",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.45,
        color=(100, 100, 105),
    ))

    # Input shaft with gear
    input_shaft = create_smooth_cylinder(radius=0.025, height=0.5, segments=20)
    input_shaft = input_shaft.transform(rotation_matrix_x(np.pi / 2))
    input_shaft = input_shaft.transform(translation_matrix(-0.15, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=input_shaft,
        name="Input Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 1.0]),
        explosion_distance=0.5,
        color=(70, 70, 75),
    ))

    # Input gear
    input_gear = create_gear(
        outer_radius=0.08,
        inner_radius=0.025,
        thickness=0.02,
        num_teeth=18,
        tooth_height=0.008,
        segments=3,
    )
    input_gear = input_gear.transform(translation_matrix(-0.15, 0, 0.1))

    assembly.add_part(AssemblyPart(
        mesh=input_gear,
        name="Input Gear",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-0.5, 0.3, 0.5]),
        explosion_distance=0.35,
        color=get_color(0),
    ))

    # Output shaft
    output_shaft = create_smooth_cylinder(radius=0.025, height=0.5, segments=20)
    output_shaft = output_shaft.transform(rotation_matrix_x(np.pi / 2))
    output_shaft = output_shaft.transform(translation_matrix(0.15, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=output_shaft,
        name="Output Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, -1.0]),
        explosion_distance=0.5,
        color=(70, 70, 75),
    ))

    # Gear pairs (6 speeds)
    gear_specs = [
        (0.12, 0.06, 36, 18, -0.15),  # 1st: large input, small output
        (0.10, 0.07, 30, 21, -0.08),  # 2nd
        (0.09, 0.08, 27, 24, -0.01),  # 3rd
        (0.085, 0.085, 25, 25, 0.06),  # 4th (direct)
        (0.07, 0.10, 21, 30, 0.13),   # 5th
        (0.06, 0.11, 18, 33, 0.20),   # 6th: small input, large output
    ]

    for i, (r_in, r_out, t_in, t_out, z_pos) in enumerate(gear_specs):
        # Input side gear
        gear_in = create_gear(
            outer_radius=r_in,
            inner_radius=0.025,
            thickness=0.018,
            num_teeth=t_in,
            tooth_height=0.006,
            segments=3,
        )
        gear_in = gear_in.transform(translation_matrix(-0.15, 0, z_pos))

        assembly.add_part(AssemblyPart(
            mesh=gear_in,
            name=f"Gear {i+1} Input",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([-0.4, 0.3 + i * 0.08, z_pos]),
            explosion_distance=0.3,
            color=get_color(i % 4),
        ))

        # Output side gear
        gear_out = create_gear(
            outer_radius=r_out,
            inner_radius=0.025,
            thickness=0.018,
            num_teeth=t_out,
            tooth_height=0.006,
            segments=3,
        )
        gear_out = gear_out.transform(translation_matrix(0.15, 0, z_pos))

        assembly.add_part(AssemblyPart(
            mesh=gear_out,
            name=f"Gear {i+1} Output",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.4, 0.3 + i * 0.08, z_pos]),
            explosion_distance=0.3,
            color=get_color((i + 2) % 4),
        ))

    # Synchronizers
    for i, z_pos in enumerate([-0.12, 0.02, 0.16]):
        synchro = create_smooth_cylinder(radius=0.04, height=0.025, segments=24)
        synchro = synchro.transform(translation_matrix(0.15, 0, z_pos))

        assembly.add_part(AssemblyPart(
            mesh=synchro,
            name=f"Synchronizer {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.6, 0.2, z_pos * 0.5]),
            explosion_distance=0.25,
            color=(180, 150, 50),
        ))

    # Shift forks
    for i, z_pos in enumerate([-0.12, 0.02, 0.16]):
        fork = create_smooth_cylinder(radius=0.01, height=0.1, segments=12)
        fork = fork.transform(translation_matrix(0.15, 0.08, z_pos))

        assembly.add_part(AssemblyPart(
            mesh=fork,
            name=f"Shift Fork {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.3, 0.8, 0.0]),
            explosion_distance=0.3,
            color=(150, 150, 155),
        ))

    # Bearings
    bearing_positions = [
        (-0.15, 0, 0.22), (-0.15, 0, -0.22),
        (0.15, 0, 0.22), (0.15, 0, -0.22),
    ]

    for i, (bx, by, bz) in enumerate(bearing_positions):
        bearing = create_bearing(
            outer_radius=0.04,
            inner_radius=0.025,
            width=0.015,
            num_balls=8,
            ball_radius=0.005,
            segments=20,
        )
        bearing = bearing.transform(translation_matrix(bx, by, bz))

        assembly.add_part(AssemblyPart(
            mesh=bearing,
            name=f"Shaft Bearing {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([bx * 2, 0.3, bz]),
            explosion_distance=0.25,
            color=(200, 180, 50),
        ))

    return assembly


# Update the factory
REALISTIC_ASSEMBLY_CREATORS = {
    "realistic_engine": create_realistic_engine_assembly,
    "realistic_turbofan": create_realistic_turbofan_assembly,
    "realistic_gearbox": create_realistic_gearbox_assembly,
}


def create_realistic_assembly(name: str) -> Assembly:
    """Create a realistic assembly by name."""
    if name not in REALISTIC_ASSEMBLY_CREATORS:
        raise ValueError(f"Unknown: {name}. Available: {list(REALISTIC_ASSEMBLY_CREATORS.keys())}")
    return REALISTIC_ASSEMBLY_CREATORS[name]()


def get_available_realistic_assemblies() -> list:
    """Get list of available realistic assemblies."""
    return list(REALISTIC_ASSEMBLY_CREATORS.keys())
