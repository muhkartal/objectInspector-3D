"""
Procedural assembly generators for demo purposes.
Creates Tesla-style engineering demo models using existing primitives.
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


def create_engine_assembly() -> Assembly:
    """
    Create an engine block assembly (~12 parts).

    Parts:
    - Engine block
    - Cylinder head
    - 4 pistons
    - Crankshaft
    - Oil pan
    - Intake manifold
    - Exhaust manifold
    """
    assembly = Assembly("4-Cylinder Engine")

    # Engine block
    block = create_cube(size=0.8)
    block = block.transform(scale_matrix(1.0, 0.8, 0.6))

    assembly.add_part(AssemblyPart(
        mesh=block,
        name="Engine Block",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=get_color(7),  # Silver
    ))

    # Cylinder head
    head = create_cube(size=0.75)
    head = head.transform(scale_matrix(1.0, 0.3, 0.55))
    head = head.transform(translation_matrix(0, 0.45, 0))

    assembly.add_part(AssemblyPart(
        mesh=head,
        name="Cylinder Head",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.8,
        color=get_color(0),  # Blue
    ))

    # Pistons (4 cylinders)
    piston_spacing = 0.18
    for i in range(4):
        px = (i - 1.5) * piston_spacing

        piston = create_cylinder(radius=0.06, height=0.15, segments=12)
        piston = piston.transform(translation_matrix(px, 0.1, 0))

        assembly.add_part(AssemblyPart(
            mesh=piston,
            name=f"Piston {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.0, 1.0, 0.0]),
            explosion_distance=0.5 + i * 0.1,
            color=get_color(3),  # Gold
        ))

    # Crankshaft
    crankshaft = create_cylinder(radius=0.04, height=0.9, segments=10)
    crankshaft = crankshaft.transform(rotation_matrix_z(np.pi / 2))
    crankshaft = crankshaft.transform(translation_matrix(0, -0.35, 0))

    assembly.add_part(AssemblyPart(
        mesh=crankshaft,
        name="Crankshaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.6,
        color=get_color(2),  # Teal
    ))

    # Oil pan
    pan = create_cube(size=0.7)
    pan = pan.transform(scale_matrix(1.0, 0.2, 0.5))
    pan = pan.transform(translation_matrix(0, -0.5, 0))

    assembly.add_part(AssemblyPart(
        mesh=pan,
        name="Oil Pan",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.8,
        color=get_color(7),  # Silver
    ))

    # Intake manifold
    intake = create_cube(size=0.4)
    intake = intake.transform(scale_matrix(1.0, 0.2, 0.3))
    intake = intake.transform(translation_matrix(0, 0.35, 0.35))

    assembly.add_part(AssemblyPart(
        mesh=intake,
        name="Intake Manifold",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.5, 1.0]),
        explosion_distance=0.6,
        color=get_color(0),  # Blue
    ))

    # Exhaust manifold
    exhaust = create_cylinder(radius=0.05, height=0.6, segments=8)
    exhaust = exhaust.transform(rotation_matrix_z(np.pi / 2))
    exhaust = exhaust.transform(translation_matrix(0, 0.1, -0.35))

    assembly.add_part(AssemblyPart(
        mesh=exhaust,
        name="Exhaust Manifold",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.3, -1.0]),
        explosion_distance=0.7,
        color=get_color(1),  # Red
    ))

    return assembly


def create_gearbox_assembly() -> Assembly:
    """
    Create a gearbox assembly (~12 parts).

    Parts:
    - Housing (2 halves)
    - Input shaft with gear
    - Output shaft with gear
    - Layshaft with gears
    - Bearings
    """
    assembly = Assembly("Manual Gearbox")

    # Housing bottom half
    housing_bottom = create_cube(size=0.6)
    housing_bottom = housing_bottom.transform(scale_matrix(1.2, 0.4, 0.8))
    housing_bottom = housing_bottom.transform(translation_matrix(0, -0.15, 0))

    assembly.add_part(AssemblyPart(
        mesh=housing_bottom,
        name="Housing Bottom",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.5,
        color=get_color(7),  # Silver
    ))

    # Housing top half
    housing_top = create_cube(size=0.6)
    housing_top = housing_top.transform(scale_matrix(1.2, 0.35, 0.8))
    housing_top = housing_top.transform(translation_matrix(0, 0.2, 0))

    assembly.add_part(AssemblyPart(
        mesh=housing_top,
        name="Housing Top",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.6,
        color=get_color(7),  # Silver
    ))

    # Input shaft
    input_shaft = create_cylinder(radius=0.04, height=0.5, segments=10)
    input_shaft = input_shaft.transform(rotation_matrix_x(np.pi / 2))
    input_shaft = input_shaft.transform(translation_matrix(-0.25, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=input_shaft,
        name="Input Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 1.0]),
        explosion_distance=0.6,
        color=get_color(2),  # Teal
    ))

    # Input gear
    input_gear = create_torus(major_radius=0.12, minor_radius=0.04,
                              major_segments=20, minor_segments=8)
    input_gear = input_gear.transform(translation_matrix(-0.25, 0, 0.1))

    assembly.add_part(AssemblyPart(
        mesh=input_gear,
        name="Input Gear",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-0.5, 0.5, 0.5]),
        explosion_distance=0.5,
        color=get_color(0),  # Blue
    ))

    # Output shaft
    output_shaft = create_cylinder(radius=0.04, height=0.5, segments=10)
    output_shaft = output_shaft.transform(rotation_matrix_x(np.pi / 2))
    output_shaft = output_shaft.transform(translation_matrix(0.25, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=output_shaft,
        name="Output Shaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, -1.0]),
        explosion_distance=0.6,
        color=get_color(2),  # Teal
    ))

    # Output gears (3 speeds)
    gear_positions = [-0.1, 0.0, 0.1]
    gear_sizes = [0.15, 0.12, 0.09]

    for i, (pz, size) in enumerate(zip(gear_positions, gear_sizes)):
        gear = create_torus(major_radius=size, minor_radius=0.03,
                           major_segments=20, minor_segments=8)
        gear = gear.transform(translation_matrix(0.25, 0, pz))

        assembly.add_part(AssemblyPart(
            mesh=gear,
            name=f"Output Gear {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([0.5, 0.3 + i * 0.2, -0.3]),
            explosion_distance=0.4 + i * 0.15,
            color=get_color(3 + i),  # Gold, Purple, Green
        ))

    # Layshaft
    layshaft = create_cylinder(radius=0.03, height=0.45, segments=10)
    layshaft = layshaft.transform(rotation_matrix_x(np.pi / 2))
    layshaft = layshaft.transform(translation_matrix(0, -0.08, 0))

    assembly.add_part(AssemblyPart(
        mesh=layshaft,
        name="Layshaft",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.5, 0.5]),
        explosion_distance=0.5,
        color=get_color(2),  # Teal
    ))

    # Bearings (4)
    bearing_positions = [
        (-0.25, 0, 0.25),
        (-0.25, 0, -0.2),
        (0.25, 0, 0.25),
        (0.25, 0, -0.2),
    ]

    for i, (bx, by, bz) in enumerate(bearing_positions):
        bearing = create_torus(major_radius=0.05, minor_radius=0.015,
                              major_segments=12, minor_segments=6)
        bearing = bearing.transform(translation_matrix(bx, by, bz))

        assembly.add_part(AssemblyPart(
            mesh=bearing,
            name=f"Bearing {i+1}",
            base_position=np.array([0.0, 0.0, 0.0]),
            explosion_direction=np.array([bx * 2, 0.5, bz]),
            explosion_distance=0.3,
            color=get_color(1),  # Red
        ))

    return assembly


def create_watch_assembly() -> Assembly:
    """
    Create a mechanical watch assembly (~10 parts).

    Parts:
    - Case back
    - Case middle
    - Movement plate
    - Main spring barrel
    - Escape wheel
    - Balance wheel
    - Crown
    - Crystal
    - Dial
    - Hands
    """
    assembly = Assembly("Mechanical Watch")

    # Case back
    case_back = create_cylinder(radius=0.4, height=0.05, segments=24)
    case_back = case_back.transform(translation_matrix(0, -0.15, 0))

    assembly.add_part(AssemblyPart(
        mesh=case_back,
        name="Case Back",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -1.0, 0.0]),
        explosion_distance=0.4,
        color=get_color(7),  # Silver
    ))

    # Case middle (ring)
    case_middle = create_torus(major_radius=0.38, minor_radius=0.06,
                               major_segments=32, minor_segments=12)
    case_middle = case_middle.transform(rotation_matrix_x(np.pi / 2))

    assembly.add_part(AssemblyPart(
        mesh=case_middle,
        name="Case Middle",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 0.0, 0.0]),
        explosion_distance=0.0,
        color=get_color(7),  # Silver
    ))

    # Movement plate
    movement = create_cylinder(radius=0.35, height=0.02, segments=20)
    movement = movement.transform(translation_matrix(0, -0.08, 0))

    assembly.add_part(AssemblyPart(
        mesh=movement,
        name="Movement Plate",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, -0.5, 0.0]),
        explosion_distance=0.25,
        color=get_color(3),  # Gold
    ))

    # Main spring barrel
    barrel = create_cylinder(radius=0.1, height=0.06, segments=16)
    barrel = barrel.transform(translation_matrix(-0.12, -0.05, 0))

    assembly.add_part(AssemblyPart(
        mesh=barrel,
        name="Mainspring Barrel",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([-0.5, -0.3, 0.0]),
        explosion_distance=0.3,
        color=get_color(0),  # Blue
    ))

    # Escape wheel
    escape = create_torus(major_radius=0.06, minor_radius=0.01,
                          major_segments=16, minor_segments=6)
    escape = escape.transform(rotation_matrix_x(np.pi / 2))
    escape = escape.transform(translation_matrix(0.1, -0.03, 0.1))

    assembly.add_part(AssemblyPart(
        mesh=escape,
        name="Escape Wheel",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.3, -0.3, 0.3]),
        explosion_distance=0.25,
        color=get_color(1),  # Red
    ))

    # Balance wheel
    balance = create_torus(major_radius=0.08, minor_radius=0.012,
                           major_segments=20, minor_segments=6)
    balance = balance.transform(rotation_matrix_x(np.pi / 2))
    balance = balance.transform(translation_matrix(0.15, 0, -0.1))

    assembly.add_part(AssemblyPart(
        mesh=balance,
        name="Balance Wheel",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.5, 0.3, -0.3]),
        explosion_distance=0.35,
        color=get_color(2),  # Teal
    ))

    # Crown
    crown = create_cylinder(radius=0.04, height=0.06, segments=10)
    crown = crown.transform(rotation_matrix_z(np.pi / 2))
    crown = crown.transform(translation_matrix(0.42, 0, 0))

    assembly.add_part(AssemblyPart(
        mesh=crown,
        name="Crown",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([1.0, 0.0, 0.0]),
        explosion_distance=0.3,
        color=get_color(3),  # Gold
    ))

    # Crystal (glass)
    crystal = create_cylinder(radius=0.36, height=0.02, segments=24)
    crystal = crystal.transform(translation_matrix(0, 0.1, 0))

    assembly.add_part(AssemblyPart(
        mesh=crystal,
        name="Crystal",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.35,
        color=(200, 220, 255),  # Light blue for glass
    ))

    # Dial
    dial = create_cylinder(radius=0.34, height=0.01, segments=24)
    dial = dial.transform(translation_matrix(0, 0.05, 0))

    assembly.add_part(AssemblyPart(
        mesh=dial,
        name="Dial",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.2,
        color=(40, 40, 45),  # Dark dial
    ))

    # Hands (simplified as small cone)
    hands = create_cone(radius=0.02, height=0.25, segments=6)
    hands = hands.transform(rotation_matrix_z(np.pi / 2))
    hands = hands.transform(translation_matrix(0, 0.07, 0))

    assembly.add_part(AssemblyPart(
        mesh=hands,
        name="Hands",
        base_position=np.array([0.0, 0.0, 0.0]),
        explosion_direction=np.array([0.0, 1.0, 0.0]),
        explosion_distance=0.25,
        color=get_color(0),  # Blue
    ))

    return assembly


# Factory for accessing assemblies by name
ASSEMBLY_CREATORS = {
    "engine": create_engine_assembly,
    "gearbox": create_gearbox_assembly,
    "watch": create_watch_assembly,
}


def create_assembly(name: str) -> Assembly:
    """
    Create an assembly by name.

    Args:
        name: Assembly name (engine, gearbox, watch, or complex/architectural)

    Returns:
        Assembly object

    Raises:
        ValueError: If assembly name is not recognized
    """
    # First check local creators
    if name in ASSEMBLY_CREATORS:
        return ASSEMBLY_CREATORS[name]()

    # Try complex assemblies
    try:
        from src.geometry.complex_assemblies import COMPLEX_ASSEMBLY_CREATORS
        if name in COMPLEX_ASSEMBLY_CREATORS:
            return COMPLEX_ASSEMBLY_CREATORS[name]()
    except ImportError:
        pass

    # Try architectural assemblies
    try:
        from src.geometry.architectural_models import ARCHITECTURAL_ASSEMBLY_CREATORS
        if name in ARCHITECTURAL_ASSEMBLY_CREATORS:
            return ARCHITECTURAL_ASSEMBLY_CREATORS[name]()
    except ImportError:
        pass

    # Try realistic assemblies
    try:
        from src.geometry.realistic_assemblies import REALISTIC_ASSEMBLY_CREATORS
        if name in REALISTIC_ASSEMBLY_CREATORS:
            return REALISTIC_ASSEMBLY_CREATORS[name]()
    except ImportError:
        pass

    # If not found anywhere, raise error
    all_assemblies = get_available_assemblies()
    raise ValueError(
        f"Unknown assembly: {name}. Available: {all_assemblies}"
    )


def get_available_assemblies() -> list:
    """Get list of all available assembly names from all modules."""
    assemblies = list(ASSEMBLY_CREATORS.keys())

    try:
        from src.geometry.complex_assemblies import COMPLEX_ASSEMBLY_CREATORS
        assemblies.extend(COMPLEX_ASSEMBLY_CREATORS.keys())
    except ImportError:
        pass

    try:
        from src.geometry.architectural_models import ARCHITECTURAL_ASSEMBLY_CREATORS
        assemblies.extend(ARCHITECTURAL_ASSEMBLY_CREATORS.keys())
    except ImportError:
        pass

    try:
        from src.geometry.realistic_assemblies import REALISTIC_ASSEMBLY_CREATORS
        assemblies.extend(REALISTIC_ASSEMBLY_CREATORS.keys())
    except ImportError:
        pass

    return assemblies
