"""Geometry module - 3D shapes and mesh operations."""

from src.geometry.mesh import Mesh
from src.geometry.primitives import (
    create_cube,
    create_sphere,
    create_cylinder,
    create_cone,
    create_torus,
    create_plane,
    create_pyramid,
    create_shape,
    SHAPE_CREATORS,
)
from src.geometry.transforms import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    translation_matrix,
    scale_matrix,
    transform_vertices,
)
from src.geometry.assembly import Assembly, AssemblyPart
from src.geometry.procedural_assemblies import (
    create_assembly,
    get_available_assemblies,
    create_engine_assembly,
    create_gearbox_assembly,
    create_watch_assembly,
)

# Advanced parametric surfaces
from src.geometry.advanced_surfaces import (
    create_klein_bottle,
    create_mobius_strip,
    create_superquadric,
    create_trefoil_knot,
    create_seashell,
    create_boy_surface,
    create_enneper_surface,
    create_dini_surface,
    create_roman_surface,
    create_catenoid,
    create_helicoid,
    create_spring,
    create_icosahedron,
    create_dodecahedron,
    create_octahedron,
    create_tetrahedron,
    create_geodesic_sphere,
    create_advanced_surface,
    subdivide_mesh,
    ADVANCED_SURFACE_CREATORS,
)

# Scientific visualization models
from src.geometry.scientific_models import (
    create_dna_helix,
    create_molecule,
    create_crystal_lattice,
    create_atomic_orbital,
    create_magnetic_field,
    create_virus_model,
    create_protein_structure,
    create_scientific_model,
    SCIENTIFIC_MODEL_CREATORS,
)

# Complex mechanical assemblies
from src.geometry.complex_assemblies import (
    create_jet_engine_assembly,
    create_robotic_arm_assembly,
    create_satellite_assembly,
    create_microscope_assembly,
    create_differential_assembly,
    create_complex_assembly,
    get_available_complex_assemblies,
    COMPLEX_ASSEMBLY_CREATORS,
)

# Architectural and engineering models
from src.geometry.architectural_models import (
    create_suspension_bridge_assembly,
    create_space_station_assembly,
    create_wind_turbine_assembly,
    create_terrain_mesh,
    create_skyscraper,
    create_geodesic_dome,
    create_architectural_assembly,
    create_architectural_mesh,
    get_available_architectural_assemblies,
    get_available_architectural_meshes,
    ARCHITECTURAL_ASSEMBLY_CREATORS,
    ARCHITECTURAL_MESH_CREATORS,
)

__all__ = [
    "Mesh",
    "Assembly",
    "AssemblyPart",
    # Basic primitives
    "create_cube",
    "create_sphere",
    "create_cylinder",
    "create_cone",
    "create_torus",
    "create_plane",
    "create_pyramid",
    "create_shape",
    "SHAPE_CREATORS",
    # Transforms
    "rotation_matrix_x",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "translation_matrix",
    "scale_matrix",
    "transform_vertices",
    # Procedural assemblies
    "create_assembly",
    "get_available_assemblies",
    "create_engine_assembly",
    "create_gearbox_assembly",
    "create_watch_assembly",
    # Advanced parametric surfaces
    "create_klein_bottle",
    "create_mobius_strip",
    "create_superquadric",
    "create_trefoil_knot",
    "create_seashell",
    "create_boy_surface",
    "create_enneper_surface",
    "create_dini_surface",
    "create_roman_surface",
    "create_catenoid",
    "create_helicoid",
    "create_spring",
    "create_icosahedron",
    "create_dodecahedron",
    "create_octahedron",
    "create_tetrahedron",
    "create_geodesic_sphere",
    "create_advanced_surface",
    "subdivide_mesh",
    "ADVANCED_SURFACE_CREATORS",
    # Scientific models
    "create_dna_helix",
    "create_molecule",
    "create_crystal_lattice",
    "create_atomic_orbital",
    "create_magnetic_field",
    "create_virus_model",
    "create_protein_structure",
    "create_scientific_model",
    "SCIENTIFIC_MODEL_CREATORS",
    # Complex assemblies
    "create_jet_engine_assembly",
    "create_robotic_arm_assembly",
    "create_satellite_assembly",
    "create_microscope_assembly",
    "create_differential_assembly",
    "create_complex_assembly",
    "get_available_complex_assemblies",
    "COMPLEX_ASSEMBLY_CREATORS",
    # Architectural models
    "create_suspension_bridge_assembly",
    "create_space_station_assembly",
    "create_wind_turbine_assembly",
    "create_terrain_mesh",
    "create_skyscraper",
    "create_geodesic_dome",
    "create_architectural_assembly",
    "create_architectural_mesh",
    "get_available_architectural_assemblies",
    "get_available_architectural_meshes",
    "ARCHITECTURAL_ASSEMBLY_CREATORS",
    "ARCHITECTURAL_MESH_CREATORS",
]
