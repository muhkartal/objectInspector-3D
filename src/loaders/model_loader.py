"""
External model loaders for OBJ and GLTF/GLB files.
Auto-calculates explosion directions from part positions.
"""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np

from config import settings
from src.geometry.mesh import Mesh
from src.geometry.assembly import Assembly, AssemblyPart


def load_obj(filepath: str) -> Tuple[List[Mesh], List[str]]:
    """
    Load an OBJ file and return meshes and group names.

    Supports:
    - Vertices (v)
    - Faces (f)
    - Groups (g/o)
    - Basic materials (usemtl for color hints)

    Args:
        filepath: Path to OBJ file

    Returns:
        Tuple of (list of meshes, list of group names)
    """
    vertices = []
    current_group = "default"
    groups: Dict[str, Dict] = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            cmd = parts[0]

            if cmd == 'v':
                # Vertex position
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])

            elif cmd == 'g' or cmd == 'o':
                # Group/object name
                current_group = parts[1] if len(parts) > 1 else "default"
                if current_group not in groups:
                    groups[current_group] = {'faces': [], 'material': None}

            elif cmd == 'usemtl':
                # Material (just store name for now)
                if current_group not in groups:
                    groups[current_group] = {'faces': [], 'material': None}
                groups[current_group]['material'] = parts[1] if len(parts) > 1 else None

            elif cmd == 'f':
                # Face
                if current_group not in groups:
                    groups[current_group] = {'faces': [], 'material': None}

                face_verts = []
                for vert_data in parts[1:]:
                    # Parse vertex index (v/vt/vn format)
                    indices = vert_data.split('/')
                    v_idx = int(indices[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(v_idx)

                # Triangulate if more than 3 vertices
                if len(face_verts) >= 3:
                    for i in range(1, len(face_verts) - 1):
                        groups[current_group]['faces'].append([
                            face_verts[0], face_verts[i], face_verts[i + 1]
                        ])

    # Convert to numpy
    vertices_array = np.array(vertices, dtype=np.float32)

    # Create meshes for each group
    meshes = []
    group_names = []

    for group_name, group_data in groups.items():
        if not group_data['faces']:
            continue

        faces_array = np.array(group_data['faces'], dtype=np.int32)

        # Create mesh
        mesh = Mesh(vertices=vertices_array.copy(), faces=faces_array)
        meshes.append(mesh)
        group_names.append(group_name)

    return meshes, group_names


def load_gltf(filepath: str) -> Tuple[List[Mesh], List[str]]:
    """
    Load a GLTF/GLB file and return meshes and names.

    Uses pygltflib for parsing.

    Args:
        filepath: Path to GLTF or GLB file

    Returns:
        Tuple of (list of meshes, list of mesh names)
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        print("pygltflib not installed. Run: pip install pygltflib")
        return [], []

    gltf = GLTF2().load(filepath)

    meshes = []
    mesh_names = []

    # Get base path for resolving buffers
    base_path = os.path.dirname(filepath)

    # Load binary data
    def get_buffer_data(buffer_index: int) -> bytes:
        buffer = gltf.buffers[buffer_index]
        if buffer.uri:
            if buffer.uri.startswith('data:'):
                # Embedded data
                import base64
                data_start = buffer.uri.index(',') + 1
                return base64.b64decode(buffer.uri[data_start:])
            else:
                # External file
                buffer_path = os.path.join(base_path, buffer.uri)
                with open(buffer_path, 'rb') as f:
                    return f.read()
        else:
            # GLB binary chunk
            return gltf.binary_blob()

    # Process each mesh in the GLTF
    for mesh_idx, gltf_mesh in enumerate(gltf.meshes):
        mesh_name = gltf_mesh.name or f"Mesh_{mesh_idx}"

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for primitive in gltf_mesh.primitives:
            # Get position accessor
            pos_accessor_idx = primitive.attributes.POSITION
            if pos_accessor_idx is None:
                continue

            pos_accessor = gltf.accessors[pos_accessor_idx]
            pos_buffer_view = gltf.bufferViews[pos_accessor.bufferView]
            buffer_data = get_buffer_data(pos_buffer_view.buffer)

            # Extract vertices
            byte_offset = (pos_buffer_view.byteOffset or 0) + (pos_accessor.byteOffset or 0)
            vertex_count = pos_accessor.count

            # Parse float32 vertices
            vertices = np.frombuffer(
                buffer_data[byte_offset:byte_offset + vertex_count * 12],
                dtype=np.float32
            ).reshape(-1, 3)

            all_vertices.append(vertices)

            # Get indices
            if primitive.indices is not None:
                idx_accessor = gltf.accessors[primitive.indices]
                idx_buffer_view = gltf.bufferViews[idx_accessor.bufferView]
                idx_buffer_data = get_buffer_data(idx_buffer_view.buffer)

                idx_byte_offset = (idx_buffer_view.byteOffset or 0) + (idx_accessor.byteOffset or 0)

                # Determine index type
                if idx_accessor.componentType == 5123:  # UNSIGNED_SHORT
                    indices = np.frombuffer(
                        idx_buffer_data[idx_byte_offset:idx_byte_offset + idx_accessor.count * 2],
                        dtype=np.uint16
                    )
                elif idx_accessor.componentType == 5125:  # UNSIGNED_INT
                    indices = np.frombuffer(
                        idx_buffer_data[idx_byte_offset:idx_byte_offset + idx_accessor.count * 4],
                        dtype=np.uint32
                    )
                else:
                    continue

                # Reshape to triangles
                faces = indices.reshape(-1, 3) + vertex_offset
                all_faces.append(faces)
            else:
                # No indices, assume triangle list
                faces = np.arange(vertex_count).reshape(-1, 3) + vertex_offset
                all_faces.append(faces)

            vertex_offset += vertex_count

        if all_vertices and all_faces:
            vertices_combined = np.vstack(all_vertices).astype(np.float32)
            faces_combined = np.vstack(all_faces).astype(np.int32)

            mesh = Mesh(vertices=vertices_combined, faces=faces_combined)
            meshes.append(mesh)
            mesh_names.append(mesh_name)

    return meshes, mesh_names


def load_model(filepath: str) -> Tuple[List[Mesh], List[str]]:
    """
    Load a 3D model file based on extension.

    Supports OBJ, GLTF, and GLB formats.

    Args:
        filepath: Path to model file

    Returns:
        Tuple of (list of meshes, list of names)
    """
    ext = Path(filepath).suffix.lower()

    if ext == '.obj':
        return load_obj(filepath)
    elif ext in ['.gltf', '.glb']:
        return load_gltf(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def create_assembly_from_model(
    filepath: str,
    name: Optional[str] = None,
    explosion_scale: float = 1.5,
) -> Assembly:
    """
    Load a model file and create an assembly with auto-calculated explosion directions.

    Args:
        filepath: Path to model file
        name: Optional assembly name (defaults to filename)
        explosion_scale: Scale factor for explosion distances

    Returns:
        Configured Assembly object
    """
    meshes, mesh_names = load_model(filepath)

    if not meshes:
        raise ValueError(f"No meshes found in {filepath}")

    # Use filename as assembly name if not provided
    if name is None:
        name = Path(filepath).stem

    assembly = Assembly(name)

    # Calculate overall center of all meshes
    centers = [mesh.center for mesh in meshes]
    overall_center = np.mean(centers, axis=0) if centers else np.zeros(3)

    # Get colors from palette
    colors = settings.ASSEMBLY_COLORS

    for i, (mesh, mesh_name) in enumerate(zip(meshes, mesh_names)):
        center = mesh.center

        # Calculate explosion direction (away from overall center)
        direction = center - overall_center
        length = np.linalg.norm(direction)

        if length < 0.01:
            # Part is at center, use upward direction
            direction = np.array([0.0, 1.0, 0.0])
        else:
            direction = direction / length

        # Calculate explosion distance based on mesh size
        distance = np.linalg.norm(mesh.size) * explosion_scale

        # Get color from palette
        color = colors[i % len(colors)]

        part = AssemblyPart(
            mesh=mesh,
            name=mesh_name,
            base_position=np.zeros(3),
            explosion_direction=direction,
            explosion_distance=distance,
            color=color
        )

        assembly.add_part(part)

    return assembly


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.obj', '.gltf', '.glb']


def is_supported_format(filepath: str) -> bool:
    """Check if a file format is supported."""
    ext = Path(filepath).suffix.lower()
    return ext in get_supported_extensions()
