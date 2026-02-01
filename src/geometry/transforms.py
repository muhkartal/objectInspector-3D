"""
3D transformation matrices and operations.
"""

import numpy as np


def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around the X axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around the Y axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around the Z axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotation_matrix_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around an arbitrary axis.

    Args:
        axis: 3D axis vector (will be normalized)
        angle: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis

    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotation_matrix_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix from Euler angles (YXZ order).

    Args:
        yaw: Rotation around Y axis (radians)
        pitch: Rotation around X axis (radians)
        roll: Rotation around Z axis (radians)

    Returns:
        4x4 rotation matrix
    """
    return rotation_matrix_y(yaw) @ rotation_matrix_x(pitch) @ rotation_matrix_z(roll)


def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Create a 4x4 translation matrix.

    Args:
        tx, ty, tz: Translation amounts

    Returns:
        4x4 translation matrix
    """
    return np.array(
        [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def scale_matrix(sx: float, sy: float = None, sz: float = None) -> np.ndarray:
    """
    Create a 4x4 scale matrix.

    Args:
        sx: Scale factor for X (or uniform scale if sy/sz not provided)
        sy: Scale factor for Y (optional)
        sz: Scale factor for Z (optional)

    Returns:
        4x4 scale matrix
    """
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx

    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create a view matrix that looks from eye towards target.

    Args:
        eye: Camera position
        target: Point to look at
        up: Up vector

    Returns:
        4x4 view matrix
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Forward vector (from target to eye for right-handed coordinate system)
    forward = eye - target
    forward = forward / np.linalg.norm(forward)

    # Right vector
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    # Recalculate up vector
    up = np.cross(forward, right)

    # Create view matrix
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up, eye)
    view[2, 3] = -np.dot(forward, eye)

    return view


def perspective_matrix(
    fov: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """
    Create a perspective projection matrix.

    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio (width / height)
        near: Near clipping plane
        far: Far clipping plane

    Returns:
        4x4 perspective projection matrix
    """
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2)

    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def orthographic_matrix(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """
    Create an orthographic projection matrix.

    Args:
        left, right: Left and right clipping planes
        bottom, top: Bottom and top clipping planes
        near, far: Near and far clipping planes

    Returns:
        4x4 orthographic projection matrix
    """
    return np.array(
        [
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to a set of vertices.

    Args:
        vertices: Nx3 array of vertex positions
        matrix: 4x4 transformation matrix

    Returns:
        Nx3 array of transformed vertices
    """
    # Convert to homogeneous coordinates
    ones = np.ones((len(vertices), 1), dtype=np.float32)
    vertices_h = np.hstack([vertices, ones])

    # Apply transformation
    transformed = (matrix @ vertices_h.T).T

    # Convert back to 3D (handle perspective division if w != 1)
    w = transformed[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1, w)  # Avoid division by zero
    return transformed[:, :3] / w


def transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply transformation to normal vectors (uses inverse transpose).

    Args:
        normals: Nx3 array of normal vectors
        matrix: 4x4 transformation matrix

    Returns:
        Nx3 array of transformed normals (normalized)
    """
    # Use inverse transpose of upper 3x3 for normals
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    transformed = (normal_matrix @ normals.T).T

    # Normalize
    lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    return transformed / lengths


def compose_matrices(*matrices: np.ndarray) -> np.ndarray:
    """
    Compose multiple transformation matrices (multiply in order).

    Args:
        *matrices: Variable number of 4x4 matrices

    Returns:
        4x4 composed transformation matrix
    """
    result = np.eye(4, dtype=np.float32)
    for m in matrices:
        result = result @ m
    return result
