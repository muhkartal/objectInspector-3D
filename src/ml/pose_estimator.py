"""
3D pose estimation for objects.
"""

from typing import Any, Optional, Dict, Tuple

import numpy as np

from config import settings
from src.ml.base import MLModel
from src.ml.model_manager import load_onnx_model, is_onnx_available

# Try to import OpenCV for PnP solving
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class PoseEstimator(MLModel):
    """
    Estimates 3D pose (rotation and translation) of objects.

    Can use either:
    - Learned pose estimation (CNN)
    - Classical PnP solving with detected keypoints
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = f"models/pretrained/{settings.POSE_MODEL}"

        super().__init__("pose_estimator", model_path)

        # Input size
        self.input_size = (224, 224)

        # Camera intrinsics (default, should be calibrated)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(5)

        # Reference 3D points for PnP (cube corners)
        self.reference_points_3d = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ], dtype=np.float32)

    def load(self) -> bool:
        """Load pose estimation model."""
        if not is_onnx_available():
            print("PoseEstimator: ONNX Runtime not available")
            self.loaded = False
            return False

        self.model = load_onnx_model(self.model_path)
        if self.model is not None:
            self.loaded = True
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            return True

        # Failed to load
        print(f"PoseEstimator: Model not found at {self.model_path}")
        self.loaded = False
        return False

    def set_camera_intrinsics(self, width: int, height: int, fov: float = 60.0):
        """Set camera intrinsic matrix from image size and FOV."""
        fx = width / (2 * np.tan(np.radians(fov / 2)))
        fy = fx
        cx = width / 2
        cy = height / 2

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for pose estimation."""
        from PIL import Image

        self._original_size = (frame.shape[1], frame.shape[0])

        # Set camera intrinsics if not set
        if self.camera_matrix is None:
            self.set_camera_intrinsics(frame.shape[1], frame.shape[0])

        # Resize
        img = Image.fromarray(frame)
        img = img.resize(self.input_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std

        # Convert to NCHW
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def infer(self, input_tensor: np.ndarray) -> Any:
        """Run pose estimation inference."""
        if self.model is None:
            return None

        outputs = self.model.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]

    def _euler_to_rotation_matrix(
        self, yaw: float, pitch: float, roll: float
    ) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float32)

        return R

    def postprocess(self, output: Any) -> Optional[Dict[str, Any]]:
        """Process pose estimation output."""
        if output is None:
            return None

        if isinstance(output, dict):
            # Fallback output (if any legacy path remains)
            return output

        # Process model output
        # Assuming output is [rotation_6d, translation_3d] or similar

        # For now, return fallback format
        rotation = np.eye(3, dtype=np.float32)
        center = (self._original_size[0] / 2, self._original_size[1] / 2)

        return {
            "rotation": rotation,
            "center": center,
            "translation": np.zeros(3),
            "euler_angles": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
            },
            "confidence": 0.0,
            "scale": 50,  # Axis visualization scale
        }

    def estimate_from_keypoints(
        self,
        keypoints_2d: np.ndarray,
        keypoints_3d: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Estimate pose from 2D-3D keypoint correspondences using PnP.

        Args:
            keypoints_2d: Nx2 array of 2D image points
            keypoints_3d: Nx3 array of corresponding 3D points
                         (uses reference_points_3d if None)

        Returns:
            Pose estimation result
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available for PnP solving"}

        if self.camera_matrix is None:
            return {"error": "Camera intrinsics not set"}

        if keypoints_3d is None:
            keypoints_3d = self.reference_points_3d

        if len(keypoints_2d) < 4:
            return {"error": "Need at least 4 keypoints for PnP"}

        try:
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                keypoints_3d,
                keypoints_2d.astype(np.float32),
                self.camera_matrix,
                self.dist_coeffs,
            )

            if not success:
                return {"error": "PnP solving failed"}

            # Convert rotation vector to matrix
            rotation, _ = cv2.Rodrigues(rvec)

            return {
                "rotation": rotation,
                "translation": tvec.flatten(),
                "rvec": rvec.flatten(),
                "tvec": tvec.flatten(),
                "confidence": 1.0,
            }

        except Exception as e:
            return {"error": str(e)}

    def estimate(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to estimate pose.

        Args:
            frame: RGB image

        Returns:
            Pose estimation results
        """
        return self.process(frame)
