"""
Feature detector - edges, corners, and keypoints.
"""

from typing import Any, Optional, Dict, List, Tuple

import numpy as np

from config import settings

from src.ml.base import MLModel

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class FeatureDetector(MLModel):
    """
    Detects features in images: edges, corners, keypoints.

    Uses a combination of classical CV (OpenCV) and optional
    learned feature detection.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = f"models/pretrained/{settings.FEATURE_MODEL}"

        super().__init__("feature_detector", model_path)

        # Detection settings
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        self.corner_quality = 0.01
        self.corner_min_distance = 10
        self.max_corners = 100
        self.max_keypoints = 500

        # ORB detector for keypoints
        self.orb = None
        if CV2_AVAILABLE:
            self.orb = cv2.ORB_create(nfeatures=self.max_keypoints)

    def load(self) -> bool:
        """Load feature detector (uses OpenCV, model optional)."""
        if not CV2_AVAILABLE:
            print("FeatureDetector: OpenCV not available, using fallback")

        self.loaded = True
        return True

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale for processing."""
        if len(frame.shape) == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                # Manual grayscale conversion
                gray = (
                    0.299 * frame[:, :, 0] +
                    0.587 * frame[:, :, 1] +
                    0.114 * frame[:, :, 2]
                ).astype(np.uint8)
        else:
            gray = frame

        return gray

    def infer(self, input_tensor: np.ndarray) -> Dict[str, Any]:
        """Detect features using OpenCV."""
        gray = input_tensor
        results = {}

        if CV2_AVAILABLE:
            # Edge detection (Canny)
            edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
            results["edges_mask"] = edges

            # Corner detection (Shi-Tomasi)
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.corner_quality,
                minDistance=self.corner_min_distance,
            )
            if corners is not None:
                results["corners"] = corners.reshape(-1, 2)
            else:
                results["corners"] = np.array([])

            # Keypoint detection (ORB)
            if self.orb is not None:
                keypoints = self.orb.detect(gray, None)
                results["keypoints"] = [
                    (kp.pt[0], kp.pt[1], kp.size, kp.angle)
                    for kp in keypoints
                ]
            else:
                results["keypoints"] = []

        else:
            # Fallback: simple edge detection
            results["edges_mask"] = self._simple_edge_detect(gray)
            results["corners"] = self._simple_corner_detect(gray)
            results["keypoints"] = []

        return results

    def _simple_edge_detect(self, gray: np.ndarray) -> np.ndarray:
        """Simple edge detection without OpenCV."""
        # Sobel-like edge detection
        h, w = gray.shape
        edges = np.zeros_like(gray)

        # Horizontal and vertical gradients
        gx = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16))
        gy = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16))

        # Combine
        edges[:, :-1] = np.maximum(edges[:, :-1], gx)
        edges[:-1, :] = np.maximum(edges[:-1, :], gy)

        # Threshold
        edges = (edges > 30).astype(np.uint8) * 255

        return edges

    def _simple_corner_detect(self, gray: np.ndarray) -> np.ndarray:
        """Simple corner detection without OpenCV."""
        # Very basic corner detection using gradient changes
        h, w = gray.shape
        corners = []

        # Sample grid
        step = 20
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                # Check gradient in multiple directions
                region = gray[y - 5:y + 5, x - 5:x + 5]
                if region.size == 0:
                    continue

                var = region.var()
                if var > 500:  # High variance indicates corner-like region
                    corners.append([x, y])

        return np.array(corners[:self.max_corners]) if corners else np.array([])

    def postprocess(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format detection results."""
        edges_mask = output.get("edges_mask")
        corners = output.get("corners", np.array([]))
        keypoints = output.get("keypoints", [])

        # Extract edge line segments (simplified)
        edge_lines = []
        if edges_mask is not None and CV2_AVAILABLE:
            # Use HoughLinesP for line detection
            lines = cv2.HoughLinesP(
                edges_mask,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10,
            )
            if lines is not None:
                edge_lines = [
                    ((int(l[0][0]), int(l[0][1])), (int(l[0][2]), int(l[0][3])))
                    for l in lines[:100]  # Limit number of lines
                ]

        return {
            "edges": edge_lines,
            "corners": corners.tolist() if isinstance(corners, np.ndarray) else corners,
            "keypoints": [
                {"x": kp[0], "y": kp[1], "size": kp[2] if len(kp) > 2 else 5}
                for kp in keypoints
            ],
            "num_corners": len(corners),
            "num_keypoints": len(keypoints),
            "num_edges": len(edge_lines),
        }

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to detect features.

        Args:
            frame: RGB image

        Returns:
            Detection results dict
        """
        return self.process(frame)
