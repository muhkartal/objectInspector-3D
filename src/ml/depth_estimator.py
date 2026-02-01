"""
Monocular depth estimation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import settings

from src.ml.base import MLModel
from src.ml.model_manager import load_onnx_model, is_onnx_available


class DepthEstimator(MLModel):
    """
    Monocular depth estimation using CNN.

    Estimates depth from a single RGB image.
    Based on MiDaS or similar architecture.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = f"models/pretrained/{settings.DEPTH_MODEL}"

        super().__init__("depth_estimator", model_path)

        # Input size (MiDaS small uses 256x256)
        self.input_size = (256, 256)

        # Output settings
        self.normalize_output = True
        self.invert_depth = True  # Closer = higher values

    def load(self) -> bool:
        """Load depth estimation model."""
        if not is_onnx_available():
            print("DepthEstimator: ONNX Runtime not available")
            self.loaded = False
            return False

        self.model = load_onnx_model(self.model_path)
        if self.model is not None:
            self.loaded = True
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

            # Get actual input size from model
            input_shape = self.model.get_inputs()[0].shape
            if len(input_shape) >= 4:
                self.input_size = (input_shape[2], input_shape[3])

            return True

        # Failed to load
        print(f"DepthEstimator: Model not found at {self.model_path}")
        self.loaded = False
        return False

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for depth estimation.

        Args:
            frame: RGB image (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        from PIL import Image

        # Store original size for output
        self._original_size = (frame.shape[1], frame.shape[0])

        # Resize
        img = Image.fromarray(frame)
        img = img.resize(self.input_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std

        # Convert to NCHW
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def infer(self, input_tensor: np.ndarray) -> Any:
        """Run depth estimation inference."""
        if self.model is None:
            return None

        outputs = self.model.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]

    def postprocess(self, output: Any) -> Optional[Dict[str, Any]]:
        """
        Process depth output.

        Returns:
            Dict with 'depth_map', 'min_depth', 'max_depth'
        """
        if output is None:
            return None

        depth = output[0] if isinstance(output, (list, tuple)) else output

        # Remove batch dimension if present
        if len(depth.shape) == 4:
            depth = depth[0]
        if len(depth.shape) == 3:
            depth = depth[0]

        # Resize to original size
        if hasattr(self, "_original_size"):
            from PIL import Image
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize(self._original_size, Image.BILINEAR)
            depth = np.array(depth_img)

        # Normalize to 0-1
        if self.normalize_output:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max - depth_min > 0:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)

        # Invert so closer = higher values
        if self.invert_depth:
            depth = 1.0 - depth

        return {
            "depth_map": depth,
            "min_depth": float(depth.min()),
            "max_depth": float(depth.max()),
            "mean_depth": float(depth.mean()),
        }

    def estimate(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to estimate depth.

        Args:
            frame: RGB image

        Returns:
            Depth estimation results
        """
        return self.process(frame)

    def get_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to colormap for visualization.

        Args:
            depth_map: 2D depth array (0-1)

        Returns:
            RGB colormap (H, W, 3)
        """
        # Jet-like colormap
        depth_normalized = np.clip(depth_map, 0, 1)

        # Create RGB channels
        r = np.clip(4 * depth_normalized - 2, 0, 1)
        g = np.clip(2 - 4 * np.abs(depth_normalized - 0.5), 0, 1)
        b = np.clip(2 - 4 * depth_normalized, 0, 1)

        colormap = np.stack([r, g, b], axis=-1)
        return (colormap * 255).astype(np.uint8)
