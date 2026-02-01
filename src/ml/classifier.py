"""
Object classifier using CNN.
"""

from typing import Any, Optional, Dict, List

import numpy as np

from config import settings
from src.ml.base import MLModel
from src.ml.model_manager import load_onnx_model, is_onnx_available


class Classifier(MLModel):
    """
    CNN-based object classifier.

    Classifies shapes/objects in the input frame.
    Uses MobileNetV2 or similar lightweight architecture.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = f"models/pretrained/{settings.CLASSIFIER_MODEL}"

        super().__init__("classifier", model_path)

        # Classification settings
        self.labels = settings.SHAPE_LABELS
        self.input_size = (224, 224)  # Standard ImageNet size
        self.confidence_threshold = settings.ML_CONFIDENCE_THRESHOLD

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load(self) -> bool:
        """Load the classifier model."""
        if not is_onnx_available():
            print("Classifier: ONNX Runtime not available")
            self.loaded = False
            return False

        self.model = load_onnx_model(self.model_path)
        if self.model is not None:
            self.loaded = True
            # Get input/output names
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            return True

        # Failed to load
        print(f"Classifier: Model not found at {self.model_path}")
        self.loaded = False
        return False

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for classification.

        Args:
            frame: RGB image (H, W, 3), values 0-255

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        # Resize to input size
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize(self.input_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize
        arr = (arr - self.mean) / self.std

        # Convert to NCHW format
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def infer(self, input_tensor: np.ndarray) -> Any:
        """Run classification inference."""
        if self.model is None:
            return None

        outputs = self.model.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]

    def postprocess(self, output: Any) -> Optional[Dict[str, Any]]:
        """
        Convert logits to classification result.

        Returns:
            Dict with 'label', 'confidence', 'all_scores'
        """
        if output is None:
            return None

        logits = output[0] if isinstance(output, list) else output
        logits = logits.flatten()

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        # Get top prediction
        top_idx = probs.argmax()
        confidence = float(probs[top_idx])

        # Map to label
        if top_idx < len(self.labels):
            label = self.labels[top_idx]
        else:
            label = "unknown"

        # Get top-k predictions
        top_k = 3
        top_indices = probs.argsort()[-top_k:][::-1]
        top_predictions = [
            {"label": self.labels[i] if i < len(self.labels) else "unknown",
             "confidence": float(probs[i])}
            for i in top_indices
        ]

        return {
            "label": label,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "all_scores": probs.tolist(),
        }

    def classify(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to classify a frame.

        Args:
            frame: RGB image

        Returns:
            Classification result dict
        """
        return self.process(frame)
