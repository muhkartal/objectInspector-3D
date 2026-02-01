"""
ML Model manager - loads and manages ONNX models.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from config import settings

from src.ml.base import MLModel

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. ML features will be limited.")


class ModelManager:
    """
    Manages loading and running of ML models.

    Handles model file discovery, loading, and inference routing.
    """

    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.models_dir = Path("models/pretrained")

        # ONNX session options
        self.session_options = None
        if ONNX_AVAILABLE:
            self.session_options = ort.SessionOptions()
            self.session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get full path to a model file."""
        # Check multiple possible locations
        paths_to_check = [
            self.models_dir / model_name,
            self.models_dir / f"{model_name}.onnx",
            Path(model_name),
        ]

        for path in paths_to_check:
            if path.exists():
                return path

        return None

    def register_model(self, name: str, model: MLModel):
        """Register a model instance."""
        self.models[name] = model

    def load_model(self, name: str) -> bool:
        """Load a registered model."""
        if name not in self.models:
            print(f"Model '{name}' not registered")
            return False

        return self.models[name].load()

    def load_all(self) -> Dict[str, bool]:
        """Load all registered models."""
        results = {}
        for name in self.models:
            results[name] = self.load_model(name)
        return results

    def start_all(self):
        """Start async processing for all models."""
        for model in self.models.values():
            model.start()

    def stop_all(self):
        """Stop all models."""
        for model in self.models.values():
            model.stop()

    def process(self, name: str, frame: np.ndarray) -> Optional[Any]:
        """Process frame with a specific model."""
        if name not in self.models:
            return None
        return self.models[name].process(frame)

    def process_all(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame with all loaded models."""
        results = {}
        for name, model in self.models.items():
            if model.is_loaded():
                result = model.process(frame)
                if result is not None:
                    results[name] = result
        return results

    def get_model(self, name: str) -> Optional[MLModel]:
        """Get a model instance."""
        return self.models.get(name)

    def list_available_models(self) -> list:
        """List available model files."""
        if not self.models_dir.exists():
            return []

        return [f.stem for f in self.models_dir.glob("*.onnx")]

    def get_stats(self) -> Dict[str, dict]:
        """Get stats for all models."""
        return {name: model.get_stats() for name, model in self.models.items()}


def load_onnx_model(model_path: str) -> Optional[Any]:
    """
    Load an ONNX model.

    Args:
        model_path: Path to .onnx file

    Returns:
        ONNX InferenceSession or None
    """
    if not ONNX_AVAILABLE:
        print("ONNX Runtime not available")
        return None

    path = Path(model_path)
    if not path.exists():
        print(f"Model file not found: {model_path}")
        return None

    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        print(f"Loaded ONNX model: {path.name}")
        return session

    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None


def is_onnx_available() -> bool:
    """Check if ONNX Runtime is available."""
    return ONNX_AVAILABLE
