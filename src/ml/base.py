"""
Abstract base class for ML models with async processing.
"""

import time
import threading
from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Any, Dict, Optional

import numpy as np

from config import settings


class MLModel(ABC):
    """
    Abstract base class for ML models.

    Supports async processing in a background thread to avoid
    blocking the main render loop.
    """

    def __init__(self, name: str, model_path: str = None):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.loaded = False

        # Async processing
        self.async_enabled = settings.ML_ASYNC_PROCESSING
        self.input_queue: Queue = Queue(maxsize=2)
        self.output_queue: Queue = Queue(maxsize=2)
        self.running = False
        self.process_thread: Optional[threading.Thread] = None

        # Latest result cache
        self._latest_result: Optional[Any] = None
        self._result_lock = threading.Lock()

        # Processing stats
        self.inference_time = 0.0
        self.frame_count = 0

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model.

        Returns:
            True if loaded successfully
        """
        pass

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess input frame for model.

        Args:
            frame: RGB image array (H, W, 3)

        Returns:
            Preprocessed tensor ready for model input
        """
        pass

    @abstractmethod
    def postprocess(self, output: Any) -> Any:
        """
        Postprocess model output.

        Args:
            output: Raw model output

        Returns:
            Processed results
        """
        pass

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> Any:
        """
        Run model inference.

        Args:
            input_tensor: Preprocessed input

        Returns:
            Raw model output
        """
        pass

    def start(self) -> bool:
        """Start async processing thread."""
        if not self.loaded:
            if not self.load():
                return False

        if self.async_enabled and not self.running:
            self.running = True
            self.process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self.process_thread.start()
            print(f"ML model '{self.name}' started (async)")

        return True

    def stop(self):
        """Stop async processing."""
        self.running = False
        if self.process_thread:
            # Put None to unblock the thread
            try:
                self.input_queue.put_nowait(None)
            except:
                pass
            self.process_thread.join(timeout=1.0)
            self.process_thread = None
        print(f"ML model '{self.name}' stopped")

    def process(self, frame: np.ndarray) -> Optional[Any]:
        """
        Process a frame (sync or async depending on settings).

        Args:
            frame: RGB image array

        Returns:
            Results if sync, or cached result if async
        """
        if not self.loaded:
            return None

        if self.async_enabled:
            # Submit for async processing
            try:
                self.input_queue.put_nowait(frame)
            except:
                pass  # Queue full, skip frame

            # Return latest cached result
            return self.get_latest_result()
        else:
            # Sync processing
            return self._process_frame(frame)

    def get_latest_result(self) -> Optional[Any]:
        """Get the most recent processing result."""
        # Check for new results
        try:
            while True:
                result = self.output_queue.get_nowait()
                with self._result_lock:
                    self._latest_result = result
        except Empty:
            pass

        with self._result_lock:
            return self._latest_result

    def _process_loop(self):
        """Background processing loop."""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                if frame is None:
                    continue

                result = self._process_frame(frame)

                # Put result in output queue (discard old if full)
                try:
                    self.output_queue.put_nowait(result)
                except:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(result)
                    except:
                        pass

            except Empty:
                continue
            except Exception as e:
                print(f"ML processing error in {self.name}: {e}")

    def _process_frame(self, frame: np.ndarray) -> Any:
        """Process a single frame (internal)."""
        start = time.time()

        # Preprocess
        input_tensor = self.preprocess(frame)

        # Inference
        output = self.infer(input_tensor)

        # Postprocess
        result = self.postprocess(output)

        # Track stats
        self.inference_time = time.time() - start
        self.frame_count += 1

        return result

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.loaded

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "name": self.name,
            "loaded": self.loaded,
            "inference_time_ms": self.inference_time * 1000,
            "frame_count": self.frame_count,
        }
