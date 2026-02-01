"""
Webcam input source using OpenCV.
"""

import threading
import time
from typing import Optional

import numpy as np

from config import settings
from src.input.base import InputSource


# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Webcam input disabled.")


class WebcamInput(InputSource):
    """
    Webcam input using OpenCV VideoCapture.

    Captures frames from the default camera in a background thread.
    """

    def __init__(self, device_id: int = None):
        super().__init__("webcam")

        if device_id is None:
            device_id = settings.WEBCAM_DEVICE_ID

        self.device_id = device_id
        self.cap = None
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None

        self.width = settings.WEBCAM_WIDTH
        self.height = settings.WEBCAM_HEIGHT
        self.fps = settings.WEBCAM_FPS

    def start(self) -> bool:
        """Start webcam capture."""
        if not CV2_AVAILABLE:
            print("OpenCV not available. Cannot start webcam.")
            return False

        if self.running:
            return True

        try:
            self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                print(f"Failed to open webcam device {self.device_id}")
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Get actual resolution
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.running = True

            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self.capture_thread.start()

            print(f"Webcam started: {self.width}x{self.height}")
            return True

        except Exception as e:
            print(f"Error starting webcam: {e}")
            return False

    def stop(self) -> None:
        """Stop webcam capture."""
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        print("Webcam stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (RGB)."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def _capture_loop(self):
        """Background thread for capturing frames."""
        while self.running and self.cap is not None:
            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    with self.lock:
                        self.current_frame = frame_rgb
                else:
                    # Short sleep on read failure
                    import time
                    time.sleep(0.01)

            except Exception as e:
                print(f"Webcam capture error: {e}")
                import time
                time.sleep(0.1)

    def set_resolution(self, width: int, height: int):
        """Change capture resolution (requires restart)."""
        self.width = width
        self.height = height

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def is_webcam_available() -> bool:
    """Check if webcam is available."""
    if not CV2_AVAILABLE:
        return False

    try:
        cap = cv2.VideoCapture(settings.WEBCAM_DEVICE_ID)
        available = cap.isOpened()
        cap.release()
        return available
    except Exception:
        return False
