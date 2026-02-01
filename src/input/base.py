"""
Abstract base class for input sources.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class InputSource(ABC):
    """
    Abstract base class for input sources.

    Input sources provide frames for ML processing.
    """

    def __init__(self, name: str):
        self.name = name
        self.running = False
        self.width = 0
        self.height = 0

    @abstractmethod
    def start(self) -> bool:
        """
        Start the input source.

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the input source."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame.

        Returns:
            RGB numpy array (H, W, 3) or None if not available
        """
        pass

    def is_running(self) -> bool:
        """Check if input source is running."""
        return self.running

    def get_resolution(self) -> tuple:
        """Get frame resolution (width, height)."""
        return (self.width, self.height)
