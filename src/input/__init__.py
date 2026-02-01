"""Input sources module - webcam and synthetic input."""

from src.input.base import InputSource
from src.input.webcam import WebcamInput
from src.input.synthetic import SyntheticInput

__all__ = ["InputSource", "WebcamInput", "SyntheticInput"]
