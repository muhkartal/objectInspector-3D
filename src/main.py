"""
Object Inspector 3D - Main Application

A pygame-based 3D object inspection tool with ML capabilities.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pygame
import numpy as np
from typing import Dict, Any, Optional

from config import settings
from src.geometry.primitives import create_shape, SHAPE_CREATORS
from src.geometry.transforms import rotation_matrix_y, rotation_matrix_x
from src.geometry.assembly import Assembly
from src.geometry.procedural_assemblies import create_assembly, get_available_assemblies
from src.rendering.camera import Camera
from src.rendering.renderer import Renderer
from src.rendering.label_renderer import LabelRenderer
from src.visualization.manager import VisualizerManager
from src.effects.post_processing import PostProcessor
from src.input.synthetic import SyntheticInput
from src.input.webcam import WebcamInput, is_webcam_available
from src.ui.panel import UIPanel
from src.loaders.model_loader import create_assembly_from_model, is_supported_format

class ObjectInspector3D:
    """
    Main application class for 3D Object Inspector.

    Integrates all components:
    - 3D rendering with pygame
    - Multiple visualization modes
    - Camera controls (orbit, pan, zoom)
    - ML inference (classification, depth, pose, features)
    - Post-processing effects
    """

    def __init__(self):
        # Core components
        self.renderer: Optional[Renderer] = None
        self.camera: Optional[Camera] = None
        self.visualizer_manager: Optional[VisualizerManager] = None
        self.post_processor: Optional[PostProcessor] = None
        self.ui_panel: Optional[UIPanel] = None

        # Input sources
        self.synthetic_input: Optional[SyntheticInput] = None
        self.webcam_input: Optional[WebcamInput] = None
        self.use_webcam = False

        # Current state
        self.current_shape_name = settings.DEFAULT_SHAPE
        self.current_mesh = None
        self.shape_rotation = 0.0
        self.auto_rotate = True

        # Available shapes
        self.shape_names = list(SHAPE_CREATORS.keys())

        # Assembly state for exploded view
        self.current_assembly: Optional[Assembly] = None
        self.label_renderer: Optional[LabelRenderer] = None
        self.assembly_names = get_available_assemblies()
        self.current_assembly_index = 0

        # Application state
        self.running = False

        # Cached rotated mesh
        self._cached_rotation = 0.0
        self._cached_rotated_mesh = None

    def setup(self):
        """Initialize all components."""
        print("Initializing Object Inspector 3D...")

        # Initialize pygame
        pygame.init()

        # Create renderer
        self.renderer = Renderer()
        print(f"Window: {self.renderer.width}x{self.renderer.height}")

        # Create camera
        self.camera = Camera()

        # Create visualizer manager
        self.visualizer_manager = VisualizerManager()

        # Create post-processor
        self.post_processor = PostProcessor(
            self.renderer.width, self.renderer.height
        )
        self.post_processor.set_preset(settings.DEFAULT_EFFECT_PRESET)

        # Create UI panel
        self.ui_panel = UIPanel()

        # Setup explosion slider
        self.ui_panel.setup_explosion_slider(self._on_explosion_change)

        # Create label renderer for exploded view
        self.label_renderer = LabelRenderer()

        # Set label renderer on exploded visualizer
        exploded_viz = self.visualizer_manager.get_exploded_visualizer()
        if exploded_viz:
            exploded_viz.set_label_renderer(self.label_renderer)

        # Create initial shape
        self._create_shape(self.current_shape_name)

        # Setup input sources
        self._setup_input_sources()

        # Setup callbacks
        self._setup_callbacks()

        print("Initialization complete!")

    def _create_shape(self, shape_name: str):
        """Create a new shape mesh."""
        if shape_name not in SHAPE_CREATORS:
            print(f"Unknown shape: {shape_name}")
            return

        self.current_shape_name = shape_name
        self.current_mesh = create_shape(shape_name)
        self.current_mesh.set_color(settings.SHAPE_COLOR)

        # Invalidate cached rotated mesh
        self._cached_rotated_mesh = None

        print(f"Created shape: {shape_name}")

    def _setup_input_sources(self):
        """Initialize input sources."""
        # Synthetic input (always available)
        self.synthetic_input = SyntheticInput()
        self.synthetic_input.start()

        # Webcam input (optional)
        if settings.WEBCAM_ENABLED and is_webcam_available():
            self.webcam_input = WebcamInput()
            print("Webcam available")
        else:
            print("Webcam not available")

    def _setup_callbacks(self):
        """Setup renderer callbacks."""
        self.renderer.on_quit = self._on_quit
        self.renderer.on_key = self._on_key
        self.renderer.on_resize = self._on_resize

    def _on_quit(self):
        """Handle quit event."""
        self.running = False

    def _on_key(self, key: int, mod: int):
        """Handle key press."""
        # Shape selection (1-9)
        if pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            if idx < len(self.shape_names):
                self._create_shape(self.shape_names[idx])

        # Tab - cycle view mode
        elif key == pygame.K_TAB:
            self.visualizer_manager.cycle_mode()
            print(f"View mode: {self.visualizer_manager.current_mode}")

        # C - toggle webcam/synthetic
        elif key == pygame.K_c:
            self._toggle_input_source()

        # G - toggle glow
        elif key == pygame.K_g:
            self.post_processor.toggle_glow()
            print(f"Glow: {'on' if self.post_processor.glow_enabled else 'off'}")

        # V - toggle vignette
        elif key == pygame.K_v:
            self.post_processor.toggle_vignette()
            print(f"Vignette: {'on' if self.post_processor.vignette_enabled else 'off'}")

        # R - reset camera
        elif key == pygame.K_r:
            self.camera.reset()
            self.shape_rotation = 0.0
            print("Camera reset")

        # H - toggle help
        elif key == pygame.K_h:
            self.ui_panel.toggle_help()

        # Space - toggle auto-rotate
        elif key == pygame.K_SPACE:
            self.auto_rotate = not self.auto_rotate
            print(f"Auto-rotate: {'on' if self.auto_rotate else 'off'}")

        # E - toggle exploded view mode
        elif key == pygame.K_e:
            self._toggle_exploded_view()

        # F1-F12 - load assemblies
        elif key == pygame.K_F1:
            self._load_demo_assembly("engine")
        elif key == pygame.K_F2:
            self._load_demo_assembly("engine")
        elif key == pygame.K_F3:
            self._load_demo_assembly("gearbox")
        elif key == pygame.K_F4:
            self._load_demo_assembly("watch")
        elif key == pygame.K_F5:
            self._load_demo_assembly("jet_engine")
        elif key == pygame.K_F6:
            self._load_demo_assembly("robotic_arm")
        elif key == pygame.K_F7:
            self._load_demo_assembly("satellite")
        elif key == pygame.K_F8:
            self._load_demo_assembly("microscope")
        elif key == pygame.K_F9:
            self._load_demo_assembly("differential")
        elif key == pygame.K_F10:
            self._load_demo_assembly("suspension_bridge")
        elif key == pygame.K_F11:
            self._load_demo_assembly("space_station")
        elif key == pygame.K_F12:
            self._load_demo_assembly("wind_turbine")

        # 8 - load engine assembly (legacy)
        elif key == pygame.K_8:
            self._load_demo_assembly("engine")

        # 9 - load gearbox assembly (legacy)
        elif key == pygame.K_9:
            self._load_demo_assembly("gearbox")

        # 0 - load watch assembly (legacy)
        elif key == pygame.K_0:
            self._load_demo_assembly("watch")

        # A - cycle through all assemblies
        elif key == pygame.K_a:
            self._cycle_assembly()

        # Shift+F keys for realistic assemblies
        elif key == pygame.K_F1 and mod & pygame.KMOD_SHIFT:
            self._load_demo_assembly("realistic_engine")
        elif key == pygame.K_F2 and mod & pygame.KMOD_SHIFT:
            self._load_demo_assembly("realistic_turbofan")
        elif key == pygame.K_F3 and mod & pygame.KMOD_SHIFT:
            self._load_demo_assembly("realistic_gearbox")

        # L - load model file
        elif key == pygame.K_l:
            self._open_file_dialog()

    def _on_resize(self, width: int, height: int):
        """Handle window resize."""
        self.post_processor.resize(width, height)
        self.visualizer_manager.on_resize(width, height)

    def _toggle_input_source(self):
        """Toggle between webcam and synthetic input."""
        if self.webcam_input is None:
            print("Webcam not available")
            return

        if self.use_webcam:
            self.webcam_input.stop()
            self.use_webcam = False
            print("Using synthetic input")
        else:
            if self.webcam_input.start():
                self.use_webcam = True
                print("Using webcam input")
            else:
                print("Failed to start webcam")

    def _on_explosion_change(self, value: float):
        """Handle explosion slider value change."""
        self.visualizer_manager.set_explosion_factor(value)

    def _toggle_exploded_view(self):
        """Toggle exploded view mode."""
        if self.current_assembly is None:
            # Load default assembly if none loaded
            self._load_demo_assembly("engine")

        if self.visualizer_manager.current_mode == "exploded":
            # Switch back to solid view
            self.visualizer_manager.set_mode("solid")
            self.ui_panel.set_explosion_slider_visible(False)
        else:
            # Switch to exploded view
            self.visualizer_manager.set_mode("exploded")
            self.ui_panel.set_explosion_slider_visible(True)
        print(f"View mode: {self.visualizer_manager.current_mode}")

    def _load_demo_assembly(self, name: str):
        """Load a demo assembly by name."""
        try:
            assembly = create_assembly(name)
            self._load_assembly(assembly)
            # Update index to match loaded assembly
            if name in self.assembly_names:
                self.current_assembly_index = self.assembly_names.index(name)
            print(f"Loaded assembly: {name} ({assembly.part_count} parts)")
        except ValueError as e:
            print(f"Error loading assembly: {e}")

    def _cycle_assembly(self):
        """Cycle through available assemblies."""
        self.current_assembly_index = (self.current_assembly_index + 1) % len(self.assembly_names)
        name = self.assembly_names[self.current_assembly_index]
        self._load_demo_assembly(name)

    def _load_assembly(self, assembly: Assembly):
        """Load an assembly and switch to exploded view."""
        self.current_assembly = assembly
        self.visualizer_manager.set_assembly(assembly)

        # Update UI
        self.ui_panel.set_assembly_info(assembly.name, assembly.part_count)
        self.ui_panel.set_explosion_value(assembly.explosion_factor)
        self.ui_panel.set_explosion_slider_visible(True)

        # Switch to exploded view
        self.visualizer_manager.set_mode("exploded")

    def _open_file_dialog(self):
        """Open file dialog to load a model file."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            # Create hidden root window
            root = tk.Tk()
            root.withdraw()

            # Open file dialog
            filepath = filedialog.askopenfilename(
                title="Load 3D Model",
                filetypes=[
                    ("3D Models", "*.obj *.gltf *.glb"),
                    ("OBJ files", "*.obj"),
                    ("GLTF files", "*.gltf *.glb"),
                    ("All files", "*.*"),
                ]
            )

            root.destroy()

            if filepath and is_supported_format(filepath):
                try:
                    assembly = create_assembly_from_model(filepath)
                    self._load_assembly(assembly)
                    print(f"Loaded model: {filepath}")
                except Exception as e:
                    print(f"Error loading model: {e}")
            elif filepath:
                print(f"Unsupported file format: {filepath}")

        except ImportError:
            print("tkinter not available for file dialog")
        except Exception as e:
            print(f"Error opening file dialog: {e}")

    def _handle_events(self) -> bool:
        """Handle all events including slider."""
        # Peek at events for slider without consuming
        events = pygame.event.get()

        # First pass: slider handles relevant events
        for event in events:
            self.ui_panel.handle_slider_event(event)

        # Re-queue all events for renderer
        for event in events:
            pygame.event.post(event)

        # Now let renderer handle all events
        return self.renderer.handle_events(self.camera)

    def _get_rotated_mesh(self):
        """Get rotated mesh with caching for performance."""
        if self.current_mesh is None:
            return None

        # Only recompute if rotation changed significantly
        if (self._cached_rotated_mesh is None or
            abs(self.shape_rotation - self._cached_rotation) > 0.001):
            rotation = rotation_matrix_y(self.shape_rotation)
            self._cached_rotated_mesh = self.current_mesh.transform(rotation)
            self._cached_rotation = self.shape_rotation

        return self._cached_rotated_mesh

    def _get_state(self) -> Dict[str, Any]:
        """Get current application state for UI."""
        return {
            "fps": self.renderer.fps,
            "shape": self.current_shape_name,
            "mode": self.visualizer_manager.current_mode,
            "input_source": "webcam" if self.use_webcam else "synthetic",
            "effects": self.post_processor.get_state(),
            "camera": {
                "distance": self.camera.distance,
                "yaw": self.camera.yaw,
                "pitch": self.camera.pitch,
            },
        }

    def run(self):
        """Main application loop."""
        self.setup()
        self.running = True

        print("\nObject Inspector 3D is running!")
        print("Press H for help, Esc to quit.\n")

        while self.running:
            # Handle events
            if not self._handle_events():
                break

            # Auto-rotate shape
            if self.auto_rotate:
                self.shape_rotation += 0.01

            # Transform mesh for rotation (with caching)
            rotated_mesh = self._get_rotated_mesh()

            # Update explosion slider display to match current value
            if self.visualizer_manager.current_mode == "exploded":
                current_explosion = self.visualizer_manager.get_explosion_factor()
                self.ui_panel.set_explosion_value(current_explosion)

            # Clear screen
            self.renderer.clear()

            # Draw grid and axes
            self.renderer.draw_grid(self.camera)
            self.renderer.draw_axes(self.camera)

            # Update and draw visualization
            if rotated_mesh:
                self.visualizer_manager.update(rotated_mesh, self.camera)
                self.visualizer_manager.draw(
                    self.renderer.surface,
                    rotated_mesh,
                    self.camera,
                )

            # Apply post-processing
            self.post_processor.apply(self.renderer.surface)

            # Draw UI
            self.ui_panel.draw(self.renderer.surface, self._get_state())

            # Present frame
            self.renderer.present()

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nShutting down...")

        # Stop input sources
        if self.synthetic_input:
            self.synthetic_input.stop()
        if self.webcam_input:
            self.webcam_input.stop()

        # Cleanup renderer
        if self.renderer:
            self.renderer.cleanup()

        print("Goodbye!")


def main():
    """Entry point."""
    app = ObjectInspector3D()
    app.run()


if __name__ == "__main__":
    main()
