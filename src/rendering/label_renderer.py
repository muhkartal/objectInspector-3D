"""
Tesla-style callout label renderer for exploded view.
Renders anchor dots, leader lines, and rounded label backgrounds.
"""

from typing import List, Tuple, Optional
import math

import pygame
import numpy as np

from config import settings


class LabelRenderer:
    """
    Renders Tesla-style callout labels with leader lines.

    Features:
    - Anchor dots on parts
    - Leader lines with optional 45-degree elbows
    - Rounded rectangle label backgrounds
    - Label collision avoidance
    """

    def __init__(self):
        # Font
        self._font: Optional[pygame.font.Font] = None

        # Styling from settings
        self.background_color = settings.LABEL_BACKGROUND_COLOR
        self.text_color = settings.LABEL_TEXT_COLOR
        self.leader_color = settings.LABEL_LEADER_COLOR
        self.dot_color = settings.LABEL_DOT_COLOR
        self.dot_radius = settings.LABEL_DOT_RADIUS
        self.padding = settings.LABEL_PADDING
        self.border_radius = settings.LABEL_BORDER_RADIUS

        # Label offset distance from anchor point
        self.label_offset = 60

        # Minimum spacing between labels
        self.min_label_spacing = 25

    def _ensure_font(self) -> None:
        """Initialize font if needed."""
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", settings.LABEL_FONT_SIZE)

    def draw_labels(
        self,
        surface: pygame.Surface,
        label_data: List[Tuple[str, Tuple[int, int], np.ndarray]],
    ) -> None:
        """
        Draw all labels with leader lines.

        Args:
            surface: Pygame surface to draw on
            label_data: List of (name, screen_pos, world_pos) tuples
        """
        self._ensure_font()

        if not label_data:
            return

        width, height = surface.get_size()

        # Calculate label positions with collision avoidance
        label_positions = self._calculate_label_positions(label_data, width, height)

        # Draw leader lines first (behind labels)
        for (name, anchor_pos, _), label_pos in zip(label_data, label_positions):
            if label_pos is not None:
                self._draw_leader_line(surface, anchor_pos, label_pos)

        # Draw anchor dots
        for name, anchor_pos, _ in label_data:
            self._draw_anchor_dot(surface, anchor_pos)

        # Draw labels
        for (name, anchor_pos, _), label_pos in zip(label_data, label_positions):
            if label_pos is not None:
                self._draw_label(surface, name, label_pos)

    def _calculate_label_positions(
        self,
        label_data: List[Tuple[str, Tuple[int, int], np.ndarray]],
        width: int,
        height: int,
    ) -> List[Optional[Tuple[int, int]]]:
        """
        Calculate label positions with collision avoidance.

        Returns:
            List of (x, y) positions for label centers
        """
        positions = []
        used_rects = []

        for name, anchor_pos, world_pos in label_data:
            # Get label size
            text_surface = self._font.render(name, True, self.text_color)
            label_width = text_surface.get_width() + self.padding[0] * 2
            label_height = text_surface.get_height() + self.padding[1] * 2

            # Try to find a position
            best_pos = self._find_label_position(
                anchor_pos, label_width, label_height,
                used_rects, width, height
            )

            if best_pos:
                positions.append(best_pos)
                # Add to used rects
                label_rect = pygame.Rect(
                    best_pos[0] - label_width // 2,
                    best_pos[1] - label_height // 2,
                    label_width,
                    label_height
                )
                used_rects.append(label_rect)
            else:
                positions.append(None)

        return positions

    def _find_label_position(
        self,
        anchor_pos: Tuple[int, int],
        label_width: int,
        label_height: int,
        used_rects: List[pygame.Rect],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a non-overlapping position for a label.

        Tries positions in order: right, top-right, top, top-left, left, etc.
        """
        ax, ay = anchor_pos

        # Directions to try (in order of preference)
        directions = [
            (1, 0),      # Right
            (1, -0.5),   # Top-right
            (0.7, -0.7), # Upper-right diagonal
            (0, -1),     # Top
            (-0.7, -0.7),# Upper-left diagonal
            (-1, -0.5),  # Top-left
            (-1, 0),     # Left
            (-1, 0.5),   # Bottom-left
            (0, 1),      # Bottom
            (1, 0.5),    # Bottom-right
        ]

        for dx, dy in directions:
            # Calculate potential position
            px = int(ax + dx * self.label_offset)
            py = int(ay + dy * self.label_offset)

            # Create label rect at this position
            label_rect = pygame.Rect(
                px - label_width // 2,
                py - label_height // 2,
                label_width,
                label_height
            )

            # Check if within screen bounds (with margin)
            margin = 10
            if (label_rect.left < margin or
                label_rect.right > width - margin or
                label_rect.top < margin or
                label_rect.bottom > height - margin):
                continue

            # Check for overlaps with existing labels
            overlaps = False
            for used_rect in used_rects:
                if label_rect.inflate(
                    self.min_label_spacing, self.min_label_spacing
                ).colliderect(used_rect):
                    overlaps = True
                    break

            if not overlaps:
                return (px, py)

        # If no position found, try farther out
        for distance in [self.label_offset * 1.5, self.label_offset * 2]:
            for dx, dy in directions:
                px = int(ax + dx * distance)
                py = int(ay + dy * distance)

                label_rect = pygame.Rect(
                    px - label_width // 2,
                    py - label_height // 2,
                    label_width,
                    label_height
                )

                margin = 10
                if (label_rect.left < margin or
                    label_rect.right > width - margin or
                    label_rect.top < margin or
                    label_rect.bottom > height - margin):
                    continue

                overlaps = False
                for used_rect in used_rects:
                    if label_rect.inflate(
                        self.min_label_spacing, self.min_label_spacing
                    ).colliderect(used_rect):
                        overlaps = True
                        break

                if not overlaps:
                    return (px, py)

        # Fallback: return offset right position even if it overlaps
        return (ax + self.label_offset, ay)

    def _draw_anchor_dot(
        self,
        surface: pygame.Surface,
        pos: Tuple[int, int],
    ) -> None:
        """Draw the anchor dot on the part."""
        # Outer ring
        pygame.draw.circle(
            surface, self.dot_color, pos, self.dot_radius + 2, 1
        )
        # Inner dot
        pygame.draw.circle(
            surface, self.dot_color, pos, self.dot_radius
        )

    def _draw_leader_line(
        self,
        surface: pygame.Surface,
        anchor_pos: Tuple[int, int],
        label_pos: Tuple[int, int],
    ) -> None:
        """Draw leader line with optional elbow."""
        ax, ay = anchor_pos
        lx, ly = label_pos

        # Calculate distance
        dx = lx - ax
        dy = ly - ay

        # Check if we need an elbow (for more horizontal lines)
        if abs(dx) > abs(dy) * 1.5:
            # Add elbow point
            elbow_x = ax + dx * 0.3
            elbow_y = ay + dy * 0.7

            # Draw from anchor to elbow
            pygame.draw.line(surface, self.leader_color, anchor_pos,
                           (int(elbow_x), int(elbow_y)), 1)
            # Draw from elbow to label
            pygame.draw.line(surface, self.leader_color,
                           (int(elbow_x), int(elbow_y)), label_pos, 1)
        else:
            # Direct line
            pygame.draw.line(surface, self.leader_color, anchor_pos, label_pos, 1)

    def _draw_label(
        self,
        surface: pygame.Surface,
        text: str,
        pos: Tuple[int, int],
    ) -> None:
        """Draw label with rounded background."""
        # Render text
        text_surface = self._font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect()

        # Calculate background rect
        bg_width = text_rect.width + self.padding[0] * 2
        bg_height = text_rect.height + self.padding[1] * 2

        bg_rect = pygame.Rect(
            pos[0] - bg_width // 2,
            pos[1] - bg_height // 2,
            bg_width,
            bg_height
        )

        # Draw background with transparency
        bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)

        # Draw rounded rectangle
        pygame.draw.rect(
            bg_surface,
            self.background_color,
            pygame.Rect(0, 0, bg_width, bg_height),
            border_radius=self.border_radius
        )

        surface.blit(bg_surface, bg_rect.topleft)

        # Draw text centered
        text_x = pos[0] - text_rect.width // 2
        text_y = pos[1] - text_rect.height // 2
        surface.blit(text_surface, (text_x, text_y))

    def draw_single_label(
        self,
        surface: pygame.Surface,
        text: str,
        anchor_pos: Tuple[int, int],
        label_pos: Tuple[int, int],
    ) -> None:
        """Draw a single label with leader line."""
        self._ensure_font()

        self._draw_leader_line(surface, anchor_pos, label_pos)
        self._draw_anchor_dot(surface, anchor_pos)
        self._draw_label(surface, text, label_pos)
