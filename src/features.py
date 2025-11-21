from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2

from .config import PipelineConfig


@dataclass
class Minutiae:
    """Represents a single minutiae point."""
    x: int
    y: int
    orientation: float  # in radians
    type: str  # 'ending' or 'bifurcation'


@dataclass
class MinutiaeExtractor:
    """Extracts minutiae from thinned fingerprint images using Crossing Number method."""
    
    def extract(self, thinned_image: np.ndarray) -> List[Minutiae]:
        """Extract minutiae using Crossing Number (CN) method.
        
        CN = 1 -> Ridge ending
        CN = 3 -> Ridge bifurcation
        """
        minutiae_list = []
        rows, cols = thinned_image.shape
        
        # Convert to binary (0 or 1)
        binary = (thinned_image > 128).astype(np.uint8)
        
        # 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if binary[y, x] == 0:  # Background pixel
                    continue
                
                # Calculate Crossing Number (CN)
                cn = 0
                for i in range(8):
                    ny, nx = y + neighbors[i][0], x + neighbors[i][1]
                    next_i = (i + 1) % 8
                    nny, nnx = y + neighbors[next_i][0], x + neighbors[next_i][1]
                    
                    # Count transitions from 0 to 1
                    if binary[ny, nx] == 0 and binary[nny, nnx] == 1:
                        cn += 1
                
                # CN = 1: Ridge ending
                # CN = 3: Ridge bifurcation
                if cn == 1:
                    orientation = self._calculate_orientation(binary, x, y)
                    minutiae_list.append(Minutiae(x, y, orientation, 'ending'))
                elif cn == 3:
                    orientation = self._calculate_orientation(binary, x, y)
                    minutiae_list.append(Minutiae(x, y, orientation, 'bifurcation'))
        
        # Remove minutiae too close to image borders
        margin = 10
        minutiae_list = [
            m for m in minutiae_list
            if margin < m.x < cols - margin and margin < m.y < rows - margin
        ]
        
        return minutiae_list
    
    def _calculate_orientation(self, binary: np.ndarray, x: int, y: int) -> float:
        """
        More robust orientation using image gradients.
        """
        img = binary.astype(float)
        grad_y, grad_x = np.gradient(img)

        gx = grad_x[y, x] if 0 <= y < img.shape[0] and 0 <= x < img.shape[1] else 0
        gy = grad_y[y, x] if 0 <= y < img.shape[0] and 0 <= x < img.shape[1] else 0

        angle = np.arctan2(gy, gx)
        return float(angle)


def build_feature_extractor(config: PipelineConfig) -> MinutiaeExtractor:
    """Build minutiae extractor."""
    return MinutiaeExtractor()
