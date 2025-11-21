from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, binary_closing

from .config import PipelineConfig


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image intensity to 0-255."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def gabor_filter(image: np.ndarray, frequency: float = 0.1, theta: float = 0) -> np.ndarray:
    """Apply Gabor filter to enhance ridges.
    
    For fingerprints, we apply multiple orientations and take the maximum response.
    """
    rows, cols = image.shape
    enhanced = np.zeros_like(image, dtype=np.float32)
    
    # Apply Gabor filters at multiple orientations
    for angle in np.arange(0, np.pi, np.pi / 8):  # 8 orientations
        kernel = cv2.getGaborKernel(
            (21, 21),  # kernel size
            5.0,  # sigma
            angle,  # theta (orientation)
            2 * np.pi * frequency,  # lambda (wavelength)
            0.5,  # gamma (aspect ratio)
            0,  # psi (phase offset)
            cv2.CV_32F
        )
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        enhanced = np.maximum(enhanced, np.abs(filtered))
    
    # Normalize to 0-255
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)


def binarize_image(image: np.ndarray) -> np.ndarray:
    """Binarize image using Otsu's method."""
    
    threshold = threshold_otsu(image)
    binary = (image > threshold).astype(np.uint8) * 255
    return binary


def thin_image(binary_image: np.ndarray) -> np.ndarray:
    """
    Improve thinning by removing noise + closing small gaps + stable skeleton.
    Input must be a binary image (0 or 255).
    """

    # Ensure binary format: 0/1
    binary = (binary_image > 0).astype(np.uint8)

    # --- Noise reduction ---
    binary = cv2.medianBlur(binary, 5)

    # --- Morphological closing to remove small holes ---
    binary = binary_closing(binary, footprint=np.ones((5, 5)))
    binary = binary.astype(np.uint8)

    # --- Thinning ---
    skeleton = skeletonize(binary > 0)

    # Convert back to 0/255 for visualization
    skeleton = (skeleton.astype(np.uint8) * 255)

    return skeleton


def preprocess_image(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Complete preprocessing pipeline: Normalize -> Gabor -> Binarize -> Thin."""
    # Step 1: Normalize
    normalized = normalize_image(image)
    
    # Step 2: Gabor filtering
    enhanced = gabor_filter(normalized, frequency=0.1)
    
    # Step 3: Binarization
    binary = binarize_image(enhanced)
    
    # Step 4: Thinning
    thinned = thin_image(binary)
    
    return thinned


def load_and_preprocess(path: Path, config: PipelineConfig) -> np.ndarray:
    image = load_image(path)
    return preprocess_image(image, config)
