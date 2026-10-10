"""Data models for face detection and comparison results."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    """Represents a single detected face in an image.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        score: Detection confidence score from the model.
        aligned_face: 112x112 aligned face crop suitable for ArcFace input.
        thumbnail: Display-sized face crop for the UI gallery.
        embedding: 512-dimensional L2-normalized face embedding vector.
    """

    bbox: tuple[int, int, int, int]
    score: float
    aligned_face: np.ndarray
    thumbnail: np.ndarray
    embedding: np.ndarray


@dataclass
class ComparisonResult:
    """Result of comparing two face embeddings.

    Attributes:
        cosine_similarity: Raw cosine similarity value in range [-1, 1].
        similarity_percentage: Normalized score mapped to [0, 100] scale.
        category: Qualitative label — "Low", "Moderate", or "High".
        explanation: Human-readable interpretation with disclaimer.
        face_a_thumbnail: Display crop of the first face used in comparison.
        face_b_thumbnail: Display crop of the second face used in comparison.
    """

    cosine_similarity: float
    similarity_percentage: float
    category: str
    explanation: str
    face_a_thumbnail: np.ndarray
    face_b_thumbnail: np.ndarray
