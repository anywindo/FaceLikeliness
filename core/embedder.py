"""Face embedding extraction and normalization."""

import numpy as np
from core.models import DetectedFace


class FaceEmbedder:
    """Manages facial feature representation extraction and normalization."""

    def extract(self, face: DetectedFace) -> np.ndarray:
        """Extracts and verifies an L2-normalized 512-dimensional face embedding.

        Args:
            face: A DetectedFace instance containing an embedding.

        Returns:
            np.ndarray: Normalized 1D float32 vector of length 512.

        Raises:
            ValueError: If the embedding is invalid or empty.
        """
        if face is None or face.embedding is None:
            raise ValueError("Invalid face: No facial embedding available.")

        embedding = np.asarray(face.embedding, dtype=np.float32).flatten()

        if embedding.size == 0:
            raise ValueError("Face embedding vector is empty.")

        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding
