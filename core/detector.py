"""Face detection and alignment using InsightFace."""

import logging
from typing import List
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from core.models import DetectedFace
from utils.image import create_thumbnail, validate_image

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces, extracts aligned face crops and normalized embeddings using InsightFace."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: List[str] | None = None,
    ):
        """Initializes the FaceAnalysis pipeline.

        Args:
            model_name: InsightFace model pack name (default: "buffalo_l").
            det_size: Image resize dimensions for detection.
            providers: ONNX execution providers list. Defaults to CoreML with CPU fallback.
        """
        if providers is None:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        logger.info(f"Loading InsightFace model '{model_name}' with providers: {providers}")
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.det_size = det_size

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detects all faces in an RGB image, sorted by size descending.

        Args:
            image: RGB image as a uint8 NumPy array.

        Returns:
            list[DetectedFace]: List of detected faces with aligned crops,
                               display thumbnails, and normalized embeddings.
        """
        rgb_image = validate_image(image)

        # InsightFace expects BGR format (standard OpenCV format)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        raw_faces = self.app.get(bgr_image)
        if not raw_faces:
            return []

        detected_faces = []
        for face in raw_faces:
            # Bounding box
            x1, y1, x2, y2 = [int(round(coord)) for coord in face.bbox]
            bbox = (x1, y1, x2, y2)
            score = float(face.det_score)

            # ⚡ Bolt Optimization: Avoid expensive and redundant `face_align.norm_crop`
            # and `cv2.cvtColor` operations. The `aligned_face` attribute is part of the
            # data contract but is not used downstream (UI uses `thumbnail`, comparator uses
            # `embedding`). Returning an empty array saves processing time and reduces
            # memory payload size when storing state in Gradio.
            aligned_rgb = np.empty((0,), dtype=np.uint8)

            # High-res display thumbnail for UI preview and galleries
            thumbnail_rgb = create_thumbnail(rgb_image, bbox, target_size=(256, 256), padding=0.35)

            # Normalized embedding vector
            if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                embedding = np.array(face.normed_embedding, dtype=np.float32)
            elif hasattr(face, "embedding") and face.embedding is not None:
                raw_emb = np.array(face.embedding, dtype=np.float32)
                norm = np.linalg.norm(raw_emb)
                embedding = raw_emb / (norm + 1e-10)
            else:
                continue

            detected_faces.append(
                DetectedFace(
                    bbox=bbox,
                    score=score,
                    aligned_face=aligned_rgb,
                    thumbnail=thumbnail_rgb,
                    embedding=embedding,
                )
            )

        # Sort detected faces by bounding box area (largest / most prominent first)
        detected_faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        return detected_faces
