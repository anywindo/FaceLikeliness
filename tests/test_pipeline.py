"""Automated test suite for Likeliness face detection and similarity comparison."""

import os
import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from unittest.mock import patch
import gradio as gr
from core.models import DetectedFace, ComparisonResult
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.comparator import FaceComparator
from utils.image import validate_image, create_thumbnail, compress_image, MAX_PIXELS
from utils.gauge import render_similarity_gauge
import app


class TestLikelinessPipeline(unittest.TestCase):
    """Test suite covering data models, preprocessing, embedding, and comparison."""

    def setUp(self):
        self.embedder = FaceEmbedder()
        self.comparator = FaceComparator()

    def test_image_validation_formats(self):
        """Tests that validate_image correctly converts various image formats to RGB uint8."""
        # Grayscale
        gray = np.full((120, 120), 128, dtype=np.uint8)
        rgb_from_gray = validate_image(gray)
        self.assertEqual(rgb_from_gray.shape, (120, 120, 3))
        self.assertEqual(rgb_from_gray.dtype, np.uint8)

        # RGBA
        rgba = np.full((120, 120, 4), 255, dtype=np.uint8)
        rgb_from_rgba = validate_image(rgba)
        self.assertEqual(rgb_from_rgba.shape, (120, 120, 3))

        # Empty image should raise ValueError
        with self.assertRaises(ValueError):
            validate_image(np.array([]))

        # None should raise ValueError
        with self.assertRaises(ValueError):
            validate_image(None)

        # Unsupported type (e.g. str / path) should raise ValueError
        with self.assertRaises(ValueError):
            validate_image("/etc/passwd")

    def test_image_compressor_and_validation(self):
        """Verifies compressor and validate_image enforce the 16 MP budget."""
        # Valid PIL Image
        pil_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        rgb = validate_image(pil_img)
        self.assertEqual(rgb.shape, (200, 200, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        # High-resolution PIL Image (e.g. 5000x5000 = 25MP) is compressed to <= 16MP
        highres_pil = Image.new("RGB", (5000, 5000))
        compressed_pil = compress_image(highres_pil, max_pixels=MAX_PIXELS)
        self.assertLessEqual(compressed_pil.width * compressed_pil.height, MAX_PIXELS)

        rgb_highres = validate_image(highres_pil)
        self.assertLessEqual(rgb_highres.shape[0] * rgb_highres.shape[1], MAX_PIXELS)
        self.assertEqual(rgb_highres.dtype, np.uint8)

        # High-resolution NumPy array (e.g. 5000x4000 = 20MP) is compressed to <= 16MP
        highres_arr = np.zeros((5000, 4000, 3), dtype=np.uint8)
        compressed_arr = compress_image(highres_arr, max_pixels=MAX_PIXELS)
        self.assertLessEqual(compressed_arr.shape[0] * compressed_arr.shape[1], MAX_PIXELS)

        rgb_arr = validate_image(highres_arr)
        self.assertLessEqual(rgb_arr.shape[0] * rgb_arr.shape[1], MAX_PIXELS)

    def test_compare_exception_handling(self):
        """Verifies that app.compare catches internal errors and returns a generic gr.Error."""
        dummy_face = DetectedFace(
            bbox=(0, 0, 10, 10),
            score=0.99,
            aligned_face=np.empty((0,), dtype=np.uint8),
            thumbnail=np.zeros((100, 100, 3), dtype=np.uint8),
            embedding=np.zeros(512, dtype=np.float32),
        )
        with patch.object(app.comparator, "compare", side_effect=RuntimeError("Internal math error")):
            with self.assertRaises(gr.Error) as ctx:
                app.compare(
                    status_a="Photo A ready",
                    status_b="Photo B ready",
                    faces_a=[dummy_face],
                    faces_b=[dummy_face],
                    idx_a=0,
                    idx_b=0,
                )
            self.assertIn("An error occurred during comparison", str(ctx.exception))

    def test_thumbnail_generation(self):
        """Tests padded thumbnail creation bounds and sizing."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        bbox = (50, 50, 150, 150)
        thumb = create_thumbnail(img, bbox, target_size=(200, 200), padding=0.2)
        self.assertEqual(thumb.shape, (200, 200, 3))

    def test_cosine_similarity_computation(self):
        """Verifies cosine similarity, percentage mapping, and threshold categorization."""
        # Identical vectors
        v1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        v2 = v1.copy()

        cos_sim = self.comparator.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(cos_sim, 1.0, places=4)

        score = self.comparator.normalize_score(cos_sim)
        self.assertAlmostEqual(score, 100.0, places=4)
        self.assertIn("High", self.comparator.categorize(cos_sim))

        # Look-alike anchor test: 0.22 cosine must map to 50.0%
        self.assertAlmostEqual(self.comparator.normalize_score(0.22), 50.0, places=1)

        # Opposite vectors (mathematical lower bound)
        v_opp = -v1
        cos_opp = self.comparator.compute_cosine_similarity(v1, v_opp)
        self.assertAlmostEqual(cos_opp, -1.0, places=4)
        self.assertAlmostEqual(self.comparator.normalize_score(cos_opp), 0.0, places=4)
        self.assertEqual(self.comparator.categorize(cos_opp), "Low Similarity")

        # Moderate threshold check
        cos_mod = 0.30
        self.assertIn("Moderate", self.comparator.categorize(cos_mod))

    def test_embedder_normalization(self):
        """Verifies that the embedder ensures unit length."""
        raw_emb = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)
        face = DetectedFace(
            bbox=(0, 0, 10, 10),
            score=0.99,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
            thumbnail=np.zeros((100, 100, 3), dtype=np.uint8),
            embedding=raw_emb,
        )
        norm_emb = self.embedder.extract(face)
        self.assertAlmostEqual(float(np.linalg.norm(norm_emb)), 1.0, places=5)

    def test_gauge_html_rendering(self):
        """Verifies visual SVG gauge renders correctly with safe parameters."""
        html = render_similarity_gauge(82.4, "High Similarity", 0.648)
        self.assertIn("82.4%", html)
        self.assertIn("High Similarity", html)
        self.assertIn("<svg", html)
        self.assertIn("SIMILARITY", html)


if __name__ == "__main__":
    unittest.main()
