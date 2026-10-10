"""Facial similarity comparison and scoring using ArcFace embeddings."""

import numpy as np
from core.models import ComparisonResult, DetectedFace


class FaceComparator:
    """Computes similarity metrics between face embeddings and generates human-readable results."""

    # Documented ArcFace cosine similarity thresholds:
    # On ArcFace (512-d normalized embeddings), cosine similarity for pairs of faces typically ranges from:
    # ~0.00-0.18 for distinctly different strangers (maps to ~15-45%),
    # ~0.22 for noticeable look-alikes / doppelgängers (calibrated to anchor at 50%),
    # ~0.38-0.40 for identity match threshold (calibrated to anchor at 75%),
    # >0.40 for same individual across various conditions (maps to 75-100%).
    THRESHOLD_LOW = 0.18
    THRESHOLD_MODERATE = 0.38

    def compute_cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Computes cosine similarity between two unit vectors.

        Args:
            emb_a: First normalized embedding vector.
            emb_b: Second normalized embedding vector.

        Returns:
            float: Cosine similarity in range [-1.0, 1.0].
        """
        a = emb_a.flatten()
        b = emb_b.flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        # Clip to valid range due to numerical precision
        return float(np.clip(cos_sim, -1.0, 1.0))

    def normalize_score(self, cosine_sim: float) -> float:
        """Converts raw cosine similarity [-1.0, 1.0] to an intuitive, calibrated 0-100% scale.

        Calibrated anchor points:
        - Strangers (cos <= 0.00): 0% to 15%
        - Unrelated / slight similarities (cos 0.00 to 0.22): 15% to 50%
        - Look-alike / Resemblance anchor (cos = 0.22): exactly 50.0%
        - Family / Moderate resemblance (cos 0.22 to 0.38): 50% to 75%
        - Same person match threshold (cos = 0.38): 75.0%
        - Same person across different photos (cos 0.38 to 1.00): 75% to 100%
        - Exact same photo (cos = 1.00): 100.0%

        Args:
            cosine_sim: Raw cosine similarity value.

        Returns:
            float: Calibrated percentage score between 0.0 and 100.0.
        """
        x = float(np.clip(cosine_sim, -1.0, 1.0))

        if x <= 0.0:
            score = max(0.0, 15.0 * (1.0 + x))
        elif x <= 0.22:
            score = 15.0 + (x / 0.22) * (50.0 - 15.0)
        elif x <= 0.38:
            score = 50.0 + ((x - 0.22) / (0.38 - 0.22)) * (75.0 - 50.0)
        else:
            score = 75.0 + ((x - 0.38) / (1.00 - 0.38)) * (100.0 - 75.0)

        return float(np.clip(score, 0.0, 100.0))

    def categorize(self, cosine_sim: float) -> str:
        """Categorizes similarity based on model-calibrated thresholds.

        Args:
            cosine_sim: Raw cosine similarity value.

        Returns:
            str: "Low Similarity", "Moderate Similarity (Look-alike)", or "High Similarity (Same Person)".
        """
        if cosine_sim < self.THRESHOLD_LOW:
            return "Low Similarity"
        elif cosine_sim < self.THRESHOLD_MODERATE:
            return "Moderate Similarity (Look-alike)"
        else:
            return "High Similarity (Same Person)"

    def generate_explanation(self, cosine_sim: float, percentage: float, category: str) -> str:
        """Produces a transparent, responsible explanation of the similarity score.

        Args:
            cosine_sim: Raw cosine similarity.
            percentage: Normalized percentage score.
            category: Qualitative category.

        Returns:
            str: Markdown explanation text.
        """
        if "High" in category:
            context = (
                "The two faces exhibit strong structural and geometric facial similarities according "
                "to the ArcFace deep feature representation. Such scores typically occur when comparing "
                "photos of the same individual under varying conditions, or between individuals with marked resemblance."
            )
        elif "Moderate" in category:
            context = (
                "The two faces share noticeable facial characteristics and proportions (look-alikes, family resemblance, "
                "or similar features), with visible differences in specific biometric landmarks."
            )
        else:
            context = (
                "The two faces present markedly distinct biometric profiles. "
                "The deep feature representations do not indicate significant visual correspondence."
            )

        explanation = (
            f"**Analysis Category:** {category}\n\n"
            f"{context}\n\n"
            f"> **Technical Details & Responsible Use:**\n"
            f"> - **Raw Cosine Similarity:** `{cosine_sim:.4f}` (range `[-1.0, 1.0]`)\n"
            f"> - **Calibrated Similarity Score:** `{percentage:.1f}%`\n"
            f"> - **Threshold Reference:** `<18%`: Strangers (<45%), `~22%`: Look-alike (~50%), `≥38%`: Same Person (≥75%)\n"
            f"> - **Notice:** This percentage is a mathematical model-derived similarity measurement, "
            f"**not** a definitive legal verification. Scores are influenced by lighting, pose, and camera angle."
        )
        return explanation

    def compare(
        self,
        face_a: DetectedFace,
        face_b: DetectedFace,
    ) -> ComparisonResult:
        """Compares two detected faces and generates a ComparisonResult.

        Args:
            face_a: First detected face with embedding and thumbnail.
            face_b: Second detected face with embedding and thumbnail.

        Returns:
            ComparisonResult: Detailed comparison result.
        """
        cos_sim = self.compute_cosine_similarity(face_a.embedding, face_b.embedding)
        percentage = self.normalize_score(cos_sim)
        category = self.categorize(cos_sim)
        explanation = self.generate_explanation(cos_sim, percentage, category)

        return ComparisonResult(
            cosine_similarity=cos_sim,
            similarity_percentage=round(percentage, 1),
            category=category,
            explanation=explanation,
            face_a_thumbnail=face_a.thumbnail,
            face_b_thumbnail=face_b.thumbnail,
        )
