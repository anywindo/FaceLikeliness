"""Likeliness — Facial Similarity Comparison.

A classic Gradio interface for analyzing facial similarity between two photos
using InsightFace ArcFace deep feature representations.
"""

import logging
import warnings
from typing import Any, List, Tuple

import gradio as gr
import numpy as np

from core.comparator import FaceComparator
from core.detector import FaceDetector
from core.models import DetectedFace

# Suppress external library deprecation warnings for clean console output
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize model detector and comparator
detector = FaceDetector(model_name="buffalo_l")
comparator = FaceComparator()


def process_image(image: np.ndarray | None, label: str) -> Tuple[List[DetectedFace], Any, str]:
    """Detects faces in an uploaded image and toggles selection gallery when multiple faces are present."""
    if image is None:
        return [], gr.update(value=None, visible=False), ""

    try:
        faces = detector.detect(image)
    except Exception as e:
        logger.error(f"Error analyzing {label}: {e}")
        return [], gr.update(value=None, visible=False), f"⚠️ Error processing {label}. Please ensure the image is valid."

    if not faces:
        return [], gr.update(value=None, visible=False), f"⚠️ No face detected in {label}. Please upload an image with a visible face."

    if len(faces) == 1:
        return faces, gr.update(value=None, visible=False), f"✅ 1 face detected in {label}."
    else:
        gallery_items = [(f.thumbnail, f"Face #{i+1}") for i, f in enumerate(faces)]
        return (
            faces,
            gr.update(value=gallery_items, visible=True),
            f"ℹ️ {len(faces)} faces detected in {label}. Defaulting to Face #1 (or select below).",
        )


def select_face_a(evt: gr.SelectData) -> Tuple[int, str]:
    """Updates selected face index for Photo A."""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return int(idx), f"✅ Selected Face #{idx + 1} for Photo A."


def select_face_b(evt: gr.SelectData) -> Tuple[int, str]:
    """Updates selected face index for Photo B."""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return int(idx), f"✅ Selected Face #{idx + 1} for Photo B."


def clear_photo_a() -> Tuple[List[Any], Any, str]:
    """Clears Photo A state and gallery."""
    return [], gr.update(value=None, visible=False), ""


def clear_photo_b() -> Tuple[List[Any], Any, str]:
    """Clears Photo B state and gallery."""
    return [], gr.update(value=None, visible=False), ""


def compare(
    img_a: np.ndarray | None,
    img_b: np.ndarray | None,
    faces_a: List[DetectedFace],
    faces_b: List[DetectedFace],
    idx_a: int,
    idx_b: int,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Compares the selected faces and returns classic Gradio output components."""
    if img_a is None or img_b is None:
        raise gr.Error("Please upload both Photo A and Photo B before comparing.")

    if not faces_a:
        raise gr.Error("No face detected in Photo A. Please upload an image with a visible face.")

    if not faces_b:
        raise gr.Error("No face detected in Photo B. Please upload an image with a visible face.")

    target_a = faces_a[max(0, min(idx_a, len(faces_a) - 1))]
    target_b = faces_b[max(0, min(idx_b, len(faces_b) - 1))]

    result = comparator.compare(target_a, target_b)

    # Standard Gradio Label format with confidence score
    score_fraction = max(0.0, min(1.0, result.similarity_percentage / 100.0))
    label_dict = {
        f"{result.category} ({result.similarity_percentage:.1f}%)": score_fraction,
    }

    # Summary text
    summary = (
        f"Similarity Score: {result.similarity_percentage:.1f}%\n"
        f"Raw Cosine Similarity: {result.cosine_similarity:+.4f}\n"
        f"Category: {result.category}"
    )

    return (
        gr.update(visible=True),   # results_row visible
        target_a.thumbnail,        # crop A
        label_dict,                # similarity label bar
        result.cosine_similarity,  # raw cosine number
        summary,                   # summary text
        target_b.thumbnail,        # crop B
    )


def reset() -> Tuple[Any, ...]:
    """Resets all interface inputs and outputs."""
    return (
        None,                                   # img_a
        None,                                   # img_b
        [],                                     # faces_a
        [],                                     # faces_b
        0,                                      # idx_a
        0,                                      # idx_b
        gr.update(value=None, visible=False),   # gallery_a
        gr.update(value=None, visible=False),   # gallery_b
        "",                                     # status_a
        "",                                     # status_b
        gr.update(visible=False),               # results_row
        None,                                   # crop_a
        None,                                   # score_label
        None,                                   # cosine_box
        "",                                     # summary_box
        None,                                   # crop_b
    )


def build_app() -> gr.Blocks:
    """Builds a classic, clean Gradio interface."""
    with gr.Blocks(title="Facial Similarity Comparison") as demo:
        gr.Markdown(
            """
            # 🔍 Facial Similarity Comparison
            Upload two photos to compare facial similarity using **InsightFace (ArcFace)** deep feature embeddings.
            """
        )

        # Internal selection states
        state_faces_a = gr.State(value=[])
        state_faces_b = gr.State(value=[])
        state_idx_a = gr.State(value=0)
        state_idx_b = gr.State(value=0)

        # Two-column Input Row
        with gr.Row():
            with gr.Column():
                img_a = gr.Image(label="Photo A", type="numpy", sources=["upload", "clipboard"])
                status_a = gr.Markdown("")
                gallery_a = gr.Gallery(
                    label="Detected Faces in Photo A (Click to choose)",
                    columns=4,
                    height="auto",
                    visible=False,
                    interactive=True,
                )

            with gr.Column():
                img_b = gr.Image(label="Photo B", type="numpy", sources=["upload", "clipboard"])
                status_b = gr.Markdown("")
                gallery_b = gr.Gallery(
                    label="Detected Faces in Photo B (Click to choose)",
                    columns=4,
                    height="auto",
                    visible=False,
                    interactive=True,
                )

        # Action Buttons
        with gr.Row():
            compare_btn = gr.Button("Compare Faces", variant="primary", scale=2)
            clear_btn = gr.Button("Clear All", variant="secondary", scale=1)

        # Output Results Section (revealed after comparison)
        with gr.Row(visible=False) as results_row:
            with gr.Column(scale=1):
                crop_a = gr.Image(label="Face A (Analyzed Crop)", interactive=False)

            with gr.Column(scale=2):
                score_label = gr.Label(label="Similarity Metric")
                cosine_box = gr.Number(label="Raw Cosine Similarity (-1.0 to +1.0)", precision=4)
                summary_box = gr.Textbox(label="Assessment Summary", lines=3, interactive=False)

            with gr.Column(scale=1):
                crop_b = gr.Image(label="Face B (Analyzed Crop)", interactive=False)

        # Classic Information Accordion
        with gr.Accordion("ℹ️ Model Information & Threshold Reference", open=False):
            gr.Markdown(
                """
                ### Calibrated Threshold Reference
                | Category | Raw Cosine | Calibrated Score | Interpretation |
                |---|---|---|---|
                | **Low Similarity** | `< 0.18` | `< 45.0%` | Distinct biometric profiles; unrelated strangers. |
                | **Look-alike / Resemblance** | `0.18 – 0.38` | `45.0% – 75.0%` | **Look-alikes (anchored at ~50%)**, family members, or noticeable facial similarities. |
                | **Same Person (Match)** | `≥ 0.38` | `≥ 75.0%` | High geometric correspondence; photos of the same person. |
                | **Identical File** | `1.00` | `100.0%` | The exact same digital photo. |

                ### Mathematical Formulation
                - **Raw Cosine Similarity:** $\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$ (for unit-normalized 512-d ArcFace embeddings)
                - **Calibrated Scale:** Anchors strangers at low scores (15–30%), look-alikes at **50%**, and identity matches at **≥75%**.

                ### Responsible Use & Privacy
                - All processing executes **locally in-memory** on your machine. No photos or face embeddings are stored.
                - The percentage score is an algorithmic distance measurement between feature vectors, **not** a calibrated probability of identity match or legal identity verification.
                """
            )

        # ---------------- Event Bindings ----------------
        img_a.upload(
            fn=lambda im: process_image(im, "Photo A"),
            inputs=[img_a],
            outputs=[state_faces_a, gallery_a, status_a],
        )
        img_a.clear(
            fn=clear_photo_a,
            inputs=[],
            outputs=[state_faces_a, gallery_a, status_a],
        )

        img_b.upload(
            fn=lambda im: process_image(im, "Photo B"),
            inputs=[img_b],
            outputs=[state_faces_b, gallery_b, status_b],
        )
        img_b.clear(
            fn=clear_photo_b,
            inputs=[],
            outputs=[state_faces_b, gallery_b, status_b],
        )

        gallery_a.select(
            fn=select_face_a,
            inputs=[],
            outputs=[state_idx_a, status_a],
        )
        gallery_b.select(
            fn=select_face_b,
            inputs=[],
            outputs=[state_idx_b, status_b],
        )

        compare_btn.click(
            fn=compare,
            inputs=[
                img_a,
                img_b,
                state_faces_a,
                state_faces_b,
                state_idx_a,
                state_idx_b,
            ],
            outputs=[
                results_row,
                crop_a,
                score_label,
                cosine_box,
                summary_box,
                crop_b,
            ],
            show_progress="full",
        )

        clear_btn.click(
            fn=reset,
            inputs=[],
            outputs=[
                img_a,
                img_b,
                state_faces_a,
                state_faces_b,
                state_idx_a,
                state_idx_b,
                gallery_a,
                gallery_b,
                status_a,
                status_b,
                results_row,
                crop_a,
                score_label,
                cosine_box,
                summary_box,
                crop_b,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
