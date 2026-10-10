# 🔍 Likeliness — AI Facial Similarity Comparison

An AI-powered facial similarity analysis application built with **InsightFace (ArcFace)** and **Gradio**. Upload two photos to automatically detect faces, extract 512-dimensional deep biometric feature embeddings, and analyze their facial similarity with calibrated, responsible reporting.

---

## ✨ Features

- **Multi-Format Image Support**: Upload JPG, JPEG, PNG, or WEBP photos independently.
- **Robust Face Detection**: Automatic facial localization and 5-point landmark alignment using InsightFace `buffalo_l`.
- **Multi-Face Disambiguation**: When multiple people are detected in an image, an interactive gallery displays cropped face previews so you can select the exact face to compare.
- **Deep Biometric Embeddings**: Extracts 512-dimensional L2-normalized feature representations via ArcFace.
- **Calibrated Similarity Metrics**:
  - Raw cosine similarity in `[-1.0, 1.0]`.
  - Normalized percentage metric mapped cleanly via `((cos + 1) / 2) * 100`.
  - Qualitative categories (*Low Similarity*, *Moderate Similarity*, *High Similarity*) aligned with ArcFace verification benchmarks.
- **Classic Gradio Interface**: Native, clean Gradio layout with side-by-side photo inputs, official `gr.Label` confidence bar, and collapsible documentation accordion.
- **Privacy & Responsible Use**: 100% local, in-memory processing on Apple Silicon (M1 Pro CoreML / CPU). No images or embeddings are stored or transmitted.

---

## 🏗️ Architecture

```
Likeliness/
├── environment.yml       # Conda environment definition (Apple Silicon optimized)
├── app.py                # Gradio Blocks web UI & event controllers
├── core/
│   ├── __init__.py
│   ├── models.py         # Data contracts (DetectedFace, ComparisonResult)
│   ├── detector.py       # InsightFace FaceAnalysis & alignment
│   ├── embedder.py       # 512-dim embedding extraction & L2 normalization
│   └── comparator.py     # Cosine similarity calculation & score categorization
├── utils/
│   ├── __init__.py
│   ├── image.py          # Format conversion, EXIF handling, padded thumbnails
│   └── gauge.py          # Modern SVG circular score gauge
└── tests/
    └── test_pipeline.py  # Automated unit test suite
```

---

## 🚀 Getting Started

### 1. Prerequisites
- [Miniforge](https://github.com/conda-forge/miniforge) or [Anaconda](https://www.anaconda.com/) installed on macOS (Apple Silicon compatible).

### 2. Environment Setup
Create and activate the dedicated conda environment:

```bash
conda env create -f environment.yml
conda activate likeliness
```

### 3. Launch the Application
Start the Gradio interface:

```bash
python app.py
```

Then open your browser at the displayed local URL (typically `http://127.0.0.1:7860`).

---

## 📊 Methodology & Thresholds

### Mathematical Scoring
1. **Cosine Similarity**: For unit-normalized embedding vectors $\mathbf{u}$ and $\mathbf{v}$:
   $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$
2. **Normalized Percentage**:
   $$\text{Score (\%)} = \left(\frac{\text{Cosine Similarity} + 1}{2}\right) \times 100$$

### Model Thresholds (ArcFace Buffalo_L)
| Category | Cosine Similarity | Normalized Score | Interpretation |
|---|---|---|---|
| **Low Similarity** | $< 0.20$ | $< 60.0\%$ | Distinct facial structures; typical of unrelated individuals. |
| **Moderate Similarity** | $0.20 - 0.40$ | $60.0\% - 70.0\%$ | Shared facial characteristics, familial resemblances, or variations in lighting/angles. |
| **High Similarity** | $\ge 0.40$ | $\ge 70.0\%$ | Strong facial geometry correspondence; typical of very similar faces or identical individuals. |

---

## 🔒 Responsible Use & Privacy Notice

- **Model-Derived Metric**: The reported percentage is a mathematical measurement of feature embedding distance, **not** a calibrated probability of identity match or definitive legal identity verification.
- **Environmental Factors**: Biometric similarity scores are influenced by camera focal length, lighting, head tilt, expression, and age variance.
- **Zero Retention**: All computations occur locally in volatile memory. No uploaded images or face representations are persisted to disk or sent to external servers.

---

## 🧪 Running Tests

Execute the unit test suite:

```bash
conda run -n likeliness python -m unittest discover tests
```
