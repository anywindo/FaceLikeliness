## 2023-10-27 - Redundant L2 Norm Computations on Normalized Embeddings
**Learning:** In pipelines using models like InsightFace (ArcFace), face embeddings are almost always L2-normalized upon extraction (e.g., returning unit vectors). Recomputing the L2 norm of these vectors before calculating cosine similarity adds unnecessary O(N) operations.
**Action:** When working with normalized embeddings, replace the full cosine similarity formula (dot product divided by magnitudes) with a simple dot product (`np.dot(a, b)`).
