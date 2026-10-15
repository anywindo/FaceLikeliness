## $(date +%Y-%m-%d) - Redundant L2 Norms in Cosine Similarity
**Learning:** Face embeddings extracted by ArcFace/InsightFace (and handled by `FaceEmbedder`) are already strictly L2-normalized unit vectors. Re-calculating `np.linalg.norm()` and dividing during every comparison is a redundant O(N) operation.
**Action:** Always check if upstream embedding extraction pipelines already normalize vectors. If they do, cosine similarity can be safely reduced to a simple `np.dot(a, b)` (with bounds clipping for precision), doubling the speed of distance calculations.
## 2024-05-24 - Gradio Network Payload Optimization
**Learning:** In Gradio, passing raw `Image` components as inputs to downstream event handlers (like the `compare` button click) forces the client to serialize and upload the full high-res image over the network on every invocation, even if the processing function doesn't actually need the raw image.
**Action:** Use extracted lightweight states (e.g., `gr.State` representing detected faces) instead of raw image components for downstream events to prevent massive network payloads.
## 2023-10-27 - Redundant L2 Norm Computations on Normalized Embeddings
**Learning:** In pipelines using models like InsightFace (ArcFace), face embeddings are almost always L2-normalized upon extraction (e.g., returning unit vectors). Recomputing the L2 norm of these vectors before calculating cosine similarity adds unnecessary O(N) operations.
**Action:** When working with normalized embeddings, replace the full cosine similarity formula (dot product divided by magnitudes) with a simple dot product (`np.dot(a, b)`).
