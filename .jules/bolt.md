## $(date +%Y-%m-%d) - Redundant L2 Norms in Cosine Similarity
**Learning:** Face embeddings extracted by ArcFace/InsightFace (and handled by `FaceEmbedder`) are already strictly L2-normalized unit vectors. Re-calculating `np.linalg.norm()` and dividing during every comparison is a redundant O(N) operation.
**Action:** Always check if upstream embedding extraction pipelines already normalize vectors. If they do, cosine similarity can be safely reduced to a simple `np.dot(a, b)` (with bounds clipping for precision), doubling the speed of distance calculations.

## $(date +%Y-%m-%d) - Gradio Image Serialization Overhead
**Learning:** In Gradio applications, passing raw `Image` components as inputs to downstream event handlers forces the client to serialize and upload the full high-res image over the network upon every interaction (e.g., button click). This can cause massive network payloads and slow down responsiveness, especially when only lightweight extracted states (like metadata or face embeddings) are required for the action.
**Action:** When chaining events or handling clicks, avoid passing heavy UI components (like `gr.Image`) to the backend if the necessary data has already been extracted into lightweight state objects (`gr.State`). Relying solely on `gr.State` prevents redundant network transfers and dramatically speeds up event handlers.
