## 2026-10-14 - Redundant L2 Norms in Cosine Similarity
**Learning:** Face embeddings extracted by ArcFace/InsightFace (and handled by `FaceEmbedder`) are already strictly L2-normalized unit vectors. Re-calculating `np.linalg.norm()` and dividing during every comparison is a redundant O(N) operation.
**Action:** Always check if upstream embedding extraction pipelines already normalize vectors. If they do, cosine similarity can be safely reduced to a simple `np.dot(a, b)` (with bounds clipping for precision), doubling the speed of distance calculations.

## 2026-10-14 - Redundant network transfer of full-resolution Image components in Gradio
**Learning:** Passing raw Gradio `Image` components to downstream event handlers (like a compare button) forces the client to serialize and upload the full high-res image (potentially up to ~50MB) over the network for *every single click*, even if the backend only needs state/metadata (like already extracted faces).
**Action:** Always check `gr.Button.click` or similar event `inputs`. If the function only needs to verify if an image was uploaded, or uses previously extracted state (like `gr.State`), remove the raw `Image` component from the inputs to prevent massive unnecessary payloads.
