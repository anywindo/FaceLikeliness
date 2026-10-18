## 2023-10-27 - Redundant L2 Norm Computations on Normalized Embeddings
**Learning:** In pipelines using models like InsightFace (ArcFace), face embeddings are almost always L2-normalized upon extraction (e.g., returning unit vectors). Recomputing the L2 norm of these vectors before calculating cosine similarity adds unnecessary O(N) operations.
**Action:** When working with normalized embeddings, replace the full cosine similarity formula (dot product divided by magnitudes) with a simple dot product (`np.dot(a, b)`).

## 2026-10-16 - Prevent Massive Network Payload Bloat on Gradio Image Component Events
**Learning:** In Gradio applications, passing raw `Image` components as inputs to downstream event handlers (e.g., a "Compare" button click) forces the client to serialize and re-upload the full high-res image over the network every single time the event is triggered, even if the image hasn't changed. This causes significant performance bottlenecks and network payload bloat.
**Action:** When the raw image data isn't needed by the downstream handler (because the required data, like extracted features or embeddings, is already available via lightweight `gr.State` or other components), replace the `Image` inputs with lightweight state variables (like `gr.State` or short status strings) to act as presence checks without the network overhead.

## 2026-10-18 - Optimize Gradio Payload States by Emptying Unused Data
**Learning:** When passing complex objects (like data contracts) into Gradio `gr.State`, unused large attributes (such as image crops or high-res arrays) within those objects get serialized and consume significant payload memory, even if the frontend doesn't need them.
**Action:** Replace expensive and unused attributes in data contracts with empty arrays (e.g., `np.empty((0,), dtype=np.uint8)`) if they are only needed to preserve the contract. This preserves the schema while eliminating processing time and reducing memory overhead in Gradio state payloads.
