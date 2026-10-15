## $(date +%Y-%m-%d) - Redundant L2 Norms in Cosine Similarity
**Learning:** Face embeddings extracted by ArcFace/InsightFace (and handled by `FaceEmbedder`) are already strictly L2-normalized unit vectors. Re-calculating `np.linalg.norm()` and dividing during every comparison is a redundant O(N) operation.
**Action:** Always check if upstream embedding extraction pipelines already normalize vectors. If they do, cosine similarity can be safely reduced to a simple `np.dot(a, b)` (with bounds clipping for precision), doubling the speed of distance calculations.
