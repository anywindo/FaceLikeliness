## 2024-06-12 - Prevent Loading Unused InsightFace Models
**Learning:** InsightFace `FaceAnalysis` initializes models for gender/age and dense landmarks by default if `allowed_modules` is not explicitly set, which needlessly increases memory consumption and startup time when only face detection and recognition are required.
**Action:** Always explicitly specify `allowed_modules=["detection", "recognition"]` when initializing `FaceAnalysis` to only load the models strictly needed for the functionality.

## 2024-06-12 - Prevent Loading Unused InsightFace Models
**Learning:** InsightFace `FaceAnalysis` initializes models for gender/age and dense landmarks by default if `allowed_modules` is not explicitly set, which needlessly increases memory consumption and startup time when only face detection and recognition are required.
**Action:** Always explicitly specify `allowed_modules=["detection", "recognition"]` when initializing `FaceAnalysis` to only load the models strictly needed for the functionality.
