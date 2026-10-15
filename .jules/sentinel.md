## 2024-05-18 - Prevent DoS via Large Image Uploads & Prevent Info Leakage
**Vulnerability:** The application allowed uploading NumPy image arrays of unlimited dimensions, posing a Denial of Service (DoS) risk through memory exhaustion. Additionally, raw exception messages (`str(e)`) were being leaked to the user UI upon image processing failures, potentially exposing internal paths or state.
**Learning:** In machine learning/image processing APIs, input dimension limits are crucial as libraries like OpenCV/Pillow can consume massive amounts of memory for gigapixel images. Error boundaries must distinguish between safe-for-user messages and internal logs.
**Prevention:** Implement a strict `MAX_DIMENSION` (e.g., 4096x4096) check on all incoming image arrays before further processing. Sanitize all user-facing error outputs to use generic language while logging the full exception internally.

## 2024-10-13 - [Denial of Service Prevention and Information Exposure]
**Vulnerability:** The application was vulnerable to Denial of Service via Pillow decompression bombs and OOM via massive numpy arrays, and exposed internal error messages in the Gradio UI.
**Learning:** Image processing endpoints must aggressively validate image dimensions to prevent memory exhaustion by deep learning models. Broad exception handling in web interfaces often leaks sensitive internal state.
**Prevention:** Always enforce `Image.MAX_IMAGE_PIXELS` for PIL, check array shapes for NumPy, and catch generic Exceptions to return safe, generic error messages to the user while logging full details internally.
