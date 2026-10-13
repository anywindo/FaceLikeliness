## 2024-10-13 - [Denial of Service Prevention and Information Exposure]
**Vulnerability:** The application was vulnerable to Denial of Service via Pillow decompression bombs and OOM via massive numpy arrays, and exposed internal error messages in the Gradio UI.
**Learning:** Image processing endpoints must aggressively validate image dimensions to prevent memory exhaustion by deep learning models. Broad exception handling in web interfaces often leaks sensitive internal state.
**Prevention:** Always enforce `Image.MAX_IMAGE_PIXELS` for PIL, check array shapes for NumPy, and catch generic Exceptions to return safe, generic error messages to the user while logging full details internally.
