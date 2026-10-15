## 2024-10-13 - [Denial of Service Prevention and Information Exposure]
**Vulnerability:** The application was vulnerable to Denial of Service via Pillow decompression bombs and OOM via massive numpy arrays, and exposed internal error messages in the Gradio UI.
**Learning:** Image processing endpoints must aggressively validate image dimensions to prevent memory exhaustion by deep learning models. Broad exception handling in web interfaces often leaks sensitive internal state.
**Prevention:** Always enforce `Image.MAX_IMAGE_PIXELS` for PIL, check array shapes for NumPy, and catch generic Exceptions to return safe, generic error messages to the user while logging full details internally.

## 2024-03-22 - Gradio UI Exception Leakage
**Vulnerability:** Information leakage / Path disclosure in `utils/image.py` exceptions bubbling up to Gradio UI.
**Learning:** In Gradio applications, unhandled or caught-and-rethrown exceptions that include raw `str(e)` are often bubbled up directly to the frontend UI as status messages (e.g., via `gr.update(value=None, visible=False), f"⚠️ Error: {str(e)}"`). This can inadvertently leak sensitive system information like file existence (LFI Oracle) or internal stack details to end users.
**Prevention:** Always fail securely by returning generic error messages in validation functions. Avoid interpolating `str(e)` into exceptions that might be rendered in the UI, especially for I/O and file handling operations.
