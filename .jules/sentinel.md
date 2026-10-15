## 2024-10-13 - [Denial of Service Prevention and Information Exposure]
**Vulnerability:** The application was vulnerable to Denial of Service via Pillow decompression bombs and OOM via massive numpy arrays, and exposed internal error messages in the Gradio UI.
**Learning:** Image processing endpoints must aggressively validate image dimensions to prevent memory exhaustion by deep learning models. Broad exception handling in web interfaces often leaks sensitive internal state.
**Prevention:** Always enforce `Image.MAX_IMAGE_PIXELS` for PIL, check array shapes for NumPy, and catch generic Exceptions to return safe, generic error messages to the user while logging full details internally.
## 2024-10-24 - [Information Exposure via Exception Handling]
**Vulnerability:** The `validate_image` function exposed sensitive file paths (e.g. OS-level errors like `Errno 2` and absolute paths) directly to users via unhandled or pass-through exception messages, leading to Information Exposure (CWE-209).
**Learning:** Broad exception handling in data processing utilities can accidentally bubble up sensitive server-side context directly to public web interfaces if the exception strings are blindly passed through.
**Prevention:** Always log full exception traces server-side (`logger.error(..., exc_info=True)`) while returning generic, safe error strings ("Unable to read image file") to the end user.
