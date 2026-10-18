## 2024-05-24 - [Information Leakage and DoS Mitigation]
**Vulnerability:** The application was catching exceptions in `app.py` and returning `str(e)` directly to the UI, which could leak internal file paths or stack details. Furthermore, `utils/image.py` processed any numpy array size, leaving it vulnerable to memory exhaustion / DoS.
**Learning:** In Gradio applications that handle uploads, raw exception strings and unconstrained input bounds often slip through because the framework automatically surfaces returned strings.
**Prevention:** Always mask raw exception strings with generic UI messages and log the detailed error internally. Enforce strict memory bounds (e.g., maximum resolution limits) on image processing entry points.
## 2025-02-14 - Fix Path Traversal and Error Leakage in Image Validation
**Vulnerability:** The `validate_image` function accepted arbitrary local file paths (strings) and opened them using `Image.open()`. Additionally, error handlers exposed raw exceptions (`str(e)`) to the frontend in both `utils/image.py` and `app.py`.
**Learning:** Functions designed to process uploads from a web interface should not accept local file paths as a fallback, as this allows Server-Side Request Forgery (SSRF) and Path Traversal if the UI boundaries are bypassed by a malicious API request. Furthermore, exposing internal exception details in UI elements violates the "Fail securely" principle and provides attackers with valuable reconnaissance information.
**Prevention:** Strictly type-check inputs to only accept memory-based objects (like NumPy arrays or PIL Images) rather than file paths. Return generic, safe error messages to the user while logging detailed stack traces internally.

## 2026-10-18 - Early DoS Mitigation in PIL Image Validation & Safe UI Exceptions
**Vulnerability:** The `validate_image` function validated NumPy array resolutions against 16MP but did not check `PIL.Image.Image` dimensions before executing expensive operations like EXIF transposition and RGB conversion, leading to memory exhaustion DoS vulnerability. Furthermore, unhandled exceptions in Gradio comparison events could leak raw tracebacks to the browser.
**Learning:** Validation constraints must be applied upfront across all input types (NumPy and PIL) before performing compute-intensive transformations. All top-level UI action handlers should capture unexpected errors and present clean messages without exposing backend call stacks.
**Prevention:** Enforce `image.width * image.height <= max_pixels` on PIL images immediately upon receipt. Wrap comparison and processing handlers in `try/except` and raise sanitized `gr.Error` messages.

## 2026-10-18 - Generic Exception Messaging in Gradio UIs
**Vulnerability:** A `ValueError` exception generated during image processing was caught and its raw string representation (`str(e)`) was directly exposed in a Gradio UI Markdown element, leading to potential information leakage.
**Learning:** Returning exception strings directly to user-facing UI elements violates the "Fail securely" principle, as it can expose internal file paths, module structures, or variable values to end users or attackers.
**Prevention:** Catch expected errors but present generic, sanitized failure messages to the user (e.g., "Invalid image provided"), while keeping the detailed error context inside secure server-side logs using `logger.warning`.
