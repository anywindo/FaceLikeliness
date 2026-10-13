## 2024-05-24 - [Information Leakage and DoS Mitigation]
**Vulnerability:** The application was catching exceptions in `app.py` and returning `str(e)` directly to the UI, which could leak internal file paths or stack details. Furthermore, `utils/image.py` processed any numpy array size, leaving it vulnerable to memory exhaustion / DoS.
**Learning:** In Gradio applications that handle uploads, raw exception strings and unconstrained input bounds often slip through because the framework automatically surfaces returned strings.
**Prevention:** Always mask raw exception strings with generic UI messages and log the detailed error internally. Enforce strict memory bounds (e.g., maximum resolution limits) on image processing entry points.
