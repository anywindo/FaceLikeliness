"""Image loading, validation, and preprocessing utilities."""

from typing import Union
import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_PIXELS = 16_000_000  # 16 Megapixels (approx 4000x4000)


def compress_image(
    image: Union[np.ndarray, Image.Image],
    max_pixels: int = MAX_PIXELS,
) -> Union[np.ndarray, Image.Image]:
    """Compresses an image to fit within the maximum allowed pixel budget.

    Preserves the original aspect ratio and shrinks image dimensions to ensure
    width * height <= max_pixels.

    Args:
        image: The input image as a NumPy array or PIL Image.
        max_pixels: Maximum allowed total pixels (default: 16,000,000).

    Returns:
        The compressed image fitting within the pixel limit.
    """
    if isinstance(image, Image.Image):
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            new_w = max(1, int(image.width * scale))
            new_h = max(1, int(image.height * scale))
            while new_w * new_h > max_pixels:
                new_w = max(1, new_w - 1)
                new_h = max(1, new_h - 1)
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    if isinstance(image, np.ndarray) and image.ndim >= 2:
        h, w = image.shape[:2]
        pixels = h * w
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            while new_w * new_h > max_pixels:
                new_w = max(1, new_w - 1)
                new_h = max(1, new_h - 1)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    return image


def validate_image(image: Union[np.ndarray, Image.Image, None]) -> np.ndarray:
    """Validates and preprocesses an image for face detection.

    Converts various input formats (NumPy array or PIL Image)
    into a standard RGB NumPy array, handling color channels and EXIF orientation.
    If an image exceeds the 16 MP budget, it is compressed to fit within limits.

    Args:
        image: The input image as a NumPy array (RGB) or PIL Image.

    Returns:
        np.ndarray: Preprocessed image in RGB uint8 format.

    Raises:
        ValueError: If the image is None, empty, corrupt, or unsupported.
    """
    if image is None:
        raise ValueError("No image provided. Please upload an image.")

    # Compress image to adhere to the 16 MP limit
    image = compress_image(image, max_pixels=MAX_PIXELS)

    # If image is a PIL Image
    if isinstance(image, Image.Image):
        try:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return np.array(image, dtype=np.uint8)
        except Exception:
            raise ValueError(
                "Error processing PIL image. The image format may be unsupported or corrupted."
            )

    # If image is a numpy array
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("The provided image is empty.")

        if image.ndim < 2:
            raise ValueError(f"Invalid image array dimensions: {image.ndim}")

        # Ensure correct dtype
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Handle different channel counts
        if image.ndim == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                # Already 3-channel (assumed RGB from Gradio)
                pass
            else:
                raise ValueError(
                    f"Unsupported number of image channels: {image.shape[2]}"
                )
        else:
            raise ValueError(f"Invalid image array dimensions: {image.ndim}")

        return image

    raise ValueError(f"Unsupported image type: {type(image)}")


def create_thumbnail(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | list[int],
    target_size: tuple[int, int] = (224, 224),
    padding: float = 0.35,
) -> np.ndarray:
    """Crops a detected face region with padding and resizes to target_size.

    Args:
        image: Original RGB image array.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        target_size: Output thumbnail dimensions (width, height).
        padding: Proportional padding around the bounding box.

    Returns:
        np.ndarray: RGB cropped and resized thumbnail.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

    w_box = max(1, x2 - x1)
    h_box = max(1, y2 - y1)

    pad_w = int(w_box * padding)
    pad_h = int(h_box * padding)

    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w_img, x2 + pad_w)
    crop_y2 = min(h_img, y2 + pad_h)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        # Fallback to direct bbox or blank image
        crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    thumbnail = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return thumbnail
