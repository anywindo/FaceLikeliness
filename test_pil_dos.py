import numpy as np
from PIL import Image
from utils.image import validate_image

def main():
    # Create a large PIL image
    img = Image.new("RGB", (5000, 5000))
    try:
        res = validate_image(img)
        print("Success:", res.shape)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
