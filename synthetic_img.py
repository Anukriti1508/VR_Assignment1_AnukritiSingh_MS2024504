import cv2
import numpy as np

def generate_synthetic_images(num_images=5, shift=50):
    """Generate synthetic overlapping images with a horizontal shift."""
    base_image = np.zeros((300, 500, 3), dtype=np.uint8)

    for i in range(num_images):
        img = base_image.copy()
        cv2.putText(img, f"Image {i+1}", (100 + i * shift, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        filename = f"image{i+1}.jpg"
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")

generate_synthetic_images()
