import cv2
import numpy as np

def preprocess_fixation_map(image, new_size, gaussian_kernel=50, dilation_kernel=3):
    """Preprocess driver fixation map before resizing."""
    # Step 1: Dilate small fixations at original size
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)

    # Step 2: Apply Gaussian Blur before resizing
    blurred = cv2.GaussianBlur(dilated, (gaussian_kernel, gaussian_kernel), 0)

    # Step 3: Resize to target dimensions (224x224)
    resized = cv2.resize(blurred, new_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to range [0, 255]
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)

    return resized.astype(np.uint8)

# Load binary fixation map (grayscale)
image = cv2.imread("/mnt/data/001_frame_41.jpg", cv2.IMREAD_GRAYSCALE)

# Resize to 224x224 while keeping fixation structure
processed_image = preprocess_fixation_map(image, (224, 224))

# Save or visualize the output
cv2.imwrite("/mnt/data/processed_fixation_map.jpg", processed_image)
cv2.imshow("Processed Fixation Map", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
