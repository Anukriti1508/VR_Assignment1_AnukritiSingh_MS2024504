"""==================================================
VR_Assignment1_AnukritiSingh_MS202450
@author Anukriti Singh
==================================================="""

import cv2
import numpy as np
import glob
import os

def detect_and_compute_keypoints(image):
    """Detect keypoints and compute descriptors using SIFT"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2):
    """Match keypoints using FLANN-based matcher."""
    if desc1 is None or desc2 is None:
        return []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Increase search checks for accuracy

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    return good_matches

def stitch_images(img1, img2):
    """Stitch two images together using feature matching and homography."""
    kp1, desc1 = detect_and_compute_keypoints(img1)
    kp2, desc2 = detect_and_compute_keypoints(img2)

    if desc1 is None or desc2 is None:
        print("No descriptors found in one of the images.")
        return None

    matches = match_keypoints(desc1, desc2)

    if len(matches) < 15:  # Increase threshold for better stitching
        print("Not enough good matches found!")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)  # Stricter RANSAC threshold

    if H is None:
        print("Homography calculation failed.")
        return None

    # Compute new dimensions for the stitched image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    min_x = min(transformed_corners[:, 0, 0].min(), 0)
    min_y = min(transformed_corners[:, 0, 1].min(), 0)
    max_x = max(transformed_corners[:, 0, 0].max(), w2)
    max_y = max(transformed_corners[:, 0, 1].max(), h2)

    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)

    # Translate transformation matrix
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    H_translated = translation_matrix @ H

    result = cv2.warpPerspective(img1, H_translated, (new_width, new_height))
    
    # Blending for smoother transition
    result[-int(min_y):h2-int(min_y), -int(min_x):w2-int(min_x)] = img2

    return result

def resize_images(images, target_height=500):
    """Resize images to a common height while maintaining aspect ratio."""
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized_images.append(cv2.resize(img, (new_w, target_height)))
    return resized_images

def stitch_multiple_images(image_files):
    """Stitch multiple images into a single panorama."""
    images = [cv2.imread(img) for img in image_files]
    
    if any(img is None for img in images):
        print("Error: Could not read some images.")
        return None

    # Resize images to a common height
    images = resize_images(images)

    panorama = images[0]
    for i in range(1, len(images)):
        stitched = stitch_images(panorama, images[i])
        if stitched is None:
            print(f"Skipping image {i} due to insufficient matches")
            continue
        panorama = stitched

    return panorama

def main():
    """Main function to load multiple images, stitch them, and save the result."""
    image_dir = input("Enter the directory containing images")
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpeg")))

    if len(image_files) < 2:
        print("Error: Need at least two images to create a panorama.")
        return

    panorama = stitch_multiple_images(image_files)

    if panorama is not None:
        output_path = os.path.join(image_dir, "panorama.jpeg")
        cv2.imwrite(output_path, panorama)
        cv2.imshow("Panorama", panorama)
if __name__ == "__main__":
    main()
