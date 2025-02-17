import cv2
import numpy as np
import glob

def detect_and_compute_keypoints(image):
    """Detect keypoints and compute descriptors using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2):
    """Match keypoints using Brute-Force matcher."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def stitch_images(img1, img2):
    """Stitch two images together using feature matching and homography."""
    kp1, desc1 = detect_and_compute_keypoints(img1)
    kp2, desc2 = detect_and_compute_keypoints(img2)
    matches = match_keypoints(desc1, desc2)
    
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        height, width = img2.shape[:2]
        result = cv2.warpPerspective(img1, H, (width * 2, height))
        result[0:height, 0:width] = img2
        return result
    else:
        print("Not enough matches found!")
        return None

def stitch_multiple_images(image_files):
    """Stitch multiple images into a single panorama."""
    images = [cv2.imread(img) for img in image_files]
    
    if any(img is None for img in images):
        print("Error: Could not read some images.")
        return None
    
    panorama = images[0]
    for i in range(1, len(images)):
        panorama = stitch_images(panorama, images[i])
        if panorama is None:
            print(f"Error stitching images {i-1} and {i}")
            return None
    
    return panorama

import cv2
import numpy as np
import glob
import os

def detect_and_compute_keypoints(image):
    """Detect keypoints and compute descriptors using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2):
    """Match keypoints using Brute-Force matcher."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def stitch_images(img1, img2):
    """Stitch two images together using feature matching and homography."""
    kp1, desc1 = detect_and_compute_keypoints(img1)
    kp2, desc2 = detect_and_compute_keypoints(img2)
    matches = match_keypoints(desc1, desc2)
    
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        height, width = img2.shape[:2]
        result = cv2.warpPerspective(img1, H, (width * 2, height))
        result[0:height, 0:width] = img2
        return result
    else:
        print("Not enough matches found!")
        return None

def stitch_multiple_images(image_files):
    """Stitch multiple images into a single panorama."""
    images = [cv2.imread(img) for img in image_files]
    
    if any(img is None for img in images):
        print("Error: Could not read some images.")
        return None
    
    panorama = images[0]
    for i in range(1, len(images)):
        panorama = stitch_images(panorama, images[i])
        if panorama is None:
            print(f"Error stitching images {i-1} and {i}")
            return None
    
    return panorama

def main():
    """Main function to load multiple images, stitch them, and save the result."""
    image_dir = input("Enter the directory containing images: ")
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))  
    
    if len(image_files) < 2:
        print("Error: Need at least two images to create a panorama.")
        return
    
    panorama = stitch_multiple_images(image_files)
    
    if panorama is not None:
        output_path = os.path.join(image_dir, "panorama.jpg")
        cv2.imwrite(output_path, panorama)
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

