import cv2
import numpy as np
import os

def detect_coins(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Use HoughCircles to detect coins
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=20, maxRadius=100)
    
    detected_image = image.copy()
    coin_count = 0
    coins = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            coin_count += 1
            cv2.circle(detected_image, (x, y), radius, (0, 255, 0), 3)
            
            # Ensure bounding box is within image boundaries
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(image.shape[1], x + radius), min(image.shape[0], y + radius)
            cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Extract individual coins safely
            coin = image[y1:y2, x1:x2].copy()
            if coin.size > 0:
                coins.append(coin)
    
    return detected_image, coins, coin_count

def save_segmented_coins(coins, output_folder="segmented_coins"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, coin in enumerate(coins):
        cv2.imwrite(f"{output_folder}/coin_{idx + 1}.png", coin)

def main(image_path):
    detected_image, coins, coin_count = detect_coins(image_path)
    
    # Save and display the detected image
    cv2.imwrite("detected_coins.png", detected_image)
    
    # Save segmented coins
    save_segmented_coins(coins)
    
    # Print the total count
    print(f"Total number of coins detected: {coin_count}")
    
    # Display the output
    cv2.imshow("Detected Coins", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "input_coins.jpg"  # Change this to your actual image path
    main(image_path)
