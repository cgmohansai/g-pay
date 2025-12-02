import cv2
import numpy as np
import os

def create_mock_image():
    # Create a black image
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Draw a palm (ellipse)
    cv2.ellipse(img, (300, 400), (100, 120), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw Wrist (Rectangle extending to bottom)
    cv2.rectangle(img, (250, 450), (350, 600), (255, 255, 255), -1)
    
    # Draw 5 fingers
    # Thumb (Left)
    cv2.line(img, (220, 350), (150, 250), (255, 255, 255), 40)
    cv2.circle(img, (150, 250), 20, (255, 255, 255), -1)
    
    # Index
    cv2.line(img, (250, 300), (220, 150), (255, 255, 255), 35)
    cv2.circle(img, (220, 150), 18, (255, 255, 255), -1)
    
    # Middle
    cv2.line(img, (300, 290), (300, 100), (255, 255, 255), 35)
    cv2.circle(img, (300, 100), 18, (255, 255, 255), -1)
    
    # Ring
    cv2.line(img, (350, 300), (380, 150), (255, 255, 255), 35)
    cv2.circle(img, (380, 150), 18, (255, 255, 255), -1)
    
    # Little
    cv2.line(img, (380, 350), (450, 250), (255, 255, 255), 30)
    cv2.circle(img, (450, 250), 15, (255, 255, 255), -1)
    
    # Draw some "veins"
    cv2.line(img, (300, 400), (300, 350), (50, 50, 50), 4)
    cv2.line(img, (300, 400), (280, 380), (50, 50, 50), 4)
    cv2.line(img, (300, 400), (320, 380), (50, 50, 50), 4)

    os.makedirs("data", exist_ok=True)
    cv2.imwrite("data/test_hand.jpg", img)
    print("Created data/test_hand.jpg")

if __name__ == "__main__":
    create_mock_image()
