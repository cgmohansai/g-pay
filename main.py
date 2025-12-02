import cv2
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.acquisition import ImageCapture
from src.preprocessing import PalmPreprocessor

def main():
    parser = argparse.ArgumentParser(description="GlyphPay Palm Vein Preprocessing")
    parser.add_argument("--image", type=str, help="Path to input image (optional). If not provided, uses camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    # Initialize modules
    capture = ImageCapture(camera_index=args.camera)
    preprocessor = PalmPreprocessor()

    # Acquire image
    if args.image:
        print(f"Loading image from {args.image}...")
        image = capture.load_image(args.image)
    else:
        print("Capturing from camera...")
        image = capture.capture_frame()

    if image is None:
        print("Error: Failed to acquire image.")
        return

    # Process image
    print("Processing image...")
    results = preprocessor.process(image)

    if results is None:
        print("Error: Preprocessing failed (Not enough defects or ROI extraction failed).")
        return

    # Visualization using Matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Original Image with Geometric Construction
    img_rgb = cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB)
    ax[0].imshow(img_rgb)
    ax[0].set_title("Geometric ROI Extraction")

    # Draw Contour
    contour = results["contour"]
    ax[0].plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, label='Contour')

    # Draw All Defect Candidates
    all_defects = results.get("all_defects", [])
    if all_defects:
        for pt in all_defects:
            ax[0].scatter(pt[0], pt[1], c='red', s=20, marker='x', alpha=0.7)
        # Add a single legend entry for candidates
        ax[0].scatter([], [], c='red', s=20, marker='x', label='All Candidates')

    # Draw Selected Defects
    debug_info = results["debug_info"]
    if debug_info:
        d1 = debug_info["d1"]
        d2 = debug_info["d2"]
        d3 = debug_info["d3"]
        d4 = debug_info["d4"]
        midpoint = debug_info["midpoint"]
        dir_line2 = debug_info["dir_line2"]
        dir_line3 = debug_info["dir_line3"]
        intersection = debug_info["intersection"]
        angle = debug_info["angle"]

        # Plot points with labels
        ax[0].scatter(d1[0], d1[1], c='yellow', s=100, marker='*', label='D1 (Thumb)')
        ax[0].text(d1[0], d1[1]-10, 'D1', color='yellow', fontsize=12, fontweight='bold')
        
        ax[0].scatter(d2[0], d2[1], c='cyan', s=50, label='D2')
        ax[0].text(d2[0], d2[1]-10, 'D2', color='cyan', fontsize=12, fontweight='bold')
        
        ax[0].scatter(d3[0], d3[1], c='cyan', s=50, label='D3')
        ax[0].text(d3[0], d3[1]-10, 'D3', color='cyan', fontsize=12, fontweight='bold')
        
        ax[0].scatter(d4[0], d4[1], c='cyan', s=50, label='D4')
        ax[0].text(d4[0], d4[1]-10, 'D4', color='cyan', fontsize=12, fontweight='bold')
        
        ax[0].scatter(midpoint[0], midpoint[1], c='orange', s=50, marker='o', label='Midpoint')

        # Draw Line 1 (D2-D4)
        ax[0].plot([d2[0], d4[0]], [d2[1], d4[1]], 'b-', linewidth=2, label='Line 1')

        # Draw Line 2 (Perpendicular Bisector)
        scale = 300
        p1 = midpoint - scale * dir_line2
        p2 = midpoint + scale * dir_line2
        ax[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=2, label='Line 2')

        # Draw Line 3 (Perpendicular from D1 to Line 2)
        p3 = d1 - scale * dir_line3
        p4 = d1 + scale * dir_line3
        ax[0].plot([p3[0], p4[0]], [p3[1], p4[1]], 'm--', linewidth=2, label='Line 3')

        # Draw Intersection
        ax[0].scatter(intersection[0], intersection[1], c='lime', s=150, marker='X', label='Center')

        # Draw Rotated ROI Box
        # We need to draw a rectangle centered at 'intersection', size 255x255, rotated by 'angle'
        # Matplotlib Rectangle takes (bottom-left) corner, width, height, angle
        # We need to calculate bottom-left corner of the unrotated rectangle relative to center, then rotate it?
        # Actually, Rectangle(xy, width, height, angle=angle, rotation_point='center') is not standard.
        # Standard is rotation around xy.
        
        # Let's calculate the 4 corners manually.
        size = 255
        half = size / 2
        # Unrotated corners relative to center
        corners = np.array([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half]
        ])
        
        # Rotation matrix
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        # Rotate corners
        rotated_corners = np.dot(corners, R.T)
        
        # Translate to center
        final_corners = rotated_corners + intersection
        
        # Draw polygon
        poly = patches.Polygon(final_corners, linewidth=2, edgecolor='white', facecolor='none', label='Rotated ROI')
        ax[0].add_patch(poly)

    ax[0].legend()

    # 2. Extracted ROI and Enhanced Veins
    veins_rgb = cv2.cvtColor(results["enhanced_veins"], cv2.COLOR_BGR2RGB)
    ax[1].imshow(veins_rgb, cmap='gray')
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.acquisition import ImageCapture
from src.preprocessing import PalmPreprocessor

def main():
    parser = argparse.ArgumentParser(description="GlyphPay Palm Vein Preprocessing")
    parser.add_argument("--image", type=str, help="Path to input image (optional). If not provided, uses camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    # Initialize modules
    capture = ImageCapture(camera_index=args.camera)
    preprocessor = PalmPreprocessor()

    # Acquire image
    if args.image:
        print(f"Loading image from {args.image}...")
        image = capture.load_image(args.image)
    else:
        print("Capturing from camera...")
        image = capture.capture_frame()

    if image is None:
        print("Error: Failed to acquire image.")
        return

    # Process image
    print("Processing image...")
    results = preprocessor.process(image)

    if results is None:
        print("Error: Preprocessing failed (Not enough defects or ROI extraction failed).")
        return

    # Visualization using Matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Original Image with Geometric Construction
    img_rgb = cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB)
    ax[0].imshow(img_rgb)
    ax[0].set_title("Geometric ROI Extraction")

    # Draw Contour
    contour = results["contour"]
    ax[0].plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, label='Contour')

    # Draw All Defect Candidates
    all_defects = results.get("all_defects", [])
    if all_defects:
        for pt in all_defects:
            ax[0].scatter(pt[0], pt[1], c='red', s=20, marker='x', alpha=0.7)
        # Add a single legend entry for candidates
        ax[0].scatter([], [], c='red', s=20, marker='x', label='All Candidates')

    # Draw Selected Defects
    debug_info = results["debug_info"]
    if debug_info:
        d1 = debug_info["d1"]
        d2 = debug_info["d2"]
        d3 = debug_info["d3"]
        d4 = debug_info["d4"]
        midpoint = debug_info["midpoint"]
        dir_line2 = debug_info["dir_line2"]
        dir_line3 = debug_info["dir_line3"]
        intersection = debug_info["intersection"]
        angle = debug_info["angle"]

        # Plot points with labels
        ax[0].scatter(d1[0], d1[1], c='yellow', s=100, marker='*', label='D1 (Thumb)')
        ax[0].text(d1[0], d1[1]-10, 'D1', color='yellow', fontsize=12, fontweight='bold')
        
        ax[0].scatter(d2[0], d2[1], c='cyan', s=50, label='D2')
        ax[0].text(d2[0], d2[1]-10, 'D2', color='cyan', fontsize=12, fontweight='bold')
        
        ax[0].scatter(d3[0], d3[1], c='cyan', s=50, label='D3')
        ax[0].text(d3[0], d3[1]-10, 'D3', color='cyan', fontsize=12, fontweight='bold')
        
        ax[0].scatter(d4[0], d4[1], c='cyan', s=50, label='D4')
        ax[0].text(d4[0], d4[1]-10, 'D4', color='cyan', fontsize=12, fontweight='bold')
        
        ax[0].scatter(midpoint[0], midpoint[1], c='orange', s=50, marker='o', label='Midpoint')

        # Draw Line 1 (D2-D4)
        ax[0].plot([d2[0], d4[0]], [d2[1], d4[1]], 'b-', linewidth=2, label='Line 1')

        # Draw Line 2 (Perpendicular Bisector)
        scale = 300
        p1 = midpoint - scale * dir_line2
        p2 = midpoint + scale * dir_line2
        ax[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=2, label='Line 2')

        # Draw Line 3 (Perpendicular from D1 to Line 2)
        p3 = d1 - scale * dir_line3
        p4 = d1 + scale * dir_line3
        ax[0].plot([p3[0], p4[0]], [p3[1], p4[1]], 'm--', linewidth=2, label='Line 3')

        # Draw Intersection
        ax[0].scatter(intersection[0], intersection[1], c='lime', s=150, marker='X', label='Center')

        # Draw Rotated ROI Box
        # We need to draw a rectangle centered at 'intersection', size 255x255, rotated by 'angle'
        # Matplotlib Rectangle takes (bottom-left) corner, width, height, angle
        # We need to calculate bottom-left corner of the unrotated rectangle relative to center, then rotate it?
        # Actually, Rectangle(xy, width, height, angle=angle, rotation_point='center') is not standard.
        # Standard is rotation around xy.
        
        # Let's calculate the 4 corners manually.
        size = debug_info.get("roi_size", 1000)
        half = size / 2
        # Unrotated corners relative to center
        corners = np.array([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half]
        ])
        
        # Rotation matrix
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        # Rotate corners
        rotated_corners = np.dot(corners, R.T)
        
        # Translate to center
        final_corners = rotated_corners + intersection
        
        # Draw polygon
        poly = patches.Polygon(final_corners, linewidth=2, edgecolor='white', facecolor='none', label='Rotated ROI')
        ax[0].add_patch(poly)

    ax[0].legend()

    # 2. Extracted ROI and Enhanced Veins
    veins_rgb = cv2.cvtColor(results["enhanced_veins"], cv2.COLOR_BGR2RGB)
    ax[1].imshow(veins_rgb, cmap='gray')
    ax[1].set_title("Enhanced Veins (Aligned)")

    plt.tight_layout()
    plt.show()

    # Save results
    os.makedirs("data", exist_ok=True)
    
    # Step 1: Cropped Input
    cv2.imwrite("data/step1_cropped_input.jpg", results["original"])
    
    # Step 2: ROI (Fixed 750px)
    cv2.imwrite("data/step2_roi_750px_fixed.jpg", results["roi"])
    
    # Step 3: CLAHE Output (Grayscale Veins)
    cv2.imwrite("data/step3_clahe_output.jpg", results["clahe_veins"])
    
    # Step 4: Final Binary Stencil
    cv2.imwrite("data/step4_final_binary.jpg", results["enhanced_veins"])
    
    print("Results saved to data/:")
    print(" - step1_cropped_input.jpg")
    print(" - step2_roi_750px_fixed.jpg")
    print(" - step3_clahe_output.jpg")
    print(" - step4_final_binary.jpg")
    print(" - step5_visualization.png")

    # Save visualization
    plt.savefig("data/step5_visualization.png")
    plt.show()

if __name__ == "__main__":
    main()
