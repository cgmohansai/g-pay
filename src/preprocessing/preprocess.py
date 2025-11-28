import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def prompt_selected_person(default_person: str = "007") -> str:
    """
    Ask the user which person ID to process. Returns a zero-padded ID.
    """
    user_input = input(f"Enter person ID to process (e.g. 007) [{default_person}]: ").strip()
    if not user_input:
        user_input = default_person
    # Normalize to at least 3 digits (dataset naming convention)
    return user_input.zfill(3)


def main():
    # Resolve project root: this file is src/preprocessing/preprocess.py
    project_root = Path(__file__).resolve().parents[2]

    # dataset/images/ relative to project root
    image_directory = str(project_root / "dataset" / "images") + os.sep

    selected_person = prompt_selected_person()
    selected_hand = "l"
    selected_spectrum = "940"

    pattern = f"{selected_person}_{selected_hand}_{selected_spectrum}_*.jpg"
    matching_files = glob.glob(image_directory + pattern)
    selected_files = matching_files[:3]

    if not selected_files:
        print(f"[WARN] No files found matching {pattern} in {image_directory}")
        return

    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 24))
    fig2, axes2 = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))

    output_directory = project_root /"dataset" / "img_out"
    output_directory.mkdir(parents=True, exist_ok=True)

    for idx, file_path in enumerate(selected_files):
        person_id, hand, spectrum, number = (file_path.split("/")[-1]).split("_")
        image_stem = Path(file_path).stem
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Initial crop: Remove wrist area (bottom portion) to focus on palm only
        # Crop from bottom to remove wrist imperfections
        wrist_crop_height = int(image.shape[0] * 0.25)  # Remove bottom 25% (wrist area)
        palm_image = image[:image.shape[0] - wrist_crop_height, :]

        # Step 1: Create threshold image from cropped palm image
        blurred = cv2.GaussianBlur(palm_image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Step 2: Create contour from threshold image
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print(f"[WARN] No contours found for {file_path}")
            continue
        
        largest_contour = max(contours, key=cv2.contourArea)

        # Original image
        axes[0, idx].imshow(image, cmap='gray')
        axes[0, idx].set_title(f"Original {number, person_id}")
        axes[0, idx].axis('off')

        # Cropped palm image (wrist removed)
        axes[1, idx].imshow(palm_image, cmap='gray')
        axes[1, idx].set_title(f"Cropped Palm (Wrist Removed) {number, person_id}")
        axes[1, idx].axis('off')

        # Thresholded image
        axes[2, idx].imshow(thresholded, cmap='gray')
        axes[2, idx].set_title(f"Thresholded {number, person_id}")
        axes[2, idx].axis('off')

        # Step 3: Draw contour
        image_with_contours = np.copy(palm_image)
        cv2.drawContours(image_with_contours, [largest_contour], -1, (255, 255, 255), 2)

        # Image with contours
        axes[3, idx].imshow(image_with_contours, cmap='gray')
        axes[3, idx].set_title(f"Largest Contour {number, person_id}")
        axes[3, idx].axis('off')

        # Step 4: Calculate convexity defects
        defects_image = np.copy(image_with_contours)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        
        if len(hull) < 4:
            print(f"[WARN] Not enough points in hull for {file_path}")
            continue
            
        defects = cv2.convexityDefects(largest_contour, hull)
        
        if defects is None or len(defects) == 0:
            print(f"[WARN] No defects found for {file_path}")
            continue

        # Sort defects based on depth (deepest first)
        defects_list = sorted(defects, key=lambda x: x[0, 3], reverse=True)

        # Collect defect points (top 4 defects) and keep segments for context
        raw_defect_points = []
        defect_colors = [
            (255, 0, 0),    # defect 1 - blue channel
            (0, 255, 0),    # defect 2 - green
            (0, 0, 255),    # defect 3 - red
            (255, 255, 0),  # defect 4 - yellow
        ]
        for i in range(min(6, len(defects_list))):
            s, e, f, d = defects_list[i][0]
            start = tuple(largest_contour[s][0])
            end = tuple(largest_contour[e][0])
            far = tuple(largest_contour[f][0])
            raw_defect_points.append(far)

            cv2.line(defects_image, start, end, [0, 255, 0], 2)

        # Suppress duplicates so that closely spaced defect points collapse to a single candidate.
        defect_far_points = []
        min_distance = max(12, palm_image.shape[1] // 40)
        for far in raw_defect_points:
            far_arr = np.array(far, dtype=np.int32)
            if all(np.linalg.norm(far_arr - np.array(existing)) > min_distance for existing in defect_far_points):
                defect_far_points.append(tuple(far_arr))
            if len(defect_far_points) == 4:
                break

        if len(defect_far_points) < 4:
            defect_far_points = raw_defect_points[:4]

        for i, far in enumerate(defect_far_points):
            point_color = defect_colors[i % len(defect_colors)]
            cv2.circle(defects_image, far, 8, point_color, -1)

            label = f"({far[0]}, {far[1]})"
            cv2.putText(
                defects_image,
                label,
                (far[0] + 10, far[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        palm_center = None
        roi_vertices = None
        roi_size = None

        # Draw a white line between the 2nd and 4th defect points
        # Here "2nd" and "4th" correspond to indices 0 and 3 in defect_far_points
        # (based on your manual inspection). Also draw a perpendicular through
        # the midpoint of this line.
        if len(defect_far_points) >= 4:
            second_defect_far = defect_far_points[0]
            fourth_defect_far = defect_far_points[3]

            # Main line between 2nd and 4th defects
            cv2.line(defects_image, second_defect_far, fourth_defect_far, (255, 255, 255), 2)

            # Geometry for perpendicular lines
            x1, y1 = second_defect_far
            x2, y2 = fourth_defect_far
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length > 0:
                # Midpoint of the main line
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

                # Unit direction along the main line and its perpendicular
                ux_main = dx / length
                uy_main = dy / length
                ux_perp = -uy_main
                uy_perp = ux_main

                # First perpendicular: through the midpoint of the 2nd–4th line
                half_len = 0.4 * length
                p1 = (int(mid_x - ux_perp * half_len), int(mid_y - uy_perp * half_len))
                p2 = (int(mid_x + ux_perp * half_len), int(mid_y + uy_perp * half_len))
                cv2.line(defects_image, p1, p2, (255, 255, 255), 2)

                # Second perpendicular: from defect_far_points[1] to intersect the main line
                px, py = defect_far_points[1]
                apx = px - x1
                apy = py - y1
                t = (apx * ux_main + apy * uy_main)
                qx = x1 + ux_main * t
                qy = y1 + uy_main * t
                # Extend this perpendicular slightly beyond the intersection so the join is clearer
                perp_dx = qx - px
                perp_dy = qy - py
                perp_len = np.hypot(perp_dx, perp_dy)
                if perp_len > 0:
                    extend_factor = 0.2 * length
                    scale = (perp_len + extend_factor) / perp_len
                    ex = px + perp_dx * scale
                    ey = py + perp_dy * scale
                    cv2.line(defects_image, (px, py), (int(ex), int(ey)), (255, 255, 255), 2)
                else:
                    cv2.line(defects_image, (px, py), (int(qx), int(qy)), (255, 255, 255), 2)

                # Third line: from defect_far_points[2] to intersect the midpoint line at 90°
                # Draw a segment parallel to the main line so it meets the midpoint
                # perpendicular (p1–p2) exactly at right angle.
                third_px, third_py = defect_far_points[2]
                delta_x = mid_x - third_px
                delta_y = mid_y - third_py
                t3 = delta_x * ux_main + delta_y * uy_main
                ix = third_px + ux_main * t3
                iy = third_py + uy_main * t3
                q3 = (int(ix), int(iy))
                cv2.line(defects_image, (third_px, third_py), q3, (255, 255, 255), 2)

                # Use the intersection point of defect[2]'s line as the ROI center
                palm_center = q3
                roi_size = int(length * 0.8)
                roi_size = max(50, min(roi_size, 200))

                roi_half = roi_size / 2.0
                square_local = np.array(
                    [
                        [-roi_half, -roi_half],
                        [roi_half, -roi_half],
                        [roi_half, roi_half],
                        [-roi_half, roi_half],
                    ],
                    dtype=np.float32,
                )
                rotation_matrix = np.array(
                    [
                        [ux_main, ux_perp],
                        [uy_main, uy_perp],
                    ],
                    dtype=np.float32,
                )
                rotated_square = square_local @ rotation_matrix.T
                roi_vertices = (rotated_square + np.array(palm_center)).astype(np.int32)

        # Defects image
        axes[4, idx].imshow(defects_image, cmap='gray')
        axes[4, idx].set_title(f"Defects {number, person_id}")
        axes[4, idx].axis('off')

        # Fallback: if we could not build ROI from defect geometry, revert to centroid-based square
        if roi_vertices is None or roi_size is None or palm_center is None:
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                print(f"[WARN] Cannot calculate centroid for {file_path}")
                continue

            palm_center_x = int(M["m10"] / M["m00"])
            palm_center_y = int(M["m01"] / M["m00"])
            palm_center = (palm_center_x, palm_center_y)

            x, y, w, h = cv2.boundingRect(largest_contour)
            roi_size = min(w, h) // 2
            roi_size = max(50, min(roi_size, 200))

            roi_half = roi_size // 2
            roi_vertices = np.array([
                [palm_center_x - roi_half, palm_center_y - roi_half],
                [palm_center_x + roi_half, palm_center_y - roi_half],
                [palm_center_x + roi_half, palm_center_y + roi_half],
                [palm_center_x - roi_half, palm_center_y + roi_half],
            ], dtype=np.int32)

        # Draw ROI on defects image
        cv2.circle(defects_image, palm_center, 5, [0, 0, 255], -1)  # Mark palm center
        cv2.polylines(
            defects_image,
            [roi_vertices],
            isClosed=True,
            color=[255, 0, 0],
            thickness=2,
        )

        # Image with contours, defects, and ROI
        axes[5, idx].imshow(defects_image, cmap='gray')
        axes[5, idx].set_title(f"Defects with ROI (Palm Center) {number, person_id}")
        axes[5, idx].axis('off')

        # Extract ROI from original image
        roi_vertices_float = roi_vertices.astype(np.float32)
        rectified_order = np.array(
            [[0, 0], [roi_size, 0], [roi_size, roi_size], [0, roi_size]],
            dtype=np.float32,
        )

        # Perspective transformation to extract and rectify ROI
        transform_matrix = cv2.getPerspectiveTransform(
            roi_vertices_float,
            rectified_order,
        )
        rectified_image = cv2.warpPerspective(palm_image, transform_matrix, (roi_size, roi_size))
        rectified_image_equalized = cv2.equalizeHist(rectified_image)

        # Image with Extracted ROI and no changes
        axes2[0, idx].imshow(rectified_image_equalized, cmap='gray')
        axes2[0, idx].set_title(f"Extracted ROI {number, person_id}")
        axes2[0, idx].axis('off')

        roi_raw_output = output_directory / f"{image_stem}_roi_raw.png"
        cv2.imwrite(str(roi_raw_output), rectified_image_equalized)

        g_kernel_size = 5
        g_sigma = 2.5
        g_theta = np.pi / 3
        g_lambda = 8.0
        g_gamma = 0.4
        g_psi = 0.0

        # Create the Gabor kernel
        gabor_kernel = cv2.getGaborKernel(
            (g_kernel_size, g_kernel_size),
            g_sigma,
            g_theta,
            g_lambda,
            g_gamma,
            g_psi,
            ktype=cv2.CV_32F,
        )
        filtered_veins = cv2.filter2D(
            rectified_image_equalized,
            cv2.CV_32F,
            gabor_kernel,
        )

        # Normalize the filtered image
        filtered_veins = cv2.normalize(
            filtered_veins,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )

        # Image with extracted ROI and gabor filter applied
        axes2[1, idx].imshow(filtered_veins, cmap='gray')
        axes2[1, idx].set_title(f"Gabor filter on ROI {number, person_id}")
        axes2[1, idx].axis('off')

        # Apply thresholding and CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        clahe_veins = clahe.apply(filtered_veins)
        clahe_blurred = cv2.GaussianBlur(clahe_veins, (5, 5), 0)  # gaussian blur
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        clahe_veins = clahe.apply(clahe_blurred)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_veins = clahe.apply(clahe_veins)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        clahe_veins = clahe.apply(clahe_veins)

        # CLAHE veins
        axes2[2, idx].imshow(clahe_veins, cmap='gray')
        axes2[2, idx].set_title(f"CLAHE filtered Veins {number, person_id}")
        axes2[2, idx].axis('off')

        _, binary_veins = cv2.threshold(clahe_veins, 110, 255, cv2.THRESH_BINARY)

        # Filtered veins
        axes2[3, idx].imshow(binary_veins, cmap='gray')
        axes2[3, idx].set_title(f"Filtered Veins {number, person_id}")
        axes2[3, idx].axis('off')

        binary_output = output_directory / f"{image_stem}_roi_binary.png"
        cv2.imwrite(str(binary_output), binary_veins)

    fig.tight_layout()
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
