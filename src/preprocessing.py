import cv2
import numpy as np
import math
import itertools

class PalmPreprocessor:
    def __init__(self):
        pass

    def extract_roi(self, image):
        # 1. Convert to Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. Wrist Crop (User Logic: Remove bottom 25%)
        h, w = gray.shape
        wrist_crop_height = int(h * 0.25)
        cropped_gray = gray[:h - wrist_crop_height, :]
        
        # Keep a copy of the cropped original for returning
        cropped_original = image[:h - wrist_crop_height, :] if len(image.shape) == 3 else cropped_gray

        # 3. Thresholding
        blurred = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        hand_contour = max(contours, key=cv2.contourArea)

        return hand_contour, thresh, cropped_original

    def get_defects(self, contour, image_shape):
        """
        Finds convexity defects using User's Logic:
        1. Sort by Depth (Deepest first).
        2. Suppress duplicates based on distance.
        3. Return top 4.
        """
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) <= 3:
             return None, None

        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return None, None
            
        # Sort defects based on depth (deepest first) - User Logic
        # defects shape is (N, 1, 4) -> [start, end, far, depth]
        # depth is at index 3
        defects_list = sorted(defects, key=lambda x: x[0, 3], reverse=True)
        
        raw_defect_points = []
        for i in range(min(10, len(defects_list))): # Check top 10 deep defects
            s, e, f, d = defects_list[i][0]
            if d > 1000: # Keep a basic noise threshold
                far = tuple(contour[f][0])
                raw_defect_points.append(far)

        # Suppress duplicates - User Logic
        defect_far_points = []
        min_distance = max(12, image_shape[1] // 40)
        
        for far in raw_defect_points:
            far_arr = np.array(far, dtype=np.int32)
            # Check distance against all currently selected points
            if all(np.linalg.norm(far_arr - np.array(existing)) > min_distance for existing in defect_far_points):
                defect_far_points.append(tuple(far_arr))
            
            if len(defect_far_points) == 4:
                break
        
        return defect_far_points, raw_defect_points

    def calculate_geometric_center(self, defect_points):
        """
        Calculates geometric center for a given set of 4 defects.
        Assumes defects are sorted Left-to-Right (D1, D2, D3, D4).
        """
        d1 = np.array(defect_points[0])
        d2 = np.array(defect_points[1])
        d3 = np.array(defect_points[2])
        d4 = np.array(defect_points[3])
        
        # Line 1: Connect D2 and D4
        midpoint_d2_d4 = (d2 + d4) / 2.0
        vec_d2_d4 = d4 - d2
        
        vx, vy = vec_d2_d4
        mag = np.sqrt(vx**2 + vy**2)
        if mag == 0: return None, 0, None
        vx /= mag
        vy /= mag
        
        # Line 2: Perpendicular bisector (Direction: -vy, vx)
        dir_line2 = np.array([-vy, vx])
        
        # Line 3: Perpendicular to Line 2 (Parallel to Line 1), passing through D1
        # Direction is (vx, vy)
        
        # Intersection
        vec_m_d1 = d1 - midpoint_d2_d4
        t = np.dot(vec_m_d1, dir_line2)
        
        intersection = midpoint_d2_d4 + t * dir_line2
        
        angle_rad = math.atan2(vy, vx)
        angle_deg = math.degrees(angle_rad)
        
        # Fixed ROI Size: 750px
        roi_size = 750
        
        debug_info = {
            "d1": d1, "d2": d2, "d3": d3, "d4": d4,
            "midpoint": midpoint_d2_d4,
            "dir_line2": dir_line2,
            "dir_line3": np.array([vx, vy]),
            "intersection": intersection,
            "angle": angle_deg,
            "roi_size": roi_size
        }
        
        return intersection, angle_deg, debug_info, roi_size

    def extract_roi_square(self, image, center, size=255, angle=0):
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
        
        h, w = image.shape[:2]
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        x, y = center
        half = size // 2
        
        x1 = int(x - half)
        y1 = int(y - half)
        x2 = x1 + size
        y2 = y1 + size
        
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)
        
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = rotated_image[src_y1:src_y2, src_x1:src_x2]
            
        return canvas

    def enhance_veins(self, image):
        """
        Applies the 4-step vein enhancement pipeline:
        1. Histogram Equalization
        2. Gabor Filter
        3. CLAHE (Multi-pass)
        4. Binary Thresholding
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Invert so veins (dark) become bright features for Gabor
        gray = cv2.bitwise_not(gray)

        # Step 1: Histogram Equalization
        equalized = cv2.equalizeHist(gray)

        # Step 2: Multi-Angle Gabor Filter
        g_kernel_size = 5
        g_sigma = 2.5
        g_lambda = 8.0
        g_gamma = 0.4
        g_psi = 0.0
        
        # Accumulate max response from multiple angles
        final_filtered = np.zeros_like(equalized, dtype=np.float32)
        
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            gabor_kernel = cv2.getGaborKernel(
                (g_kernel_size, g_kernel_size),
                g_sigma,
                theta,
                g_lambda,
                g_gamma,
                g_psi,
                ktype=cv2.CV_32F,
            )
            filtered = cv2.filter2D(equalized, cv2.CV_32F, gabor_kernel)
            final_filtered = np.maximum(final_filtered, filtered)
        
        # Normalize to 0-255
        filtered_veins = cv2.normalize(final_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Step 3: CLAHE (Multi-pass)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        clahe_veins = clahe.apply(filtered_veins)
        
        clahe_blurred = cv2.GaussianBlur(clahe_veins, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        clahe_veins = clahe.apply(clahe_blurred)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_veins = clahe.apply(clahe_veins)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        clahe_veins = clahe.apply(clahe_veins)
        
        # Step 3.5: 5th CLAHE Pass (Zoom In/Perfect Detail)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        clahe_veins = clahe.apply(clahe_veins)

        # Step 4: Binary Thresholding (Otsu's Method)
        # Use Otsu to automatically find the best threshold
        # THRESH_BINARY_INV + THRESH_OTSU -> Black strokes on white
        _, binary_veins = cv2.threshold(clahe_veins, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary_veins, clahe_veins

    def process(self, image):
        if image is None: return None

        # 1. Extract ROI (with Wrist Crop)
        contour, binary, cropped_original = self.extract_roi(image)
        if contour is None: return None

        # 2. Get Defects (User Logic: Depth Sort + Suppression)
        defect_points, all_candidates = self.get_defects(contour, cropped_original.shape)
        
        if defect_points is None or len(defect_points) < 4:
            print(f"Not enough defects found. Found: {len(defect_points) if defect_points else 0}")
            return None

        # 3. Sort Selected Defects Left-to-Right (Geometric Labeling)
        # D1 = Leftmost, D4 = Rightmost
        sorted_defects = sorted(defect_points, key=lambda p: p[0])
        
        # 4. Calculate Center
        center, angle, debug_info, roi_size = self.calculate_geometric_center(sorted_defects)
        
        if center is None: return None
            
        # 5. Extract Final ROI Square
        roi = self.extract_roi_square(cropped_original, center, size=roi_size, angle=angle)
        
        enhanced_veins, clahe_veins = self.enhance_veins(roi)
        
        return {
            "original": cropped_original, # Return the cropped image as "original" for visualization
            "roi": roi,
            "contour": contour,
            "defects": sorted_defects,
            "all_defects": all_candidates,
            "center": center.astype(int),
            "enhanced_veins": enhanced_veins,
            "clahe_veins": clahe_veins,
            "debug_info": debug_info,
            "binary": binary
        }
