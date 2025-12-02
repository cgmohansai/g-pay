import cv2
import os

class ImageCapture:
    def __init__(self, camera_index=0):
        """
        Initialize the ImageCapture class.
        :param camera_index: Index of the camera to use (default is 0).
        """
        self.camera_index = camera_index

    def capture_frame(self):
        """
        Captures a single frame from the camera.
        :return: Captured frame (numpy array) or None if failed.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None

        ret, frame = cap.read()
        cap.release()

        if ret:
            return frame
        else:
            print("Error: Could not read frame.")
            return None

    def load_image(self, file_path):
        """
        Loads an image from a file path.
        :param file_path: Path to the image file.
        :return: Loaded image (numpy array) or None if failed.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not decode image at {file_path}")
            return None
            
        return image
