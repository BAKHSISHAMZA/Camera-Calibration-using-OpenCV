import cv2
import numpy as np
import matplotlib.pyplot as plt

class CameraCalibrationDemo:
    def __init__(self, image_path, pattern_size=(9,6)):
        self.image_path = image_path
        self.pattern_size = pattern_size
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found at the given path.")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.corners = None
        self.corners_drawn = None
        self.undistorted = None
        self.camera_matrix = None
        self.dist_coeffs = None

    def find_and_draw_corners(self):
        ret, corners = cv2.findChessboardCorners(self.gray, self.pattern_size, None)
        if ret:
            self.corners = corners
            self.corners_drawn = self.image.copy()
            cv2.drawChessboardCorners(self.corners_drawn, self.pattern_size, corners, ret)
        else:
            print("❌ Corners not found. Try a clearer or higher-res chessboard image.")
        return ret

    def calibrate(self):
        if self.corners is None:
            print("❌ Cannot calibrate without detected corners.")
            return

        # Prepare object points for the chessboard
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)

        objpoints = [objp]  # 3D points in real world
        imgpoints = [self.corners]  # 2D points in image plane

        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, self.gray.shape[::-1], None, None
        )

        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.undistorted = cv2.undistort(self.image, camera_matrix, dist_coeffs)

            print("\n✅ Calibration Successful")
            print("📷 Camera Matrix:\n", camera_matrix)
            print("\n🎯 Distortion Coefficients:\n", dist_coeffs)
        else:
            print("❌ Calibration failed.")

    def show_all(self):
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        if self.corners_drawn is not None:
            plt.imshow(cv2.cvtColor(self.corners_drawn, cv2.COLOR_BGR2RGB))
            plt.title("With Corners")
        else:
            plt.text(0.5, 0.5, "Corners not found", ha='center')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        if self.undistorted is not None:
            plt.imshow(cv2.cvtColor(self.undistorted, cv2.COLOR_BGR2RGB))
            plt.title("Undistorted Image")
        else:
            plt.text(0.5, 0.5, "Not Calibrated", ha='center')
        plt.axis("off")

        plt.tight_layout()
        plt.show()
