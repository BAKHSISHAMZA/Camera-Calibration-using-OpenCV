from camera_calibration import CameraCalibrationDemo


if __name__ == "__main__":
    # 1. Set your image path
    image_path = # give the path of the image #

    # 2. Create the object
    demo = CameraCalibrationDemo(image_path)

    # 3. Find corners
    corners_found = demo.find_and_draw_corners()

    # 4. Calibrate (undistort)
    if corners_found:
        demo.calibrate()

    # 5. Show results
    demo.show_all()
