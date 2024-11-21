import numpy as np
import cv2

def correct_distrotion():
    # Load calibrated camera parameters
    calibratio = np.load('camera_cal/calib.npz')
    mtx = calibratio['mtx']
    dist = calibratio['dist']
    rvecs = calibratio['rvecs']
    tvecs = calibratio['tvecs']

    # Load one of the test images
    img = cv2.imread('camera_cal/calibration1.jpg')
    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    # Display the final result
    cv2.imshow('chess board', np.hstack((img, undistortedImg)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()