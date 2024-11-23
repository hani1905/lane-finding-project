import numpy as np
import cv2

def correct_distrotion(img):
    # Load calibrated camera parameters
    calibratio = np.load('camera_cal/calib.npz')
    mtx = calibratio['mtx']
    dist = calibratio['dist']
    rvecs = calibratio['rvecs']
    tvecs = calibratio['tvecs']

    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    return undistortedImg