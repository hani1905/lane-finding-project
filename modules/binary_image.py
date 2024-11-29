import numpy as np
import cv2

def combine_thresholds(image):
    
    ##combine binary images
    mag_binary = magnitude_threshold(image, sobel_kernel=3, mag_thresh=(50,255))
    color_binary = color_threshold(image, s_thresh=(100,255), v_thresh=(50,255))
    
    combined_binary = np.zeros_like(mag_binary)
    combined_binary[(mag_binary == 1) | (color_binary == 1)] = 1

    return combined_binary

def magnitude_threshold(image, sobel_kernel=7, mag_thresh=(3, 255)):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel u x i y pravcu
    sobelX= cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Magnituda gradijenta
    magnitude = np.sqrt(sobelX**2 + sobelY**2)
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    
    # Threshold
    binary_output = np.zeros_like(gray)
    binary_output[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude <= mag_thresh[1])] = 1

    cv2.imshow("sobel", binary_output*255)
    
    return binary_output



def color_threshold(image, s_thresh=(0, 255), v_thresh=(0, 255)):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    cv2.imshow("color filter", output*255)
    return output
    
