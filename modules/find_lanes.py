import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

def calculate_histogram(binary_image):
    return np.sum(binary_image[binary_image.shape[0]//2:, :], axis=0)

def sliding_window(binary_image, num_windows=9, window_margin=100, min_pixels=50):
    histogram = calculate_histogram(binary_image)
    
    #lower part of histogram
    midpoint = int(histogram.shape[0] / 2)


    left_base_x = np.argmax(histogram[:midpoint])
    right_base_x = np.argmax(histogram[midpoint:]) + midpoint
    window_height = int(binary_image.shape[0] / num_windows)

    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_current_x = left_base_x
    right_current_x = right_base_x

    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        win_y_low = binary_image.shape[0] - (window + 1) * window_height
        win_y_high = binary_image.shape[0] - window * window_height
        win_xleft_low = left_current_x - window_margin
        win_xleft_high = left_current_x + window_margin
        win_xright_low = right_current_x - window_margin
        win_xright_high = right_current_x + window_margin

        valid_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        valid_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_indices.append(valid_left_indices)
        right_lane_indices.append(valid_right_indices)

        if len(valid_left_indices) > min_pixels:
            left_current_x = int(np.mean(nonzerox[valid_left_indices]))
        if len(valid_right_indices) > min_pixels:
            right_current_x = int(np.mean(nonzerox[valid_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    left_x_positions = nonzerox[left_lane_indices]
    left_y_positions = nonzeroy[left_lane_indices]
    right_x_positions = nonzerox[right_lane_indices]
    right_y_positions = nonzeroy[right_lane_indices]

    return left_x_positions, left_y_positions, right_x_positions, right_y_positions



def fit_and_visualize(binary_image):
    left_x_positions, left_y_positions, right_x_positions, right_y_positions = sliding_window(binary_image)
    left_fit_coefficients = np.polyfit(left_y_positions, left_x_positions, 2)
    right_fit_coefficients = np.polyfit(right_y_positions, right_x_positions, 2)

    ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])


    left_fit_x_values = left_fit_coefficients[0] * ploty**2 + left_fit_coefficients[1] * ploty + left_fit_coefficients[2]
    right_fit_x_values = right_fit_coefficients[0] * ploty**2 + right_fit_coefficients[1] * ploty + right_fit_coefficients[2]


   
    
    '''
    out_img = np.dstack((binary_image, binary_image, binary_image)) * 255
    plt.imshow(out_img, cmap='gray')
    plt.plot(left_fit_x_values, ploty, color='yellow', linewidth=3)
    plt.plot(right_fit_x_values, ploty, color='yellow', linewidth=3)
    plt.scatter(left_x_positions, left_y_positions, color='red', s=10, label='Left Lane Pixels')
    plt.scatter(right_x_positions, right_y_positions, color='blue', s=10, label='Right Lane Pixels')

    plt.text(50, 50, f"f_left(y) = {left_fit_coefficients[0]:.3e}y² + {left_fit_coefficients[1]:.3e}y + {left_fit_coefficients[2]:.3e}", color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    plt.text(50, 100, f"f_right(y) = {right_fit_coefficients[0]:.3e}y² + {right_fit_coefficients[1]:.3e}y + {right_fit_coefficients[2]:.3e}", color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    plt.legend()
    plt.title("Lane Detection with Fitted Curves")
    plt.xlabel("X-axis (pixels)")
    plt.ylabel("Y-axis (pixels)")
    plt.xlim(0, binary_image.shape[1])
    plt.ylim(binary_image.shape[0], 0)
    plt.show()


    # Zatvori prozor
    cv.destroyAllWindows()
    '''
    return left_fit_x_values,right_fit_x_values