import numpy as np

import numpy as np

def radius_and_position_func(binary_warped, left_fit, right_fit):

    if binary_warped.shape[0] == 540:
        # Conversion factors for pixels to meters
        ym_per_pix = 30 / 540  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 480  # meters per pixel in x dimension (average lane width is 3.7 meters)
    else:
        ym_per_pix = 30 / 720 
        xm_per_pix = 3.7 / 700
    # Generate y values (vertical points on the image)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)  # Evaluate curvature at the bottom of the image
    
    # Calculate x coordinates of the bottom of the left and right lane lines
    left_x_bottom = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x_bottom = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    
    # Calculate the center of the lane
    lane_center = (left_x_bottom + right_x_bottom) / 2
    
    # Calculate the vehicle's position in pixels
    image_center = binary_warped.shape[1] / 2  # Assume the camera is at the center of the image
    vehicle_offset_pixels = image_center - lane_center  # Positive means left, negative means right
    
    # Convert the offset to meters
    vehicle_offset_meters = vehicle_offset_pixels * xm_per_pix

    # Calculate curvature for the left and right lane lines in real-world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, 
                             (left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]) * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, 
                              (right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]) * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])
    
    # Combine left and right curvature as the average
    curvature_radius = (left_curverad + right_curverad) / 2

    return curvature_radius, vehicle_offset_meters

