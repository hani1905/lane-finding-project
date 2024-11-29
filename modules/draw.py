import numpy as np
import cv2 as cv
from modules.birds_eye import warper as warp

def warp_back_and_draw_lines(original_image, binary_warped, left_fit_x, right_fit_x, ploty, src, dst):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_points = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    points = np.hstack((left_points, right_points))
    
    
    cv.fillPoly(color_warp, np.int_([points]), (0, 255, 0))
    
    # Iscrtavanje linija
    for i in range(len(left_fit_x) - 1):
        # left line
        cv.line(color_warp, 
                (int(left_fit_x[i]), int(ploty[i])), 
                (int(left_fit_x[i + 1]), int(ploty[i + 1])), 
                (0, 0, 255), 50)  # Žuta boja, debljina 10
        
        # right line
        cv.line(color_warp, 
                (int(right_fit_x[i]), int(ploty[i])), 
                (int(right_fit_x[i + 1]), int(ploty[i + 1])), 
                (0, 0, 255), 50)  # Crvena boja, debljina 10

    
    Minv = cv.getPerspectiveTransform(dst, src)  
    unwarped = cv.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))

    # linije plus originalna slika
    result = cv.addWeighted(original_image, 1, unwarped, 0.3, 0)
    return result
