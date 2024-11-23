import os
import cv2
import modules.calibration as cal
import modules.distortion_correct as dc
import modules.binary_image as bi
import modules.birds_eye as be
import numpy as np


# Funkcija za prikaz informacija o pikselima
def prikazi_piksele(event, x, y, flags, param):
    slika = param
    if event == cv2.EVENT_LBUTTONDOWN:  # Kada kliknete levim tasterom miša
        pixel = slika[y, x]
        print(f"Koordinate: ({x}, {y}), Vrednost piksela: {pixel}")


if __name__ == "__main__":

    ## CALIBRATION 1. ##
    file_path = "camera_cal/calib.npz"
    if not os.path.exists(file_path):
        cal.calibration()
    
    ## Distortion apply, only to some photos ##
    image = cv2.imread("test_images/straight_lines1.jpg")
        
    calibrated_image = dc.correct_distrotion(image)


    ## creating binary image
    binary_output = bi.combine_thresholds(calibrated_image)

    ## results
    cv2.imshow('Original Image', image)
    cv2.imshow('Calibrated Image', calibrated_image)
    cv2.imshow('Binary Image', binary_output * 255) 

    cv2.setMouseCallback('Calibrated Image', prikazi_piksele, calibrated_image)


    mask = np.zeros_like(binary_output)
    height, width = calibrated_image.shape[:2]
    roi_corners = np.array([[
        (50, height),                 # Donja leva tačka
        (width - 50, height),         # Donja desna tačka
        (width // 2 + 100, height // 2 + 50),  # Gornja desna tačka
        (width // 2 - 50, height // 2 + 50)   # Gornja leva tačka
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi_corners, 255)
    mask1 = cv2.bitwise_and(binary_output, mask)

    cv2.imshow('Binary Image1', mask1 * 255) 
    line_dst_offset = 200


    #################################WARPING BIRDS EYE#################################


    # [cols, rows]
    #works for 1280x720 
    #TODO 960x540
    pt_A = [0, height]
    pt_B = [width,height]
    pt_C = [width * 0.6 - 80, height/2 + 90]
    pt_D = [width * 0.45 + 20, height/2 + 90]

    
    src = np.float32([pt_D,pt_A,pt_B,pt_C])
    dst = np.float32([[0+400, 0],
                        [0+200, height],
                        [width-200, height],
                        [width - 350, 0]])
    
    
    warped  = be.warper(mask1*255,src,dst)

    cv2.imshow('warped', warped) 

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    