import os
import cv2
import modules.calibration as cal
import modules.distortion_correct as dc
import modules.binary_image as bi
import modules.birds_eye as be
import modules.find_lanes as fl
import modules.draw as rd
import modules.radius_and_position as rp
import numpy as np




def prikazi_piksele(event, x, y, flags, param):
    slika = param
    if event == cv2.EVENT_LBUTTONDOWN:  
        pixel = slika[y, x]
        print(f"Koordinate: ({x}, {y}), Vrednost piksela: {pixel}")


if __name__ == "__main__":

    file_path = "camera_cal/calib.npz"
    if not os.path.exists(file_path):
        cal.calibration()
    
   
    input_video_path = "test_videos/challenge01.mp4"  
    output_video_path = "output_video/output.mp4"  

    cap = cv2.VideoCapture(input_video_path)

    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

  
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    image = cv2.imread('test_images/straight_lines2.jpg')
    a = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  
        

        # 1. CALIBRATION - Korekcija distorzije slike
        calibrated_image = dc.correct_distrotion(frame)
        if a == 0:
            calibrated_image1 = dc.correct_distrotion(image)
            cv2.imwrite('test_images/calibrated_image2.jpg', calibrated_image1)
            a = a+1
        # 2. binary image
        binary_output = bi.combine_thresholds(calibrated_image)


        mask = np.zeros_like(binary_output)
        height, width = calibrated_image.shape[:2]
        roi_corners = np.array([[
            (50, height),
            (width - 50, height),
            (width // 2 + 100, height // 2 + 50),
            (width // 2 - 50, height // 2 + 50)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_corners, 255)
        mask1 = cv2.bitwise_and(binary_output, mask)

        if width == 1280:
        # WARPING BIRDS EYE
            pt_A = [0, height]
            pt_B = [width, height]
            pt_C = [width * 0.6 - 80, height / 2 + 90]
            pt_D = [width * 0.45 + 40, height / 2 + 90]

            src = np.float32([pt_D, pt_A, pt_B, pt_C])
            dst = np.float32([[0 + 400, 0],
                            [0 + 200, height],
                            [width - 200, height],
                            [width - 350, 0]])
        else:
            pt_A = [0, height]
            pt_B = [width, height]
            pt_C = [548,343]
            pt_D = [410,343]

            src = np.float32([pt_D, pt_A, pt_B, pt_C])
            dst = np.float32([[200, 0],
                            [0 + 150, height],
                            [width - 150, height],
                            [width - 200, 0]])

        warped = be.warper(mask1 * 255, src, dst)

        # 4. Fit and visualize lane lines
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_values, right_values,left_x,right_x = fl.fit_and_visualize(warped)

        radius, position = rp.radius_and_position_func(warped,left_values,right_values)

        # 5. Final output with warped lane lines
        final = rd.warp_back_and_draw_lines(calibrated_image, warped, left_x, right_x, ploty, src, dst,radius,position)

        cv2.imshow('Final', final)

        out.write(final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Oslobađanje resursa
    cap.release()
    out.release()
    cv2.destroyAllWindows()