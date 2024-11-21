import os
import modules.calibration as cal
import modules.distortion_correct as dc

if __name__ == "__main__":

    ## CALIBRATION 1. ##
    file_path = "camera_cal/calib.npz"
    if not os.path.exists(file_path):
        print(f"Fajl '{file_path}' ne postoji. Pozivam funkciju da ga kreira...")
        cal.calibration()
    else:
        print(f"Fajl '{file_path}' već postoji. Preskačem kreiranje.")


    ## Distortion apply, only to some photos ##
    dc.correct_distrotion()