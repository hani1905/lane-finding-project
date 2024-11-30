# lane-finding-project

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Set of chessboard images with visible pattern are used to compute the camera matrix and distortion coefficients, but firstly detect the corners of the chessboard. Then 3D point were paired with the 2D points for each calibration image. Cv.calibrateCamera() function was used to compute the camera matrix and distortion coefficients. The result is stored in camera_cal/calib.npz. 

<p align="center">
  <img src="camera_cal/calibration1.jpg" alt="Image 1" width="30%"/>
  <img src="calibrated_image.jpg" alt="Image 2" width="30%"/>
</p>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
When we use cv2.undistort(img, mtx, dist, None, newCameraMtx), we can choose if we want to use new optimal camera matrix provided from newCameraMtx = cv2.getOptimalNewCameraMatrix() or unchanged input matrix. In this project i used optimal, but instead of newCameraMtx we can simply use mtx for unchanged matrix.

<p align="center">
  <img src="test_images/straight_lines2.jpg" alt="Image 1" width="30%"/>
  <img src="calibrated_image1.jpg" alt="Image 2" width="30%"/>
  <img src="calibrated_image2.jpg" alt="Image 3" width="30%"/>
</p>

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Here is a brief explanation of the code used to create a thresholded binary image:

The color_threshold function processes the image in two color spaces, HLS and HSV. It extracts the saturation (S) channel from HLS and the value (V) channel from HSV, applying thresholds to each. The resulting binary masks are combined to isolate pixels based on their color and brightness.

The magnitude_threshold function calculates the gradient magnitude of the image using Sobel operators in the x and y directions. The combined gradient magnitude is then scaled and thresholded to produce a binary image that highlights edges based on changes in intensity.

The combine_thresholds function creates a binary image by combining two separate thresholding methods: one based on gradient magnitude and the other on color filtering. It merges the results into a single binary mask, where pixels are set to 1 if they meet the criteria from either method.

<p align="center">
  <img src="test_images/straight_lines2.jpg" alt="Image 1" width="30%"/>
  <img src="binary_image.jpg" alt="Image 2" width="30%"/>
</p>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The first step in performing a perspective (birds-eye) transform is to define the source and destination points. The source point were made by manually selecting the four corners of the road in the image so we made so called Region of Interes. The destination point were selected by manually selecting the four corners of the road in the birds-eye view. SOurce and destination points are different for different image size.

Then we perform the perspective transform with OpenCV cv2.getPerspectiveTransform() and cv.warpPerspective().

Later on in draw.py i used cv2.getPerspectiveTransform() again to inverse matrix in order to print lines on normal perspective.

<p align="center">
  <img src="test_images/straight_lines1.jpg" alt="Image 1" width="30%"/>
  <img src="birdseye.jpg" alt="Image 1" width="30%"/>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

