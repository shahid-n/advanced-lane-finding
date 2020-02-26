# Advanced Lane Finding Project

The goals of this project are as follows.

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply distortion correction to raw images
* Use colour transforms, gradients, etc., to create a thresholded binary image
* Apply a perspective transform to rectify the binary image ("bird's-eye view")
* Detect lane pixels and fit a pair of polynomials to the lane boundaries
* Determine the curvature of the lane and vehicle position with respect to the centre
* Warp the detected lane boundaries back onto the original image
* Output visual markers of the lane boundaries and numerical estimation of lane curvature and vehicle position

[//]: # (Image References)

[imgChss]: ./output_images/chssbrd.jpg "Calibration"
[imgUndist]: ./output_images/undist.jpg "Undistorted"
[imgBEV]: ./output_images/bev.jpg "Road Transformed"
[imgBin]: ./output_images/combined.jpg "Binary Example"
[imgWarp]: ./output_images/bin_warped.jpg "Warp Example"
[imgPoly]: ./output_images/clr_fit.png "Fit Visual"
[imgSample]: ./output_images/final_result.jpg "Output"
[videoProj]: ./output_videos/project_video_output.mp4 "Video"

## Algorithm Description

### This section is organised into several subsections detailing key pieces of the image processing pipeline, and concludes with an explanation of how this was extended to process videos.  

---

### Camera Calibration

The code for this step is contained in the file called `dist_pickle.py`.  

The first step was to select "object points", which are the (x, y, z) coordinates of the chessboard corners in the physical world. The assumption here is that the chessboard is fixed on the (x, y) plane at z = 0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image are successfully detected.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The output `objpoints` and `imgpoints` from the above procedure were then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. This distortion correction was then applied to the test image using the `cv2.undistort()` function, and the result is depicted below.

![alt text][imgChss]


### Pipeline (single images)

#### 1. Distortion correction

The correction for the lens distortion of the camera was performed by the code contained in the files `dist_pickle.py` and `undist.py`, respectively.

The first script was used to extract the object points and image points, and to subsequently determine the distortion coefficient array and corresponding transformation matrix, which were then stored for later use in the lane finding algorithms.

The second script applies the aforementioned transformation matrix and distortion coefficients to remove the distortion in the captured images; the photo below provides some sample output of this script.

![alt text][imgUndist]


#### 2. Thresholded binary image creation

A combination of colour and gradient thresholds was then applied to generate a binary image (the relevant function is called `grad_clr_thresh(, , (, ), (, ))`, found in lines 55 through 72 in [p2_img.py](./p2_img.py)).  An example of the output generated by this function is shown below.

![alt text][imgBin]


#### 3. Perspective transformation

The code for performing the perspective transform is contained in lines 252 through 261 in the file `p2_img.py`.  The first step was to define the source (`src`) and destination (`dst`) points.  These were hardcoded in the script in the following manner.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| (560, 460)    | (300, 0)      | 
| (20, 720)     | (300, 720)    |
| (1280, 720)   | (950, 720)    |
| (720, 460)    | (950, 0)      |

It can be seen that this perspective transform works as expected since the lane lines appear parallel in the warped image.

![alt text][imgWarp]


#### 4. Lane-line identification and curve fitting

The lane identification on single, isolated images was performed using the sliding window method (`sliding_window(binary_warped)` function contained in the `p2_img.py` script). In the video pipeline, however, this method was only applied to the very first frame, and was subsequently only used as a back-up in the event that the focused search based on previously identified polynomials were to fail.

Returning to the solitary image, here is an illustration of the sliding window method in action. Note that in this script, the capability to extrapolate the polynomial beyond the topmost (or if needed, the bottom-most) window has been added.

![alt text][imgPoly]


#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to the lane centre

These calculations were both performed within the function named `measure_curvature_real(y_eval, lFit, rFit)` within the `p2_img.py` script. They could have just as easily been handled by separate functions, but it was convenient to implement within a single one. 

#### 6. Lane highlighting, unwarping and annotation

The final step was to warp the image with the fitted lane-bounded polygon back into the original perspective, followed by printing the radius of curvature and offset from centre, both approximated in units of metres using suitable conversion coefficients.

These steps were accomplished in lines 273 through 302 in `p2_img.py`.  Here is an example of such a result from a test image:

![alt text][imgSample]

---

### Pipeline (video)

#### Summary of video pipeline

The video pipeline is merely based on the single image pipeline described in the previous section; however, a couple of key differences are:
a) the presence of a more focused search method contained in the `search_around_poly(, , , )` function within [p2.py](./p2.py), and
b) the `Line()` object class, which was used to keep track of previously fitted polynomials along with a moving average of these coefficients, which could then be substituted in place of missing lane pixels, if any, or a poorly fit polynomial (as measured in terms of the overall mean-squared error between the moving average values and the most recent set of coefficients).

Below is a list of all the key functions in the `p2.py` video pipeline.

1. `Line()` class -- to keep track of best fit curves and to apply smoothing
2. `grad_clr_thresh` function -- same as the single image case
3. `sliding_window(binary_warped)` -- only used in the very first frame, or to subsequently recover from failed searches for a polynomial similar to the one from the previous frame
4. `search_around_poly(binary_warped, offset, left_fit, right_fit)` -- the usual search method for the bulk of the video; `offset` is an extrapolation parameter to extend the highlighted region
5. `measure_curvature_real(y_eval, lFit, rFit)` -- calculate the radius of curvature and lane offset annotations
6. `process_frame(image, lineL, lineR, mtx, dist)` -- this is the main processing function that contains most of the high level logic for handling edge cases, and for calling all of the lower level processing and calculation functions in the pipeline
7. `main(arg)` -- this accepts command line arguments for the input and output videos, instantiates two copies (left and right) of the `Line()` class, loads the camera calibration parameters, and then hands off control to the `moviepy.editor.fl_image` and `process_frame` functions, respectively

In order to view the main project demonstration, please follow this [link to the video result.](./output_videos/project_video_output.mp4)

---

### Discussion and concluding remarks

#### 1. Highlights and key techniques

The advanced edge detection, colour space theory and thresholding, perspective transform and curve fitting techniques were critical to the success of this project.

Whilst experimenting with the various techniques and parameters that control the underlying calculations and algorithms, I found that it was possible to extrapolate the lane tracing polynomial to extend the highlighted region, although this approach is sensitive to image noise and can lead to unstable behaviour near the top of the highlighted region in some instances.

It was also possible to apply additional smoothing and to tune this using suitable new parameters, in order to achieve an appropriate trade-off between accurate fit to the curvature detected in the distant field on the one hand, and overall smoothness and stabilty of the lane fit polygon on the other.

The performance of my `p2.py` lane finding pipeline on the challenge video, whilst not as successful as the main project video, nevertheless displays some promise by being able to adapt to an unusually and rapidly altering host lane. The result of my attempt to process the challenge video can be seen [here.](./output_videos/challenge_video_output.mp4)

#### 2. Limitations and possible future improvements

Whilst this project proved to be a significant improvement over the first lane finding endeavour, it was made abundantly clear by the two challenge videos that many hurdles remain. A couple of them are discussed here.

1. Extreme changes in contrast are clearly still an issue, as are lane splits and merges, not to mention highly worn or faded lane markers, which would require more intelligent processing. One measure that might lead to significant improvement with respect to the first challenge video would be to compare the edge detections to the lane centre position, rather than simply latching on to the two highest peaks in the gradient histogram, without regard to their actual position in the image. Since the US DOT mandated standard lane width specification is a known quantity, this fact can be exploited to reject strong detections which are too close to the centre of the image and thus clearly inside the occupied lane -- but of course, this logic would necessarily have to be disabled during lane changing manoeuvres.

2. Reflections and extreme road curvature are also formidable challenges to overcome, hopefully sooner rather than later. In my mind the curvature issue seems like the easier one to address, by simply increasing the order of the lane fit polynomials to make them cubic -- however, proper tuning of the relatively large number of algorithm parameters to consistently obtain successful, not to mention correct, lane detections and curve fits in the majority of the frames is easier said than done. 
