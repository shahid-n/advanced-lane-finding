import numpy as np
import cv2
#import matplotlib.pyplot as plt
import moviepy.editor as mvEdit
#import matplotlib.image as mpimg
import os, pickle, sys

# pdb is used for debugging purposes; uncomment when needed
#import pdb

# Define a Line class to store the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        self.alreadyRan = False
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        # Mean-squared error in coefficients between last and best fits
        self.coeff_msErr = float(0) 
    def bestFit_calc(self):
        if not self.best_fit:
            avg = self.current_fit
        else:
            n = len(self.best_fit)
            avg = [0, 0, 0]
            for i in range(len(self.current_fit)):
                avg[i] = (self.best_fit[-1][i] * n + self.current_fit[i])\
                / (n + 1)
        return avg
    def msError_calc(self):
        error = np.array(self.current_fit) - np.array(self.best_fit[-1])
        # Return the mean-squared error
        return error.dot(error.T)

def grad_clr_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &\
             (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary),\
                              sxbinary, s_binary)) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary

def sliding_window(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) &\
                          (nonzerox < win_xleft_high) &\
                          (nonzeroy >= win_y_low) & (nonzeroy < win_y_high))\
                          .nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) &\
                           (nonzerox < win_xright_high) &\
                           (nonzeroy >= win_y_low) & (nonzeroy < win_y_high))\
                           .nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If found > minpix pixels, re-centre next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def init_fit_polynomial(binary_warped, offset):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = sliding_window(binary_warped)

    ### Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def fit_poly(img_shape, offset, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def search_around_poly(binary_warped, offset, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    win_xleft_low = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy\
                    + left_fit[2] - margin
    win_xleft_high = win_xleft_low + 2*margin
    win_xright_low =    right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy\
                        + right_fit[2] - margin
    win_xright_high = win_xright_low + 2*margin
    left_lane_inds = ((nonzerox > win_xleft_low) &\
                      (nonzerox < win_xleft_high))
    right_lane_inds = ((nonzerox > win_xright_low) &\
                       (nonzerox < win_xright_high))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit = fit_poly(binary_warped.shape, offset, leftx, lefty,\
                                   rightx, righty)
    
    return left_fit, right_fit

def measure_curvature_real(y_eval, lFit, rFit):
    '''
    Calculates the curvature of polynomial functions in metres.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # metres per pixel in y dimension
    xm_per_pix = 3.7/700 # metres per pixel in x dimension
    mid_image = 1280/2
        
    left_curverad = ((2*lFit[0]*xm_per_pix/ym_per_pix*y_eval + lFit[1]\
                      *xm_per_pix/ym_per_pix)**2 + 1)**(3/2) /\
                      np.absolute(2*lFit[0]*xm_per_pix/ym_per_pix**2)
    right_curverad = ((2*rFit[0]*xm_per_pix/ym_per_pix*y_eval + rFit[1]\
                       *xm_per_pix/ym_per_pix)**2 + 1)**(3/2) /\
                      np.absolute(2*rFit[0]*xm_per_pix/ym_per_pix**2)
    
    radius_curv = (left_curverad + right_curverad)/2
    
    xLeft = lFit[0]*y_eval**2 + lFit[1]*y_eval + lFit[2]
    xRight = rFit[0]*y_eval**2 + rFit[1]*y_eval + rFit[2]
    offset = (np.mean([xLeft, xRight]) - mid_image) * xm_per_pix
    
    return radius_curv, offset

def process_frame(image, lineL, lineR, mtx, dist):
    # Read in an image
    image_size = (image.shape[1], image.shape[0])
    
    # Undistort the image
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    clr_bin, combined = grad_clr_thresh(undist)
    
    # Define the region of interest
    srcVert = np.float32([[20, 720], [578, 450], [702, 450], [1280, 720]])
    dstVert = np.float32([[300, 720], [300, 0], [950, 0], [950, 720]])
    Mwarp = cv2.getPerspectiveTransform(srcVert, dstVert)
    Minv = cv2.getPerspectiveTransform(dstVert, srcVert)
    binary_warped = cv2.warpPerspective(combined, Mwarp, image_size,\
                                        flags=cv2.INTER_LINEAR)
    
    extplt_offset = 200
    Mtrans = np.array([[ 1, 0, 0                ],\
                       [ 0, 1, -extplt_offset   ],\
                       [ 0, 0, 1                ]])

#   The nEntries parameter dictates how many previous coefficients to store
    nEntries = 10
#   Mean square error threshold to cause the newfound polynomial to be ignored
    coeff_thresh = 5
#   Blend coefficients alpha & beta must be in the real interval [0, 1]
    alpha = 0.9
    beta = 0.7

    if not lineL.alreadyRan or not lineR.alreadyRan:
        left_fit, right_fit = init_fit_polynomial(binary_warped, extplt_offset)
        if left_fit.size == 0 or right_fit.size == 0 or lineL.coeff_msErr\
        > coeff_thresh or lineR.coeff_msErr > coeff_thresh:
            lineL.alreadyRan = False
            lineR.alreadyRan = False
            lineL.detected = False
            lineR.detected = False
        else:
            lineL.alreadyRan = True
            lineR.alreadyRan = True
            lineL.detected = True
            lineR.detected = True
            lineL.current_fit = left_fit
            lineR.current_fit = right_fit
            if not lineL.best_fit or not lineR.best_fit:
                lineL.best_fit = [lineL.bestFit_calc()]
                lineR.best_fit = [lineR.bestFit_calc()]
            else:
                if len(lineL.best_fit) >= nEntries:
                    lineL.best_fit.pop(0)
                if len(lineR.best_fit) >= nEntries:
                    lineR.best_fit.pop(0)
                lineL.best_fit.append(lineL.bestFit_calc())
                lineR.best_fit.append(lineR.bestFit_calc())
            lineL.coeff_msErr = lineL.msError_calc()
            lineR.coeff_msErr = lineR.msError_calc()
    else:
        left_fit, right_fit = search_around_poly(binary_warped, extplt_offset,\
                                                 lineL.current_fit,\
                                                 lineR.current_fit)
        if left_fit.size == 0 or right_fit.size == 0 or lineL.coeff_msErr\
        > coeff_thresh or lineR.coeff_msErr > coeff_thresh:
            if not lineL.best_fit or not lineR.best_fit:
                lineL.alreadyRan = False
                lineR.alreadyRan = False
                lineL.detected = False
                lineR.detected = False
            else:
                if len(lineL.best_fit) > 0 and len(lineR.best_fit) > 0:
                    lineL.current_fit = lineL.best_fit[-1]
                    lineR.current_fit = lineR.best_fit[-1]
                if len(lineL.best_fit) >= nEntries:
                    lineL.best_fit.pop(0)
                if len(lineR.best_fit) >= nEntries:
                    lineR.best_fit.pop(0)
                lineL.best_fit.append(lineL.bestFit_calc())
                lineR.best_fit.append(lineR.bestFit_calc())
                lineL.coeff_msErr = lineL.msError_calc()
                lineR.coeff_msErr = lineR.msError_calc()
        elif min(lineL.coeff_msErr, lineR.coeff_msErr) > coeff_thresh/2:
            left_fit = ((np.array(lineL.best_fit[-1]) + np.array(left_fit))/2)\
                        .tolist()
            right_fit = ((np.array(lineR.best_fit[-1]) + np.array(right_fit))/\
                         2).tolist()
            lineL.current_fit = left_fit
            lineR.current_fit = right_fit
            if len(lineL.best_fit) >= nEntries:
                lineL.best_fit.pop(0)
            if len(lineR.best_fit) >= nEntries:
                lineR.best_fit.pop(0)
            lineL.best_fit.append(lineL.bestFit_calc())
            lineR.best_fit.append(lineR.bestFit_calc())
            lineL.coeff_msErr = lineL.msError_calc()
            lineR.coeff_msErr = lineR.msError_calc()
        elif min(lineL.coeff_msErr, lineR.coeff_msErr) > coeff_thresh/10:
            left_fit =  (alpha*np.array(lineL.best_fit[-1])\
                         + (1 - alpha)*np.array(left_fit)).tolist()
            right_fit = (alpha*np.array(lineR.best_fit[-1])\
                         + (1 - alpha)*np.array(right_fit)).tolist()
            lineL.current_fit = left_fit
            lineR.current_fit = right_fit
            if len(lineL.best_fit) >= nEntries:
                lineL.best_fit.pop(0)
            if len(lineR.best_fit) >= nEntries:
                lineR.best_fit.pop(0)
            lineL.best_fit.append(lineL.bestFit_calc())
            lineR.best_fit.append(lineR.bestFit_calc())
            lineL.coeff_msErr = lineL.msError_calc()
            lineR.coeff_msErr = lineR.msError_calc()        
        else:
#           All checks pass, so use current_fit
            lineL.current_fit = left_fit
            lineR.current_fit = right_fit
            if len(lineL.best_fit) >= nEntries:
                lineL.best_fit.pop(0)
            if len(lineR.best_fit) >= nEntries:
                lineR.best_fit.pop(0)
            lineL.best_fit.append(lineL.bestFit_calc())
            lineR.best_fit.append(lineR.bestFit_calc())
            lineL.coeff_msErr = lineL.msError_calc()
            lineR.coeff_msErr = lineR.msError_calc()
   
    # Generate x and y values for plotting
    shp = binary_warped.shape
    ploty = np.linspace(-extplt_offset, shp[0]-1, shp[0] + extplt_offset)
    lFit = beta*np.array(lineL.best_fit[-1]) + (1 - beta)*np.array(left_fit)
    rFit = beta*np.array(lineR.best_fit[-1]) + (1 - beta)*np.array(right_fit)
    try:
        left_fitx = lFit[0]*ploty**2 + lFit[1]*ploty + lFit[2]
        right_fitx = rFit[0]*ploty**2 + rFit[1]*ploty + rFit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    # Create an image to draw the lines on
    warp_zero = np.zeros([shp[0] + extplt_offset, shp[1]]).astype(np.uint8)
    colour_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx,\
                                                 ploty + extplt_offset]))])
    pts_right = np.array([np.flipud(np.transpose(\
            np.vstack([right_fitx, ploty + extplt_offset])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(colour_warp, np.int_([pts]), (0, 255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(colour_warp, Minv.dot(Mtrans),\
                                  (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    annotated_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # Calculate the radii of curvature
    radius_curv, offset = measure_curvature_real(max(ploty), left_fit,\
                                                 right_fit)
    text1 = 'Offset from lane centre: %.2f m.' % offset
    text2 = 'Radius of Curvature: %.1f m.' % radius_curv
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    init_result = cv2.putText(annotated_img, text1, (50, 50), font, 1,\
                              (255, 255, 255), thickness, cv2.LINE_AA)
    final_result = cv2.putText(init_result, text2, (50, 100), font, 1,\
                               (255, 255, 255), thickness, cv2.LINE_AA)
    return final_result

def main(arg):
#   Debug entry point; uncomment while debugging
#    pdb.set_trace()

    in_video = os.path.abspath(arg[0])
    out_video = os.path.abspath(arg[1])
    
    if len(arg) < 2 or not os.path.isfile(in_video):
        print('One or more arguments is incorrect. Exiting.')
        sys.exit(1)
    
#    Instantiate left & right Line() objects
    lineL = Line()
    lineR = Line()
    
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( "camera_cal/dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    clip1 = mvEdit.VideoFileClip(in_video)
    annotated_clip = clip1.fl_image(lambda frame:\
                                    process_frame(frame, lineL, lineR,\
                                                  mtx, dist))
    annotated_clip.write_videofile(out_video, audio=False)

if __name__ == '__main__':
    main(sys.argv[1:])