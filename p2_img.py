import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        xDir = 1
        yDir = 0
    else:
        xDir = 0
        yDir = 1
    sobelPder = cv2.Sobel(gray, cv2.CV_64F, xDir, yDir, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobelPder)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude 
    sobelGrad = np.sqrt(sobelX**2 + sobelY**2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelGrad/np.max(sobelGrad))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(sobelGrad)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelX = np.absolute(sobelX)
    abs_sobelY = np.absolute(sobelY)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    sobelDir = np.arctan2(abs_sobelY, abs_sobelX)
    # 5) Create a binary mask where thresholds are met
    binary_output = np.zeros_like(sobelDir)
    binary_output[(sobelDir >= thresh[0]) & (sobelDir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Edit this function to create your own pipeline.
def grad_clr_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary

def sliding_window(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
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
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
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

    return leftx, lefty, rightx, righty, out_img


def init_fit_polynomial(binary_warped, offset):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = sliding_window(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(-offset, binary_warped.shape[0]-1,\
                        binary_warped.shape[0] + offset )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit

def measure_curvature_real(y_eval, lFit, rFit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    mid_image = 1280/2
        
    left_curverad = ((2*lFit[0]*xm_per_pix/ym_per_pix*y_eval + lFit[1]\
    *xm_per_pix/ym_per_pix)**2 + 1)**(3/2) /\
                      np.absolute(2*lFit[0]*xm_per_pix/ym_per_pix**2)
    right_curverad = ((2*rFit[0]*xm_per_pix/ym_per_pix*y_eval + rFit[1]\
    *xm_per_pix/ym_per_pix)**2 + 1)**(3/2) /\
                      np.absolute(2*rFit[0]*xm_per_pix/ym_per_pix**2)
    
    xLeft = lFit[0]*y_eval**2 + lFit[1]*y_eval + lFit[2]
    xRight = rFit[0]*y_eval**2 + rFit[1]*y_eval + rFit[2]
    offset = (np.mean([xLeft, xRight]) - mid_image) * xm_per_pix
    
    return left_curverad, right_curverad, offset

# Read in an image
image = mpimg.imread('test_images/test6.jpg')
image_size = (image.shape[1], image.shape[0])
# Read in an image
#img = cv2.imread('camera_cal/calibration5.jpg')

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "camera_cal/dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Undistort the image
undist = cv2.undistort(image, mtx, dist, None, mtx)

# Choose a Sobel kernel size
#ksize = 15 # Choose a larger odd number to smooth gradient measurements

## Apply the thresholding function
clr_bin, combined = grad_clr_thresh(undist)

# Define the region of interest
srcVert = np.float32([[20, 720], [560, 460], [720, 460], [1280, 720]])
pvert = np.vstack([srcVert, srcVert[0, :]])
dstVert = np.float32([[300, 720], [300, 0], [950, 0], [950, 720]])
#dvert = np.vstack([dstVert, dstVert[0, :]])
Mwarp = cv2.getPerspectiveTransform(srcVert, dstVert)
Minv = cv2.getPerspectiveTransform(dstVert, srcVert)
bev = cv2.warpPerspective(undist, Mwarp, image_size, flags=cv2.INTER_LINEAR)
binary_warped = cv2.warpPerspective(combined, Mwarp, image_size,\
                                    flags=cv2.INTER_LINEAR)

# Amount of extrapolation applied when drawing the lane curves
extplt_offset = 480
# Translation post-multiplication matrix to be used to apply the extrapolation
# offset defined immediately above
Mtrans = np.array([[ 1, 0, 0                ],\
                   [ 0, 1, -extplt_offset   ],\
                   [ 0, 0, 1                ]])
out_img, left_fitx, right_fitx,\
ploty, left_fit, right_fit = init_fit_polynomial(binary_warped, extplt_offset)

# Create an image to draw the lines on
shp = binary_warped.shape
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
rCurveL, rCurveR, offset = measure_curvature_real(max(ploty), left_fit,\
                                                  right_fit)
printout = 'Offset: %.2f m. Radii of Curvature: Left: %.1f m, Right: %.1f m.'\
% (offset, rCurveL, rCurveR)

font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
final_result = cv2.putText(annotated_img, printout, (50, 50), font, 1,\
                           (255, 255, 255), thickness, cv2.LINE_AA)

plt.close('all')
plt.style.use('dark_background')
f1to4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
#f.tight_layout()
ax1.imshow(clr_bin)
ax1.set_title('Coloured Binary', fontsize=20)
ax2.imshow(combined, cmap='gray')
ax2.plot(pvert[:, 0], pvert[:, 1], 'r--')
ax2.set_title('Combined Output', fontsize=20)
ax3.imshow(image)
ax3.set_title('Original', fontsize=20)
ax4.imshow(bev)
ax4.plot(dstVert[:, 0], dstVert[:, 1], 'r')
ax4.set_title('Bird\'s Eye View', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.05)

f5, (ax5, ax6) = plt.subplots(1, 2)
ax5.imshow(out_img)
# Plot the left and right polynomials on the lane lines
ax5.plot(left_fitx, ploty, color='yellow')
ax5.plot(right_fitx, ploty, color='yellow')
ax6.imshow(final_result)