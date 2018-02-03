from glob import glob
import cv2
import pickle
import numpy as np
import os.path
import collections
import matplotlib.pyplot as plt

##############################
##### CAMERA CALIBRATION #####
##############################

calibration_path = './calibrationValues.p'
# Tries to read camera calibration values from file, and if it exists, reads them
if os.path.isfile(calibration_path):
    f = open(calibration_path, "rb")
    [mtx, dist] = pickle.load(f)
    f.close()
else:
    #Run the camera calibration again if the pickled value file does not exist
    # Shape of chessboard
    cx = 9
    cy = 6
     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cy*cx,3), np.float32)
    objp[:,:2] = np.mgrid[0:cx, 0:cy].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    calibration_images = glob('./camera_cal/calibration*.jpg')
    for index, file in enumerate(calibration_images):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray image needed for findChessboardCorners
        ret, corners = cv2.findChessboardCorners(gray, (cx, cy), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    img_shape = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    for idx, file in enumerate(calibration_images):
        img = cv2.imread(file)
        undist = cv2.undistort(img, mtx, dist)
        cv2.imwrite('./camera_cal/undist/undist_{}.jpg'.format(idx), undist)
    # Save the camera calibration values to a path
    f = open(calibration_path, "wb")
    pickle.dump([mtx, dist], f)
    f.close()

### Thresholding to detect lines on HLS and BGR colorspace
    
def hls_rgb_threshold(img, hls_s_thresh = (0, 255), luv_l_thresh = (0, 255), lab_b_thresh = (0, 255)):
    size = img.shape[0]*img.shape[1]
    
    # Threshold b channel of Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float)
    lab_b_channel = lab[:,:,2]
    total_brgt_lab_b = np.sum(lab_b_channel)
    avg_brgt_lab_b = total_brgt_lab_b / size #average brightness per pixel in LAB B channel
    lab_b_lower_thresh = 1.1 * avg_brgt_lab_b
    if lab_b_lower_thresh < lab_b_thresh[0]:
        lab_b_lower_thresh = lab_b_thresh[0]
    color_mask_lab_b= np.zeros_like(lab_b_channel)
    color_mask_lab_b[(lab_b_channel > lab_b_lower_thresh) & (lab_b_channel <= lab_b_thresh[1])] = 1    
    
    # Threshold the S channel from HLS color space
    max_brightness_hls = 100000
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    hls_s_channel = hls[:,:,2]
    total_brgt_hls_s = np.sum(hls_s_channel)
    avg_brgt_hls_s =  total_brgt_hls_s / size #average brightness per pixel in HLS S channel
    hls_s_lower_thresh = 1.4 * avg_brgt_hls_s
    if hls_s_lower_thresh < hls_s_thresh[0]:
        hls_s_lower_thresh = hls_s_thresh[0]
    color_mask_hls_s = np.zeros_like(hls_s_channel)
    color_mask_hls_s[(hls_s_channel > hls_s_lower_thresh) & (hls_s_channel <= hls_s_thresh[1])] = 1
    total_brgt_s = np.sum(color_mask_hls_s)
    if total_brgt_s > max_brightness_hls:
        color_mask_hls_s = np.zeros_like(hls_s_channel)
    
    # Threshold the L channel from LUV color space
    max_brightness_luv = 40000
    midpoint = img.shape[1] // 2
    luv = 1
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV).astype(np.float)
    luv_l_channel = luv[:,:,0]
    total_brgt_luv_l = np.sum(luv_l_channel)
    avg_brgt_luv_l = total_brgt_luv_l / size #average brightness per pixel in l channel
    luv_l_lower_thresh = 1.3 * avg_brgt_luv_l
    if luv_l_lower_thresh < luv_l_thresh[0]:
        luv_l_lower_thresh = luv_l_thresh[0]     
    color_mask_luv_l = np.zeros_like(luv_l_channel)
    color_mask_luv_l[(luv_l_channel > luv_l_lower_thresh) & (luv_l_channel <= luv_l_thresh[1])] = 1
    color_mask_luv_l[:,:midpoint] = 0
    total_brgt_l = np.sum(color_mask_luv_l)
    if total_brgt_l > max_brightness_luv:
        color_mask_luv_l = np.zeros_like(luv_l_channel)
 
    combined_mask = np.zeros_like(hls_s_channel)
    combined_mask[(color_mask_hls_s == 1) | (color_mask_lab_b == 1) | (color_mask_luv_l == 1)] = 1
    
    # Write Images from different color masks for debug purposes
    file_img ='./output/IM_'+str(i)+'.jpg'
    file_luv_l = './output/LUV_L_'+str(i)+'.jpg'
    file_hls_s = './output/HLS_S_'+str(i)+'.jpg'
    file_lab_b = './output/LAB_B_'+str(i)+'.jpg'   
    file_c = './output/Comb_'+str(i)+'.jpg'
    luv_l_img = np.dstack((color_mask_luv_l, color_mask_luv_l, color_mask_luv_l))*255
    hls_s_img = np.dstack((color_mask_hls_s, color_mask_hls_s, color_mask_hls_s))*255
    lab_b_img = np.dstack((color_mask_lab_b, color_mask_lab_b, color_mask_lab_b))*255
    comb_img = np.dstack((combined_mask, combined_mask, combined_mask))*255
    cv2.imwrite(file_img, img)
    cv2.imwrite(file_luv_l, luv_l_img)
    cv2.imwrite(file_hls_s, hls_s_img)
    cv2.imwrite(file_lab_b, lab_b_img)
    cv2.imwrite(file_c, comb_img)
    #print ('LAB B:', total_brgt_lab_b, ' HLS S:', total_brgt_hls_s, ' LUV L:', total_brgt_l)
    
    return combined_mask

### Applies new search window when no lanes were detected yet, 
### or the search was lost in the last frame ###
def new_search_window(undist, warped):
    iy = int(warped.shape[0])
    # Take a histogram of the bottom half of the image
    # The peaks of the histogram (left and right of the midpoint)
    # are used as the starting point for the search window
    histogram = np.sum(warped[int(iy/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2) # middle of the image
    leftx_base = np.argmax(histogram[:midpoint]) # index (x) of the left peak - left lane
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # index (x) of the right peak
    nwindows = 9 # Number of sliding windows vertically
    window_height = np.int(iy / nwindows) # Height of one sliding window
    # The input to this function is the warped image, which is already thresholded
    # most values are zero (black), and only those pixels of interest (plus some noise)
    # is left in the image. These pixels we can find with the nonzero function
    nonzero = warped.nonzero() # Find indices of all nonzero elements 
    nonzeroy = np.array(nonzero[0]) # y positions of all nonzero pixels
    nonzerox = np.array(nonzero[1]) # x positions of all nonzero pixels
    leftx_current = leftx_base # sets current position to starting point
    rightx_current = rightx_base
    window_width = 80
    minpix = 50 # Minimum amount of pixels found to recenter window
    left_lane_indx = []
    right_lane_indx = []
    # Finds all indices associated with the left and right lane
    for window in range(nwindows):
        win_y_low = iy - (window+1) * window_height  # y value of the upper border of the search window
        win_y_high = iy - window * window_height # y value of lower border of the search window
        win_xleft_low = leftx_current - window_width # left window border of left lane
        win_xleft_high = leftx_current + window_width # right windows border of left lane
        win_xright_low = rightx_current - window_width # left window border of right lane
        win_xright_high = rightx_current + window_width # right window border of right lane
        # This function finds all nonzero elements within the search window border
        good_left_indx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) \
                          & (nonzerox < win_xleft_high)).nonzero()[0] # for left lane
        good_right_indx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) \
                           & (nonzerox < win_xright_high)).nonzero()[0] # for right lane
        left_lane_indx.append(good_left_indx)
        right_lane_indx.append(good_right_indx)
        # If a minimum amount of pixels per search window is exceeded, the search window x position is
        # reset to a new position. This is especially important in curved lanes, where from bottom to top
        # the x position varies widely.
        if len(good_left_indx) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_indx]))
        if len(good_right_indx) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_indx]))
    left_lane_indx = np.concatenate(left_lane_indx)
    right_lane_indx = np.concatenate(right_lane_indx)
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx] 
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]
    # Using the found lane pixels, a polynomial fit is applied to left and right lane and returned
    # to the calling function
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def follow_search_window(warped, left_fit, right_fit, margin=100):
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Finds all indices associated with the left and right lane within the area of the margin around the previously found lane fit
    left_lane_indx = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_indx = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx] 
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]
    if len(leftx) >= 10:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) >= 10:
        right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def plot_lane_warped(warped, left_fit, right_fit, margin=100):
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    #out_img[nonzeroy[left_lane_indx], nonzerox[left_lane_indx]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_indx], nonzerox[right_lane_indx]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result_warped = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    return result_warped
    
def plot_lane_image(undist, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    #warp_zero = np.zeros_like(undist).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = np.zeros_like(undist)
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result_undist = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    radius1, radius2, offset = curvature(left_fit, right_fit, y_eval = (undist.shape[0]-1))
    # Plot the curvature radius on the image
    text = 'Left radius: {}m -- Right radius: {}m -- Offset: {:.2} m'.format(int(radius1), int(radius2), offset)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    thickness = 1
    #baseline = 0
    #textSize = cv2.getTextSize(text, fontFace, fontScale, thickness, baseline)
    textOrg = (int(undist.shape[1]/10), int(undist.shape[0]/10))
    #textOrg2 = (int(undist.shape[1]/10), int(undist.shape[0]/10)+100) # print offset 100pixel below radius
    cv2.putText(result_undist, text, textOrg, fontFace, fontScale, (255, 255, 255), thickness, 8)
    
    return result_undist

def average_line_fits(last_left_fits, last_right_fits):
    # prepares new pixel spaces to plot the average line plots on
    ploty = np.linspace(0, 719, num=720, dtype=int)
    left_fitx = np.zeros_like(ploty)
    right_fitx = np.zeros_like(ploty)
    # Adds one pixel (in x) per lane fit (10 last frames)
    for i in ploty:
        lefti = []
        righti= []
        weights_left = []
        weights_right = []        
        for index, elem in enumerate(last_left_fits):
            lefti.append(int(elem[0]*i**2 + elem[1]*i + elem[2]))
            weights_left.append(index+1) # Weights so that the newest frame has weight 10, the last frame weight 1
        # Find the average weighted x values of the last 10 lanes
        left_fitx[i] = np.average(lefti, axis=0, weights=weights_left) # left lane
        for index, elem in enumerate(last_right_fits):
            righti.append(int(elem[0]*i**2 + elem[1]*i + elem[2]))
            weights_right.append(index+1)
        right_fitx[i] = np.average(righti, axis=0, weights=weights_right) # right lane
    left_fit = np.polyfit(ploty, left_fitx, 2) # new averaged fit for the left lane
    right_fit = np.polyfit(ploty, right_fitx, 2) # right lane
    return left_fit, right_fit
 
    # Calculates the curvature from the left and right lane fit and an evaluation point y_eval
def curvature(left_fit, right_fit, y_eval):
    ploty = np.linspace(0, 719, num=720, dtype=int) # Generates new y array for the left and right lane
    # Generates pixels for the left and right lane fit
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Converts the bottom width of the lane at y_eval to a 3.7m wide standard lane
    xm_per_pix = 3.7 / 700 # meters per pixel in x direction
    ym_per_pix = 30.0 / 720 # meters per pixel in y direction
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Formula for calculating the radius from the first and second derivative of a function-lane fit 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Calculates the offset by finding the center pixel between the two lanes, then measuring the distance
    # from the true center in the middle of the image
    center_pix_offset = ((rightx[-1] + leftx[-1]) / 2) - (1280/2)
    offset = center_pix_offset * xm_per_pix # transfers offset from pixels to meters
    return left_curverad, right_curverad, offset

def sanity_check(last_left_fits, last_right_fits, left_fit, right_fit):
    #Check for approximate parallelity of the two lanes
    #if not ((np.abs((left_fit[0]-right_fit[0])/right_fit[0]) < 0.1) & (np.abs((left_fit[1]-right_fit[1])/right_fit[1]) < 0.1)):
    #        print('Sanity check for parallelity of left and right lane failed')
    #        return False
    #previous_left_fit = last_left_fits[-1]
    #previous_right_fit = last_right_fits[-1]
    #left_shape_sanity = cv2.matchShapes(left_fit, previous_left_fit, cv.CV_CONTOURS_MATCH_I1, 0.0)
    #right_shape_sanity = cv2.matchShapes(right_fit, previous_right_fit, cv.CV_CONTOURS_MATCH_I1, 0.0)
    #print(left_shape_sanity, right_shape_sanity)
    return True
    
    
# Read in a video file with all properties like frame rate, shape and length
video_file_out = 'project_output.mp4'
video_file_in = 'project_video.mp4'
capture = cv2.VideoCapture(video_file_in)
frame_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
vid_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_fps = int(capture.get(cv2.CAP_PROP_FPS))
print ('Video length is ', frame_length, ' frames, size is ', vid_width, 'x', vid_height, 'Frame rate is', vid_fps, 'fps')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter(video_file_out, fourcc, vid_fps, (vid_width, vid_height))
print('Video could be opened correctly', outvideo.isOpened())

# Calculation of transformation matrix from image (src) to warped (dst) space
# M is the transformation matrix from image to warped space
# Minv is the inverse, transforming back from warped space to image space
src = np.float32([(570,465),(712,465),(1054,677),(253,677)])
dst = np.float32([(320,0), (960,0),(960,719),(320,719)])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

# Stores ten last historical fits, to be able to smooth over them
last_left_fits = collections.deque(maxlen=5)
last_right_fits = collections.deque(maxlen=5)

sanity = True
for i in range(frame_length):
    ret, img = capture.read()
    if ret == True:
        undist = cv2.undistort(img, mtx, dist)
        hls_s_thr = (50, 255)
        luv_l_thr = (150, 255)
        lab_b_thr = (80, 255)
        warped = cv2.warpPerspective(undist, M, (vid_width, vid_height))
        hls_rgb_mask = hls_rgb_threshold(warped, hls_s_thr, luv_l_thr, lab_b_thr)
        if (i == 0) | (sanity == False):
            left_fit, right_fit = new_search_window(undist, hls_rgb_mask)
            last_left_fits.append(left_fit)
            last_right_fits.append(right_fit)
        else:
            left_fit, right_fit = follow_search_window(hls_rgb_mask, left_fit, right_fit, margin=100)
        if (len(last_left_fits)>=1) & (len(last_right_fits) >=1):
            sanity = sanity_check(last_left_fits, last_right_fits, left_fit, right_fit)
            if sanity == True:
                last_left_fits.append(left_fit)
                last_right_fits.append(right_fit)
            else:
                left_fit = last_left_fits[-1]
                right_fit = last_right_fits[-1]
        left_fit, right_fit = average_line_fits(last_left_fits, last_right_fits)
        output_undist = plot_lane_image(undist, left_fit, right_fit, Minv)
        outvideo.write(np.uint8(output_undist))
        print('Frame', i)
    else:
        break

capture.release()
outvideo.release()

'''
test_image='./test_images/straight_lines1.jpg'
img = cv2.imread(test_image)
undist = cv2.undistort(img, mtx, dist)
s_thr = (50, 255)
l_thr = (150, 255)
rg_thr = (130, 255)
hls_rgb_mask = hls_rgb_threshold(undist, region_thresholds, s_thr, l_thr, rg_thr)
warped = cv2.warpPerspective(hls_rgb_mask, M, (vid_width, vid_height))
left_fit, right_fit = new_search_window(undist, warped)
output_undist = plot_lane_image(undist, left_fit, right_fit, Minv)
cv2.imwrite('./example_output_lane.jpg', output_undist)
'''