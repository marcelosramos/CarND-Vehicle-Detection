import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

YM_PER_PIX = 30/700 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension

def calibrate_cam():
    '''Compute the undistortion matrices from a series of chessboard images'''
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints

def undistort_img(image, objpoints, imgpoints):
    '''Undistort an img based on the given undistortion parameters'''
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist

def eq_histogram(image):
    '''Equilize the histogram of each color channel separately'''
    image = np.copy(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(40,40))
    image[:,:,0] = clahe.apply(image[:,:,0])
    image[:,:,1] = clahe.apply(image[:,:,1])
    image[:,:,2] = clahe.apply(image[:,:,2])
    return image

def white_threshold(hls, thresh=(235, 255)):
    '''Creates a binary image applying a threshold on the l channel. 
    This functions expects an image in the HLS colorspace
    '''
    hls = np.copy(hls)
    # Convert to HLS color space and separate the V channel
    #hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1] 
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh[0]) & (l_channel <= thresh[1])] = 1
    return l_binary  

def yellow_threshold(hls, thresh_s=(100, 250), thresh_h=(10, 22)):
    '''Creates a binary image applying a threshold on the s and the h channels. 
    This functions expects an image in the HLS colorspace
    '''
    hls = np.copy(hls)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1  
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= thresh_h[0]) & (h_channel <= thresh_h[1])] = 1
    combined = np.zeros_like(h_binary)
    combined[((s_binary == 1) & (h_binary == 1))] = 1
    return combined 

def combine_threshold(image):
    '''Creates a binary image combining the white_threshold and the yellow_threshold functions. 
    This functions expects an image in the HLS colorspace
    '''
    #image = get_histogram(image)
    #return np.dstack((np.zeros_like(image[:,:,0]), combined_sobel_threshold(image), s_channel_threshold(image))).astype(np.uint8)  * 255
    combined = np.zeros_like(image[:,:,0])
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[(yellow_threshold(image)==1) | (white_threshold(image)==1)] = 1
    return combined

def perspective_transform(image, inverse=False):
    '''Does a perspective transformation to get the bird's-eye view of an image.
    The inverse parameter indicates whether to do the forward transform getting the bird's-eye view of an image
    or to do the inverse transform getting an image on the original perspective from a bird's-eye view image'''
    src = np.float32([[521,498],[759,498],[271,665],[1009,665]])
    dst = np.float32([[290,500],[990,500],[290,720],[990,720]])
    if not inverse:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(image, M, (1280,720), flags=cv2.INTER_LINEAR)

class Line():
    def __init__(self):
        self.x = []
        self.y = []
        self.fit = [0, 0, 0]
        self.fit_cr = [0, 0, 0]
        self.ploty = np.linspace(0, 720-1, 720 )
        self.fitx = np.zeros_like(self.ploty)
        self.radius = 0
        self.n = 20
        self.last_fits = []
        self.best_fit = [0, 0, 0]
        self.best_fitx = np.zeros_like(self.ploty)
        self.last_radius = []
        self.best_radius = 0
        self.fail_count = 0

    def set_new_points(self, x, y):
        self.x = x
        self.y = y
        min_points_check = True
        radius = 0
        if len(y) > 50:
            self.fit = np.polyfit(self.y, self.x, 2)
            self.fitx = self.fit[0]*self.ploty**2 + self.fit[1]*self.ploty + self.fit[2]
            y_eval = np.max(self.ploty)
            if self.fit[0] != 0:
                 self.compute_radius()
            else: 
                min_points_check = False
        else:
            min_points_check = False
        return min_points_check, self.fitx, self.radius
            
    def compute_radius(self):
        y_eval = np.max(self.ploty)
        if len(self.y) > 0:
            self.fit_cr = np.polyfit(self.y*YM_PER_PIX, self.x*XM_PER_PIX, 2)
        if self.fit_cr[0] != 0:
            self.radius = ((1 + (2*self.fit_cr[0]*y_eval*YM_PER_PIX + self.fit_cr[1])**2)**1.5) / np.absolute(2*self.fit_cr[0])
            if self.fit_cr[0] < 0 and self.radius < 3000:
                self.radius = -self.radius
        else: 
            self.radius = 0
         
    def update_last_fits(self, sanity_check = True):
        if len(self.last_fits) == self.n:
            self.last_fits = self.last_fits[1:]
            self.last_radius = self.last_radius[1:]
        if sanity_check:
            self.last_fits.append(self.fit)
            self.last_radius.append(self.radius)
        elif len(self.last_fits)>0:
            self.last_fits.append(self.last_fits[-1])
            self.last_radius.append(self.last_radius[-1])
        if len(self.last_fits)>0:
            self.best_fit = np.mean(self.last_fits, axis=0)
            self.best_radius = np.mean(self.last_radius, axis=0)
            if abs(self.best_radius) > 3000 and self.radius < 0:
                self.best_radius = -self.best_radius
            self.best_fitx = self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2]            #self.compute_radius()
        
    def reset(self):
        self.x = []
        self.y = []
        self.fit = [0, 0, 0]
        self.fit_cr = [0, 0, 0]
        self.ploty = np.linspace(0, 720-1, 720 )
        self.fitx = np.zeros_like(self.ploty)
        self.fitx_cr = np.zeros_like(self.ploty)
        self.radius = 0
        self.n = 10
        self.last_fits = []
        self.best_fit = [0, 0, 0]
        self.best_fitx = np.zeros_like(self.ploty)
        self.best_fitx_cr = np.zeros_like(self.ploty)
        self.last_radius = 0 
        self.best_radius = 0 
        self.fail_count = 0
        
class Lane():
    def __init__(self):       
        self.left_line = Line()
        self.right_line = Line()
        self.radius = 0
        self.off_center = 0
        self.miss_count = 0
    
    def set_new_left_points(self, x, y):
        self.left_line.set_new_points(x, y)

    def set_new_right_points(self, x, y):
        self.right_line.set_new_points(x, y)
        
    def compute_data(self):
        self.radius = (self.left_line.best_radius + self.right_line.best_radius)/2
        img_center = 1280/2.
        lane_center = (self.left_line.best_fitx[-1] + self.right_line.best_fitx[-1])/2.
        self.off_center = (img_center - lane_center)*XM_PER_PIX
      
    def check_sanity(self):
        min_points = 50
        min_lane_width = 2.7/XM_PER_PIX
        max_lane_width = 4.7/XM_PER_PIX
        sanity = True
        if len(self.left_line.y) < min_points or len(self.right_line.y) < min_points:
            sanity = False
        elif ((self.left_line.radius < 0 and self.right_line.radius > 0 and self.right_line.radius < 3000) \
             or (self.left_line.radius < 3000 and self.left_line.radius > 0 and self.right_line.radius < 0)): \
            sanity = False
        elif ((self.right_line.fitx[-1] - self.left_line.fitx[-1]) < min_lane_width) \
          or ((self.right_line.fitx[-1] - self.left_line.fitx[-1]) > max_lane_width): 
            sanity = False
        self.left_line.update_last_fits(sanity)
        self.right_line.update_last_fits(sanity)
        self.compute_data()
        if not sanity:
            self.miss_count += 1
        else: 
            self.miss_count = 0       
        return sanity
    
def weighted_img(img_a, img_b, a=1., b=.6):
    """Adds two images applying a different weight to each one"""
    return cv2.addWeighted(img_a, a, img_b, b, 0.)

class Lane_detection():
    def __init__(self):
        self.objpoints, self.imgpoints = calibrate_cam()
        self.lane = Lane()
        
    def fit_lane_lines(self, binary_warped, draw_windows=False, fill=True):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        #print(nonzero)
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        draw_line_band = False
        if len(self.lane.left_line.last_fits)>0 and len(self.lane.right_line.last_fits)>0 and self.lane.miss_count<10:
            for y in self.lane.left_line.ploty.astype(np.int32):
                good_left_inds = ((nonzeroy == y) & 
                              (nonzerox >= self.lane.left_line.best_fitx[y]-margin) &  
                              (nonzerox < self.lane.left_line.best_fitx[y]+margin)
                             ).nonzero()[0]
                good_right_inds = ((nonzeroy == y) & 
                              (nonzerox >= self.lane.right_line.best_fitx[y]-margin) &  
                              (nonzerox < self.lane.right_line.best_fitx[y]+margin)
                             ).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
            draw_line_band = draw_windows
        else:
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                if draw_windows:
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,255), 2) 
                    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (255,0,255), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & 
                                  (nonzeroy < win_y_high) & 
                                  (nonzerox >= win_xleft_low) &  
                                  (nonzerox < win_xleft_high)
                                 ).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & 
                                   (nonzeroy < win_y_high) & 
                                   (nonzerox >= win_xright_low) &  
                                   (nonzerox < win_xright_high)
                                  ).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]     
        self.lane.set_new_left_points(leftx, lefty)
        self.lane.set_new_right_points(rightx, righty)
        sanity = self.lane.check_sanity()
        ploty = self.lane.left_line.ploty
        if draw_line_band:
            for y in ploty.astype(np.int32) :
                out_img[y, int(self.lane.left_line.best_fitx[y]-margin) : int(self.lane.left_line.best_fitx[y])+margin] = [50, 0, 50]
                out_img[y, int(self.lane.right_line.best_fitx[y]-margin) : int(self.lane.right_line.best_fitx[y])+margin] = [50, 0, 50]
        if fill:
            if sanity:
                fill_color = [0, 80, 0]
            else:
                fill_color = [0, 80, 0] #[50, 0, 0]
            for y in ploty.astype(np.int32) :
                out_img[y, int(self.lane.left_line.best_fitx[y]) : int(self.lane.right_line.best_fitx[y])] = fill_color
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]    
        return out_img, self.lane.left_line.best_fitx, self.lane.right_line.best_fitx, self.lane.left_line.ploty, self.lane.radius, self.lane.off_center

    def pipeline(self, img):
        u_img = undistort_img(img, self.objpoints, self.imgpoints)
        hist_eq = eq_histogram(u_img)
        hls = cv2.cvtColor(hist_eq, cv2.COLOR_RGB2HLS)
        new_img = perspective_transform(u_img)
        binary_img = combine_threshold(hls)
        perspective_img = perspective_transform(binary_img)
        fit_img, _, _, _, radius, off_center = self.fit_lane_lines(perspective_img)
        fit_img = perspective_transform(fit_img, inverse=True)
        out_img = weighted_img(fit_img, u_img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if abs(radius) > 3000:
            radius_text = 'Approximately straight'
        else:
            radius_dir = 'left' if radius < 0 else 'right'
            radius_text = 'Radius of curvature = {0:.2f}m to the {1}'.format(abs(radius), radius_dir)
        center_dir = 'left' if off_center < 0 else 'right'
        center_text = 'Vehicle is {0:.2f}m {1} from the center'.format(abs(off_center), center_dir)
        out_img = cv2.putText(out_img, radius_text, (30,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
        out_img = cv2.putText(out_img, center_text, (30,100), font, 1, (255,255,255), 2, cv2.LINE_AA)
        return out_img