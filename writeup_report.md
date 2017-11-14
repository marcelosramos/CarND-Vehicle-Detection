
## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps, normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog]: ./output_images/hog.png
[windows]: ./output_images/windows.png
[pipeline_final]: ./output_images/pipeline_final.png
[pipeline]: ./output_images/pipeline.png


## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

The present document is the writeup report.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features()` function contained in the code cell #3 of the IPython notebook CarND-Vehicle-Detection.ipynb

I started by reading in all the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=32`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` for one vehicle and one non-vehicle image:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Some didn't perform very well on the project video, some would take too long to extract the features and would make a feature vector too big. Finally, more orientations with not so many cells seemed to be the right choice.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First I extracted the HOG, spatially binned color and color histogram features of all the vehicle and non-vehicle images with the `extract_features()` function, then I normalized the feature vectors using the `sklearn.preprocessing.StandardScaler()` class and radomized and splited the data set with 90% for training and 10% for testing. Finally I used that data to train the linear SVM classifier (`LinearSVC`).

All this process can be found in the code cell #4, and the results are shown below:

|SVM training results                                           |
|:-------------------------------------------------------------:|
|105.44 Seconds to extract HOG features                         |
|Using: 32 orientations 16 pixels per cell and 2 cells per block|
|Feature vector length: 6624                                    |
|20.31 Seconds to train SVC                                     |
|Test Accuracy of SVC =  0.9916                                 |
|My SVC predicts:  [ 1.  1.  0.  1.  1.  1.  1.  0.  0.  0.]    |
|For these 10 labels:  [ 1.  1.  0.  1.  1.  1.  1.  0.  0.  0.]|
|0.00376 Seconds to predict 10 labels with SVC                  |


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search implementation can be found in the code cell #6. I used an overlap of 80%, 96x96 windows for 380 <= y <= 600,  128x128 windows for 600 <= y <= 720 and, assuming it is known that the car is already on the most left lane, only searched for x values between 600 and 1280. The image below shows those windows (96x96 in blue and 128x128 in red):

![alt text][windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For the final pipeline, the sliding windows search actually happens inside the `find_cars()` function in code cell #8. The reason I did it this way is because, in order to improve performance, I extract the HOG features all at once on the region of interest (ROI) and then slide the window over it to run the classifier with the same parameter used on the training step.

To optimize the performance, I also used 2 different scales, one with smaller windows for the region closer to the horizon, where the cars appear smaller, and one for the region closer to the point of view, where the cars appear bigger. This approach decreased the number of windows without compromising the accuracy of the vehicle detection.

Here are some example images:

![alt text][pipeline]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/S-X4RgHT7rg)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and discard the false positives.

To smooth the detections I kept a sum of the heatmaps from the last 10 frames. (code cell #12)

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. (code cell #10)

Here are the example images with the positive detections, their corresponding heat maps and their resulting bounding boxes:

![alt text][pipeline_final]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Just like in the Advanced Lane Line project, I am running out of time and could not dedicate my self to the project as much as I would like to.

This implementation seems over fitted to this specific video and, although I was able to process about 4 frames per second before combining with the lane detection, in my opinion, it is still very slow for real life aplications.

Keeping track of the final bounding boxes and considering it positive only after being detected in a few consecutive frames, instead of just acumulating the thresholded heatmaps for ten frames, and using some computer vision tecniques such as template matching for keeping track of a car that was already found would surely make it much more robust.

