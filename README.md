# Real-time-Object-2D-Recognition

## Summary
This project is about 2D object recognition. The program identifies a specified set of objects like a cutter, action figure, bracelet placed on a white surface in a translation, scale, and rotation-invariant manner from a camera looking straight down. It recognizes objects and identifies them using classification algorithms like K-Nearest Neighbors. It works in real-time video as well as on still images.

## Setup
The object is placed on a white background (a curtain) for efficient thresholding. There are two light sources to produce uniform lighting on the object as well as the background.

## Thresholding
To extract the foreground object from the white background, one-sided thresholding is implemented. Before that, the image is smoothened using a bilateral filter. The bilateral filter preserves the edges while denoising the image, unlike the Gaussian filter.

## Clean-Up
To solve the issue of holes in the foreground image, the dilation function from OpenCV is implemented. However, as the foreground object seemed to lose its original shape, the erosion morphological function from OpenCV is used to reduce it back to its original size.

## Segmented Regions
A connected component analysis is run on the image to detect each region. Regions smaller than an area of 1000 pixels are filtered out. Each region is displayed in a different color to distinguish them.

## Compute Features
Using the obtained region, a set of moments is computed using OpenCV. Moments allow computing the axis of least central moment and the oriented bounding box. The Hu moments, which are translational, rotational, and scale-invariant, are also computed and used as features, with one of them displayed on the output frame.

## Training Data
While the video is on, the user can press the 'T' key to capture a snapshot of the frame. The program will calculate the features and ask the user to label the image. All the information is then stored in a CSV file for later retrieval.

## Classify Objects
Each frame of the real-time video is captured and given as input to the classifier. Two classifiers are used for this task: a baseline classifier using a simple calculation of the L2 norm as the distance metric and retrieving the label with the least distance from the image, and a classifier using K-Nearest Neighbors that returns a label which is the mode of the K nearest neighbors.

## Evaluation
A confusion matrix is used to evaluate the model, and the accuracy is computed to be 55%.

## Demonstration
A demonstration video is available at https://drive.google.com/file/d/1g2t350y7k_MAqf5ai83yfo8BkJOKoXAh/view?usp=sharing

## Extensions

### Recognition of Multiple Objects Simultaneously
This extension recognizes multiple objects simultaneously using connected component analysis. The user inputs N, the maximum number of regions the program will find. It finds the largest N regions in the image and assigns a different color to each region.

### Recognition of Unknown Objects
This extension recognizes if an object belongs to the database. While using the K-Nearest Neighbor classifier, if the distance is above a certain threshold, the object is classified as an unknown object.

## Key Takeaways
- Learned how classical computer vision can be used to classify objects in still images or real-time.
- Gained understanding of how moments can be essential for object detection due to their invariant properties.
- Learned about the K-Nearest Neighbor algorithm.
- Deepened understanding of core concepts in computer vision and C++.

## Acknowledgment
The following websites were referred to for learning about 2D object detection:

- https://docs.opencv.org/3.4/index.html
- https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
- https://learnopencv.com/tag/confusionmatrix/
- https://www.cplusplus.com/reference/cmath/cos/
- http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Hu.pdf
- https://thispointer.com/how-to-write-data-in-a-csv-file-in-c/

The project contains the following files:
Real-Time-Object-2D-Recognition.cpp : This is the driver code
csv_helper.cpp: Used to store and read from database
knn.cpp: Implements K-Nearest Neighbour algorithm to classify objects

csv_helper.h: Header file for reading and writing CSV
knn.h: Header file for KNN classifier 
