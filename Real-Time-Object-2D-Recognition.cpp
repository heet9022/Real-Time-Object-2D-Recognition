// Real-Time-Object-2D-Recognition.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <iostream>
#include <filesystem>
#include<cmath>
#include <map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class Object {
public:
    int area = 0;
    int label = 0;
    Vec3b color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    vector<cv::Point> pixels;
};

Mat getImage(string image) {

    fs::path target_path = fs::current_path();
    target_path = (target_path / "Data/Examples") / image;
    Mat target = imread(target_path.string(), IMREAD_COLOR);
    return target;
}

Mat getThreshold(const Mat &src) {

    Mat blur;
    cv::bilateralFilter(src, blur, 20, 20*2, 20 / 2);
    Mat thresh(blur.size(), CV_8U);
    for (int i = 0; i < blur.rows; i++) {
        for (int j = 0; j < blur.cols; j++) {

            Vec3b intensity = blur.at<Vec3b>(i, j);
            if (intensity[0] > 50 && intensity[1] > 50 && intensity[2] > 50) {
                thresh.at<uchar>(i, j) = 0;
            }
            else {
                thresh.at<uchar>(i, j) = 255;
            }
        }
    }
    return thresh;
}

Mat getEroded(const Mat &src) {

    int erosion_type = MORPH_RECT;
    int erosion_size = 1;
    Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    Mat eroded;
    erode(src, eroded, element);
    return eroded;
}

Mat getDilated(const Mat &src) {

    int dilation_type = MORPH_RECT;
    int dilation_size = 1;
    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    Mat dilated;
    dilate(src, dilated, element);
    return dilated;
}

Mat getMorphed(const Mat& src) {

    Mat dilated = getDilated(src);
    Mat eroded = getEroded(dilated);
    
    return eroded;
}

bool cmp(Object &a, Object &b) {
    return a.area > b.area;
}

void computeFeatures(Mat stats, Mat centroids, vector<Object> regions) {

    vector<double> features;

    for (Object region : regions) {

    }



}

Mat getConnectedComponents(const Mat &src, const Mat &target, Mat &stats, Mat &centroids, vector<Object> &regions, int N) {

        cv::Mat labels;
        Mat dst = Mat::zeros(target.rows, target.cols, CV_8UC3);
        int comps = connectedComponentsWithStats(src, labels, stats, centroids, 4);

        //cout << "Number of components: " << comps << endl;
        vector<Object> objects;
        vector<Vec3b> colors(comps, Vec3b(0,0,0));

        for (int i = 1; i < comps; i++) {

            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < 1000)
                continue;
            Object object;
            object.area = area;
            object.label = i;
            objects.push_back(object);
        }

        sort(objects.begin(), objects.end(), cmp);

        int min = N < objects.size() ? N : objects.size();

        for (int i = 0; i < min; i++) {  
            regions.push_back(objects[i]);
            colors[objects[i].label] = objects[i].color;
        }

        //for (int i = 0; i < min; i++) {
        //    int x = stats.at<int>(objects[i].label, cv::CC_STAT_LEFT);
        //    int y = stats.at<int>(objects[i].label, cv::CC_STAT_TOP);
        //    int w = stats.at<int>(objects[i].label, cv::CC_STAT_WIDTH);
        //    int h = stats.at<int>(objects[i].label, cv::CC_STAT_HEIGHT);
        //    int area = stats.at<int>(objects[i].label, cv::CC_STAT_AREA);

        //    Scalar green(0, 255, 0);
        //    Scalar blue(255, 0, 0);
        //    Rect rect(x, y, w, h);
        //    cv::circle(dst, Point(centroids.at<double>(objects[i].label, 0), centroids.at<double>(objects[i].label, 1)), 3, blue);
        //    cv::rectangle(dst, rect, green);
        //}

        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {

                int label = labels.at<int>(i, j);
                dst.at<Vec3b>(i, j) = colors[label];
            }
        }
        return dst;
}

Mat pipeline(Mat &src, int N) {

    Mat thresh = getThreshold(src);
    Mat morphed = getMorphed(thresh);
    Mat stats, centroids;
    vector<Object> objects;
    Mat segmented = getConnectedComponents(morphed, src, stats, centroids, objects, N);
    //imshow("CC", segmented);
    //waitKey(0);
    computeFeatures(stats, centroids, objects);

    return segmented;

}

void video() {

    VideoCapture* capdev;
    capdev = new cv::VideoCapture(0);
    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;

    for (;;) {

        *capdev >> frame; // get a new frame from the camera, treat as a stream
        frame = pipeline(frame, 3);
        imshow("Video", frame);
        char key = cv::waitKey(10);
        string name;

        if (key == 'q') 
            break;

        if (key == 's') {

            cout << "Name the image: ";
            getline(cin, name);
            imwrite(name + ".jpg", frame);
            cout << name << " saved!" << endl;
        }
    }

    delete capdev;

}

int main() {

    //Mat target = getImage("sample_1.jpg");
    //Mat processed = pipeline(target, 3);
    //imshow("Original", target);
    //imshow("Processed", processed);
    //waitKey(0);

    video();
 
    return(0);

}
