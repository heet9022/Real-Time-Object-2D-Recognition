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
#include <set>

#include "csv_helper.h"
#include "knn.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class Object {
public:
    int area = 0;
    int label = 0;
    int x = 0;
    int y = 0;
    int height = 0;
    int width = 0;
    cv::Point centroid;
    Vec3b color = Vec3b(0,0,0);
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

vector<vector<double>> computeFeatures(Mat &src, map<int, Object> regions) {

    Mat dst = src.clone();
    vector<Moments> mu(regions.size());
    vector<Point2f> mc(regions.size());
    vector<vector<double>> arrayOfhuMoments(regions.size());
    map<string, int>::iterator it;

    int k = 0;
    Scalar green(0, 255, 0);
    Scalar blue(255, 0, 0);

    for (auto const& region : regions)
    {
        if (region.second.area == 0)
            continue;

        vector<Point2f> pix_;
        for (Point p : region.second.pixels) {
            pix_.push_back(Point2f(p.y, p.x));
        }

        RotatedRect box = minAreaRect(cv::Mat(pix_));
        cv::Point2f vertices[4];
        box.points(vertices);
        //cv::Point2f vertices_[4];
        //for (Point2f v : vertices) {
        //    vertices_->x = vertices->y;
        //    vertices_->y = vertices->x;
        //}

        for (int j = 0; j < 4; ++j)
            cv::line(dst, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 1, 8);

        Point p1, p2, p3, p4, k1, k2;
        p1 = (vertices[0] + vertices[1]) / 2;
        p2 = (vertices[2] + vertices[3]) / 2;

        p3 = (vertices[1] + vertices[2]) / 2;
        p4 = (vertices[3] + vertices[0]) / 2;

        double d1 = cv::norm(p1 - p2);
        double d2 = cv::norm(p3 - p4);

        if (d1 > d2) {
            k1 = p1;
            k2 = p2;
        }
        else {
            k1 = p3;
            k2 = p4;
        }

        //Rect rect(region.second.x, region.second.y, region.second.width, region.second.height);
        //cv::rectangle(dst, rect, green);
        Moments moment = moments(pix_);
        mu.push_back(moment);
        //mc.push_back(Point2f(static_cast<float>(moment.m10 / (moment.m00 + 1e-5)), static_cast<float>(moment.m01 / (moment.m00 + 1e-5))));
        //double alpha = 0.5 * atan2(2 * moment.mu11, moment.mu20 - moment.mu02);
        //alpha = (alpha / 3.142) * 180;
        
        //Point rot_projection = Point();
        //float min_x = std::numeric_limits<float>::infinity();
        //float max_x = -std::numeric_limits<float>::infinity();
        //float min_y = std::numeric_limits<float>::infinity();
        //float max_y = -std::numeric_limits<float>::infinity();
        //for (Point coord : region.second.pixels) {

        //    float y_proj = (float) ((coord.x - (moment.m10 / moment.m00)) * cos(alpha) + (coord.y - (moment.m01 / moment.m00)) * sin(alpha));
        //    float x_proj = (float) ((coord.x - (moment.m10 / moment.m00)) * ( - sin(alpha)) + (coord.y - (moment.m01 / moment.m00)) * cos(alpha));

        //    min_x = min_x < x_proj ? min_x : x_proj;
        //    max_x = max_x > x_proj ? max_x : x_proj;

        //    min_y = min_y < y_proj ? min_y : y_proj;
        //    max_y = max_y > y_proj ? max_y : y_proj;
        //}
        //Point2f cen = Point(moment.m10 / moment.m00, moment.m01 / moment.m00);
        line(dst, k1, k2, Scalar(0,255,0), 2);

        //RotatedRect box(Point2f(dst.rows-1-max_y, max_x), Point2f(dst.rows - 1 - min_y, max_x), Point2f(dst.rows - 1 - min_y, min_x));
        //Point2f vertices[4];
        //box.points(vertices);
        //for (int i = 0; i < 4; i++)
        //    line(dst, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);


        HuMoments(moment, arrayOfhuMoments[k]);
        for (int j = 0; j < 7; j++) {
            arrayOfhuMoments[k][j] = -1 * copysign(1.0, arrayOfhuMoments[k][j]) * log10(abs(arrayOfhuMoments[k][j]));
        }
        cv::putText(dst, //target image
            to_string(arrayOfhuMoments[k][1]), //text
            region.second.centroid, //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            CV_RGB(118, 185, 0), //font color
            2);
        k++;
    }
    imshow("Features", dst);

    return arrayOfhuMoments;
}

Mat getConnectedComponents(const Mat &src, const Mat &target, map<int, Object> &regions, int N) {

        Mat labels, stats, centroids;
        Mat dst = Mat::zeros(target.rows, target.cols, CV_8UC3);
        int comps = connectedComponentsWithStats(src, labels, stats, centroids, 4);

        //cout << "Number of components: " << comps << endl;
        vector<Object> objects;

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

            objects[i].color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            objects[i].x = stats.at<int>(objects[i].label, cv::CC_STAT_LEFT);
            objects[i].y = stats.at<int>(objects[i].label, cv::CC_STAT_TOP);
            objects[i].width = stats.at<int>(objects[i].label, cv::CC_STAT_WIDTH);
            objects[i].height = stats.at<int>(objects[i].label, cv::CC_STAT_HEIGHT);
            objects[i].centroid = Point(centroids.at<double>(objects[i].label, 0), centroids.at<double>(objects[i].label, 1));
            regions[objects[i].label] = objects[i];
        }

        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {

                int label = labels.at<int>(i, j);
                regions[label].pixels.push_back(Point(i, j));
                dst.at<Vec3b>(i, j) = regions[label].color;
            }
        }
        return dst;
}

vector<vector<double>> pipeline(Mat &src, int N) {

    Mat thresh = getThreshold(src);
    Mat morphed = getMorphed(thresh);
    Mat stats, centroids;
    map<int, Object> objects;
    Mat segmented = getConnectedComponents(morphed, src, objects, N);
    //imshow("Segmented", segmented);
    vector<vector<double>>  features = computeFeatures(src, objects);
    //imshow("CC", segmented);
    //waitKey(0);
    return features;

}

double euclideanDistance(vector<double> f1, vector<double> f2) {

    double sum = 0;
    for (int i = 0; i < f1.size(); i++)
        sum += pow(f1[i] - f2[i], 2);

    return sqrt(sum);
}

string classify(const vector<vector<double>>  &features) {

    string fileName = "training_feature_database.csv";
    std::vector<std::string> labels;
    std::vector<std::vector<double>> nfeatures;
    readFromFile(fileName, labels, nfeatures);

    double min_dist = std::numeric_limits<double>::infinity();
    string min_label;
    for (int i = 0; i < nfeatures.size(); i++) {
        
        double dist = euclideanDistance(nfeatures[i], features[0]);
        if (dist < min_dist) {
            min_dist = dist;
            min_label = labels[i];
        }
    }
    return min_label;
}

void video() {

    VideoCapture* capdev;
    capdev = new cv::VideoCapture(0);
    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    vector<vector<double>>  features;
    string p_class;

    bool record = false;
    bool playing = false;
    VideoWriter oVideoWriter;

    for (;;) {

        p_class = "No objects Detected";
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        features = pipeline(frame, 1);
        if (features[0].size() != 0)
             p_class = KNN(features, 5);
        cv::putText(frame, //target image
            p_class, //text
            Point(100,100), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1,
            CV_RGB(118, 185, 0), //font color
            2);

        if (record) {
            int frames_per_second;
            Size frame_size;
            if (!playing) {
                frames_per_second = 5;
                frame_size = frame.size();
                oVideoWriter.open("ObjectDetection.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), frames_per_second, frame_size, true);
                playing = true;
                cout << "Recording started" << endl;
            }
            if (playing)

                oVideoWriter.write(frame);
        }
        else
            oVideoWriter.release();

        imshow("Video", frame);
        char key = cv::waitKey(10);
        string name;

        if (key == 'q') 
            break;

        if (key == 's') {

            cout << "Name the image: ";
            getline(cin, name);
            imwrite("Results\\" + name + ".jpg", frame);
            cout << name << " saved!" << endl;
        }

        if (key == 't') {

            cout << "Name the training image: ";
            getline(cin, name);
            imwrite("Train_images\\" + name + ".jpg", frame);
            cout << name << " saved!" << endl;

            String label;
            cout << "Label the object: ";
            getline(cin, label);

            string fileName = "training_feature_database.csv";

            if (writeToFile(fileName, label, features[0]))
                cout << "a Successful write " << endl;
        }

        if (key == 'r') {

            record = !record;
            playing = false;

        }
    }

    delete capdev;

}

void evaluate() {

    fs::path imgDir = fs::current_path();
    imgDir /= "Test_images";
    map<string, map<string, int>> d;
    set<string> s;
    float total = 0;
   
    for (const auto& entry : fs::directory_iterator(imgDir)) {

        fs::path path = entry.path();
        Mat image = imread(path.string(), IMREAD_COLOR);

        vector<vector<double>>  features = pipeline(image, 1);
        string pred_class = KNN(features, 5);

        string fname = path.filename().string();
        string real_label = fname.substr(0, fname.find("_"));
        s.insert(real_label);
        //cout << pred_class << ":" << real_label << endl;
        d[pred_class][real_label] += 1;
        total++;
    }
    float correct_preds = 0;
    for (string label : s) {
        correct_preds += d[label][label];
    }
    cout <<"accuracy: " << correct_preds / total * 100 << endl;
    string fname = "confusion_matrix.csv";
    //writeConfusionMatrix(fname, s, d);
}

int main() {

    //Mat target = getImage("sample_1.jpg");
    //Mat processed = pipeline(target, 3);
    //imshow("Original", target);
    //imshow("Processed", processed);
    //waitKey(0);

    video();
    //evaluate();
 
    return(0);

}
