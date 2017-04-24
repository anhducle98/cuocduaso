#ifndef XLA_H
#define XLA_H

#include <bits/stdc++.h>
#include <opencv/cv.hpp>

#include "./lib/api_lane_detection.h"
#include "./lib/IPM.h"
#include "./lib/line_segment.h"

using namespace std;
using namespace cv;

const double ANGLE_THRESHOLD = PI / 8;
const double LENGTH_THRESHOLD = 50;
const double DIST_THRESHOLD = 800;

class TopviewConverter {
public:
    IPM ipm;

    TopviewConverter() {}

    TopviewConverter(vector<Point2f> &origPoints, int width = 640, int height = 480) {
        vector<Point2f> dstPoints;
        dstPoints.push_back(Point2f(0, height));
        dstPoints.push_back(Point2f(width, height));
        dstPoints.push_back(Point2f(width, 0));
        dstPoints.push_back(Point2f(0, 0));
        ipm = IPM(Size(width, height), Size(width, height), origPoints, dstPoints);
    }

    void convert(const Mat &inputImg, Mat &outputImg) {
        ipm.applyHomography(inputImg, outputImg);
    }
};

class XLA {
private:
    static const int LOCAL_FRAME_WIDTH = 640;
    static const int LOCAL_FRAME_HEIGHT = 480;

    template<typename T> T sqr(T x) {
        return x * x;
    }
    template<typename T> int sign(T x) {
        if (x == 0) return 0;
        return x < 0 ? -1 : 1;
    }

    TopviewConverter bird;

public:
    XLA() {}
    XLA(string file_path) { read_topview_params(file_path); }
    void run_hough_transform(const Mat &binary_frame, vector<Vec4i> &lines);
    void show_angle(Mat &output, double rad);
    void draw_lines(Mat&output, vector<Line> &lines);
    double get_steering_angle(vector<Line> &lines);
    Line process_frame(Mat &bgr_frame, VideoWriter &bgr_writer);
    double adjust_angle(Line previous_line, Line current_line);
    void convert_to_topview(const Mat &inputImg, Mat &outputImg);
    void read_topview_params(string file_path);
    vector<Line> process_one_side(Mat &binary_frame);
    double get_sum_length(vector<Line> &lines);
};


#endif //XLA_H
