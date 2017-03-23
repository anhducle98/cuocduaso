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

double process_frame(Mat &bgr_frame, VideoWriter &bgr_writer);
double get_steering_angle(vector<Vec4i> &lines);
void draw_lines(Mat&output, vector<Vec4i> &lines);
void convert_to_topview(Mat inputImg, Mat &outputImg);

#endif