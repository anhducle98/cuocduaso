#include "xla.h"
#define SHOW_OUTPUT
#define WRITE_VIDEO

void XLA::run_hough_transform(const Mat &binary_frame, vector<Vec4i> &lines) {
    int houghThreshold = 100;
    HoughLinesP(binary_frame, lines, 2, CV_PI/90, houghThreshold, 10,30);
}

void XLA::show_angle(Mat &output, double rad) {
    Point bottom(output.cols / 2, output.rows);
    Point to(-cos(rad) * 100 + output.cols / 2, output.rows - sin(rad) * 100);
    arrowedLine(output, bottom, to, Scalar(50, 50, 255), 3, 8);
}

void XLA::draw_lines(Mat&output, vector<Line> &lines) {
    printf("Hough found %d lines\n", (int)lines.size());
    for(size_t i = 0; i < lines.size(); i++) {
        line(output, lines[i].P, lines[i].Q, Scalar(0, 0, 255), 3, 8);
    }
}

double XLA::get_steering_angle(vector<Line> &lines) { //radian
    double sum_angle = 0;
    double sum_weight = 0;

    for(size_t i = 0; i < lines.size(); i++) {
        double dist = lines[i].length();
        double angle = lines[i].angle();
        sum_weight += dist;
        sum_angle += dist * angle;
    }

    sum_angle /= sum_weight;
    return sum_angle;
}

double XLA::get_sum_length(vector<Line> &lines) {
    double res = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        res += lines[i].length();
    }
    return res;
}

vector<Line> XLA::process_one_side(Mat &binary_frame) {
    vector<Vec4i> lines;
    run_hough_transform(binary_frame, lines);
    vector<Line> d;
    for (int i = 0; i < lines.size(); ++i) {
        d.push_back(Line(Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3])));
    }
    return d;
}

Line XLA::process_frame(Mat &bgr_frame, VideoWriter &bgr_writer) {
    int width = bgr_frame.cols;
    int height = bgr_frame.rows;

    static Mat gray_frame;
    static Mat topview_frame;
    static Mat binary_frame;

    if (bgr_frame.channels() == 3) cvtColor(bgr_frame, gray_frame, CV_BGR2GRAY);
    convert_to_topview(gray_frame, topview_frame);
    if (LOCAL_FRAME_WIDTH != width || LOCAL_FRAME_HEIGHT != height) {
        resize(topview_frame, topview_frame, Size(LOCAL_FRAME_WIDTH, LOCAL_FRAME_HEIGHT));
        width = LOCAL_FRAME_WIDTH;
        height = LOCAL_FRAME_WIDTH;
    }

    static Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    edgeProcessing(topview_frame, binary_frame, element, "Wavelet");

    Rect left_ROI(0, 0, width / 2, height);
    Rect right_ROI(width / 2, 0, width / 2, height);
    Mat left_image = binary_frame(left_ROI);
    Mat right_image = binary_frame(right_ROI);
    
    vector<Line> left_lane = process_one_side(left_image);
    vector<Line> right_lane = process_one_side(right_image);
    for (int i = 0; i < right_lane.size(); ++i) {
        right_lane[i].shift(width / 2);
    }

    Line res;

    if (get_sum_length(left_lane) > get_sum_length(right_lane)) {
        res = approximate(left_lane, get_steering_angle(left_lane));
        res.is_left = true;
    } else {
        res = approximate(right_lane, get_steering_angle(right_lane));
        res.is_left = false;
    }

    //lines = filter_lines(lines);
    Mat hough_frame = topview_frame.clone();
    
#ifdef WRITE_VIDEO
    draw_lines(hough_frame, left_lane);
    draw_lines(hough_frame, right_lane);
    Point A = Point(res.intersect(0), 0);
    Point B = Point(res.intersect(480), 480);
    line(hough_frame, A, B, Scalar(0, 255, 0), 3, 8);

    show_angle(hough_frame, res.angle());
#endif
    //convert_back(hough_frame, hough_frame);

#ifdef SHOW_OUTPUT
    imshow("bgr", bgr_frame);
    imshow("topview", topview_frame);
    imshow("binary", binary_frame);
    imshow("hough", hough_frame);
    waitKey(30);
#endif

#ifdef WRITE_VIDEO
    bgr_writer << hough_frame;
#endif

    return res;
}

double XLA::adjust_angle(Line previous_line, Line current_line) {
    static const int DISTANCE_THRESHOLD = LOCAL_FRAME_WIDTH / 10;
    double current_angle = current_line.angle();
    double difference = current_line.intersect(LOCAL_FRAME_HEIGHT / 2) - previous_line.intersect(LOCAL_FRAME_HEIGHT / 2);

    double multiplier = 2.5;
    double minor = 0.5;

    if (previous_line.is_left == current_line.is_left && abs(difference) >= DISTANCE_THRESHOLD) {
        cerr << "adjusted\n";
        multiplier += minor;
    }
    return current_angle * multiplier * (-1);
}

void XLA::convert_to_topview(const Mat &inputImg, Mat &outputImg) {
    bird.convert(inputImg, outputImg);
}

void XLA::read_topview_params(string file_path){
    ifstream finp(file_path.c_str());
    vector<Point2f> orig;
    for (int i = 0; i < 4; ++i) {
        int x, y;
        finp >> x >> y;
        orig.push_back(Point2f(x, y));
    }
    this->bird = TopviewConverter(orig);
}
