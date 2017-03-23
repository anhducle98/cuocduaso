#include "xla.h"
//#define SHOW_OUTPUT
//#define WRITE_VIDEO

void convert_to_topview(Mat inputImg, Mat &outputImg) {
    int height = inputImg.rows;
    int width = inputImg.cols;

    vector<Point2f> origPoints;
    origPoints.push_back(Point2f(0, height));
    origPoints.push_back(Point2f(width, height));
    origPoints.push_back(Point2f(width/2+100, 300));
    origPoints.push_back(Point2f(width/2-120, 300));

    vector<Point2f> dstPoints;
    dstPoints.push_back(Point2f(0, height));
    dstPoints.push_back(Point2f(width, height));
    dstPoints.push_back(Point2f(width, 0));
    dstPoints.push_back(Point2f(0, 0));

    IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);
    ipm.applyHomography(inputImg, outputImg);
}

void convert_back(Mat inputImg, Mat &outputImg) {
    int height = inputImg.rows;
    int width = inputImg.cols;

    vector<Point2f> origPoints;
    origPoints.push_back(Point2f(0, height));
    origPoints.push_back(Point2f(width, height));
    origPoints.push_back(Point2f(width/2+100, 300));
    origPoints.push_back(Point2f(width/2-120, 300));

    vector<Point2f> dstPoints;
    dstPoints.push_back(Point2f(0, height));
    dstPoints.push_back(Point2f(width, height));
    dstPoints.push_back(Point2f(width, 0));
    dstPoints.push_back(Point2f(0, 0));

    IPM ipm(Size(width, height), Size(width, height), dstPoints, origPoints);
    ipm.applyHomography(inputImg, outputImg);
}

void run_hough_transform(const Mat &binary_frame, vector<Vec4i> &lines) {
    int houghThreshold = 100;
    HoughLinesP(binary_frame, lines, 2, CV_PI/90, houghThreshold, 10,30);
}

#define sqr(x) ((x)*(x))

void show_angle(Mat &output, double deg) {
    double rad = deg * PI / 180;
    //cerr << "Theta = " << deg - 90 << endl;
    Point bottom(output.cols / 2, output.rows);
    Point to(-cos(rad) * 100 + output.cols / 2, output.rows - sin(rad) * 100);
    arrowedLine(output, bottom, to, Scalar(50, 50, 255), 3, 8);
}

void draw_lines(Mat&output, vector<Vec4i> &lines) {
    printf("Hough found %d lines\n", (int)lines.size());
    for(size_t i = 0; i < lines.size(); i++) {
        line(output, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
    }
}

double get_steering_angle(vector<Vec4i> &lines) {
    double sum_angle = 0;
    double sum_weight = 0;

    for(size_t i = 0; i < lines.size(); i++) {
        double dist = sqrt(sqr(lines[i][0] - lines[i][2]) + sqr(lines[i][1] - lines[i][3]));
        double angle = atan2(lines[i][3] - lines[i][1], lines[i][2] - lines[i][0]);
        if (angle < 0) {
            angle = PI + angle;
        }
        double weight = dist * max(lines[i][1], lines[i][3]);
        sum_weight += weight;
        sum_angle += weight * angle;
    }

    sum_angle /= sum_weight;
    return sum_angle * 180 / PI;
}
/*

bool mergeable(Line d1, Line d2) {
    Point M = intersection(d1, d2);
    if (max(d1.dist_to_point(M), d2.dist_to_point(M)) > DIST_THRESHOLD) return false;
    if (angle_between_two_lines(d1, d2) > ANGLE_THRESHOLD) return false;
    return true;
}

struct DSU {
    vector<int> par;
    vector<double> weight;
    DSU(int n) {
        par.assign(n, -1);
        weight.assign(n, 0);
    }

    int root(int u) {
        if (par[u] < 0) return u;
        return root(par[u]);
    }

    bool merge(int u, int v) {
        u = root(u); v = root(v);
        if (u == v) return false;
        par[u] = v;
        weight[u] += weight[v];
        return true;
    }

    double get_weight(int u) {
        return weight[root(u)];
    }
};

vector<Vec4i> filter_lines(vector<Vec4i> &lines) {
    vector<Line> a;
    for (auto it : lines) {
        a.push_back(Line(Point(it[0], it[1]), Point(it[2], it[3])));
    }
    DSU dsu(a.size());
    for (int i = 0; i < a.size(); ++i) dsu.weight[i] = a[i].length();
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (mergeable(a[i], a[j])) {
                dsu.merge(i, j);
            }
        }
    }

    vector<Vec4i> res;
    double max_weight = 0;
    for (int i = 0; i < a.size(); ++i) max_weight = max(max_weight, dsu.get_weight(i));
    cerr << "max_weight " << max_weight << endl;
    if (max_weight < LENGTH_THRESHOLD) {
        return lines;
    }
    for (int i = 0; i < a.size(); ++i) if (dsu.get_weight(i) > LENGTH_THRESHOLD) {
        //cerr << dsu.get_weight(i) << endl;
        res.push_back(Vec4i(a[i].P.x, a[i].P.y, a[i].Q.x, a[i].Q.y));
    }
    return res;
}
*/

double process_frame(Mat &bgr_frame, VideoWriter &bgr_writer) { //returns theta
    static Mat gray_frame;
    static Mat topview_frame;
    static Mat binary_frame;

    convert_to_topview(bgr_frame, topview_frame);
    cvtColor(topview_frame, gray_frame, CV_BGR2GRAY);
    
    static Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    edgeProcessing(gray_frame, binary_frame, element, "Wavelet");

    
    vector<Vec4i> lines;
    run_hough_transform(binary_frame, lines);
    //lines = filter_lines(lines);
    Mat hough_frame = topview_frame.clone();
    double steering_angle = get_steering_angle(lines);
#ifdef WRITE_VIDEO
    draw_lines(hough_frame, lines);
    show_angle(hough_frame, steering_angle);
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

    return steering_angle - 90;
}
