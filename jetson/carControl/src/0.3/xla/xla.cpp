#include "xla.h"
#include "lib/line_detect_topview.h"
#include "lib/object_detect_topview.h"
#include "lib/line_object.h"
#include "lib/api_lane_detection.h"
#define SHOW_OUTPUT
#define WRITE_VIDEO
//#define MODE_COLOR

RNG rng(12345);

void XLA::run_hough_transform(const Mat &binary_frame, vector<Vec4i> &lines) {
    int houghThreshold = 50;
    HoughLinesP(binary_frame, lines, 2, CV_PI/90, houghThreshold, 7,20);
}

void XLA::show_angle(Mat &output, double rad) {
    Point bottom(output.cols / 2, output.rows);
    Point to(-cos(rad) * 100 + output.cols / 2, output.rows - sin(rad) * 100);
    arrowedLine(output, bottom, to, Scalar(50, 50, 255), 2, 8);
}

void XLA::draw_lines(Mat&output, vector<Line> &lines) {
    printf("Hough found %d lines\n", (int)lines.size());
    for(size_t i = 0; i < lines.size(); i++) {
        line(output, lines[i].P, lines[i].Q, Scalar(0, 0, 255), 2, 8);
    }
}

double XLA::get_steering_angle(vector<Line> &lines) { //radian
    double sum_angle = 0;
    double sum_weight = 0;

    for(size_t i = 0; i < lines.size(); i++) {
        double dist = lines[i].length();
        double angle = lines[i].angle();
        double mid = (lines[i].P.y + lines[i].Q.y) / 2;
        double weight = dist * mid * mid;
        sum_weight += weight;
        sum_angle += weight * angle;
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
    d = filter_lines(d);
    return d;
}

vector<Line> XLA::get_all_lines(Mat &binary_frame) {
    vector<Vec4i> lines;
    run_hough_transform(binary_frame, lines);
    vector<Line> d;
    for (int i = 0; i < lines.size(); ++i) {
        Point A(lines[i][0],lines[i][1]);
        Point B(lines[i][2],lines[i][3]);
        Line Q(A,B);
        // if(A.x >= binary_frame.cols/2){

        // }
        if(Q.angle() <= PI / 6 || Q.angle() >= PI - PI / 6) {
            continue;
        }
        d.push_back(Line(Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3])));
    }
    return d;
}

int XLA::get_average_intensity(Mat &gray_frame, vector<Point> &contour) {
    Rect box = boundingRect(contour);
    vector<int> a;
    for (int x = box.x; x < box.x + box.width; ++x) {
        for (int y = box.y; y < box.y + box.height; ++y) {
            if (pointPolygonTest(contour, Point2f(x, y), false) > 0) {
//                sum += gray_frame.at<uchar>(y, x);
                a.push_back(gray_frame.at<uchar>(y, x));
            }
        }
    }
//    if (count == 0) return 0;
    if (a.empty()) return 0;
    sort(a.begin(), a.end(), greater<int>());
    int last = max((int)a.size() / 2, 1);
    int sum = 0;
    for (int i = 0; i < last; ++i) {
        sum += a[i];
    }
    return sum / last;
}

Rect XLA::get_bounding_box(Mat &object_binary) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(object_binary, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // cerr << "found " << contours.size() << " contours" << endl;

    int best_id = 0;
    double max_area = 0;
    for (int i = 0; i < contours.size(); ++i) {
         double area = contourArea(contours[i]);
         if (area > max_area) {
            max_area = area;
            best_id = i;
         }
    }
    if (contours.empty() || max_area < 30 * 30) return Rect(0, 0, 0, 0);
    return boundingRect(contours[best_id]);
}

void XLA::lets_be_handsome(Mat &bgr_frame, Mat &gray_frame, Mat &topview_frame, Mat &binary_frame, Rect &obj_box) {
    topview_frame = Topview_transform(bgr_frame);
    //topview_frame = topview_frame(Rect(0, topview_frame.rows*4/8, topview_frame.cols, topview_frame.rows*4/8));
    //resize(topview_frame, topview_frame, Size(topview_frame.cols / 2, topview_frame.rows / 2), INTER_LANCZOS4);
    cvtColor(topview_frame, gray_frame, CV_BGR2GRAY);
    gray_frame = Detect_color(topview_frame);
    static Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    edgeProcessing(gray_frame, binary_frame, element, "Wavelet");
// LINE OBJECT
/*
    Mat object_binary, wavelet;
    get_line_and_object(topview_frame, object_binary, binary_frame);
    obj_box = get_bounding_box(object_binary);
    waveletTransform(binary_frame, wavelet, 0.15);
    binary_frame = wavelet;
*/
    int morph_elem = 1;
    int morph_size = 10;
    Mat mor = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( binary_frame, binary_frame, 3, mor );

//



    // filter_binary_image(binary_frame, binary_frame);
#ifdef SHOW_OUTPUT
    //imshow("object_binary", object_binary);
    //imshow("binary_frame_before", binary_frame);
#endif
/*
    if (LOCAL_FRAME_WIDTH != width || LOCAL_FRAME_HEIGHT != height) {
        resize(binary_frame, binary_frame, Size(LOCAL_FRAME_WIDTH, LOCAL_FRAME_HEIGHT), INTER_LANCZOS4);
        std::cerr << "resized\n";
        width = LOCAL_FRAME_WIDTH;
        height = LOCAL_FRAME_HEIGHT;
    }
*/



    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(binary_frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    cerr << "found " << contours.size() << " contours" << endl;

    vector<vector<Point> > save;
    for (int i = 0; i < contours.size(); ++i) {
        double perimeter = max(arcLength(contours[i], true), arcLength(contours[i], false));
        double area = contourArea(contours[i]);

        if (perimeter < 100) { continue; }

        //if (get_average_intensity(gray_frame, contours[i]) < 100) { continue; }
        save.push_back(contours[i]);
    }
    contours = save;

    binary_frame = Mat::zeros(binary_frame.size(), CV_8UC1);
    for (int i = 0; i < contours.size(); ++i) {
        Scalar color = Scalar(255, 255, 255);
        drawContours(binary_frame, contours, i, color, 2, 8, hierarchy, 0, Point());
     }

}

Line XLA::process_frame(Mat &bgr_frame, VideoWriter &bgr_writer, int &topview_height, int &topview_width, Rect &obj_box) {

    int width = bgr_frame.cols;
    int height = bgr_frame.rows;

    static Mat gray_frame;
    static Mat topview_frame;
    static Mat binary_frame;
    static Mat binary_object;
/*
    convert_to_topview(bgr_frame, topview_frame);
    if (LOCAL_FRAME_WIDTH != width || LOCAL_FRAME_HEIGHT != height) {
        resize(topview_frame, topview_frame, Size(LOCAL_FRAME_WIDTH, LOCAL_FRAME_HEIGHT), INTER_LANCZOS4);
        std::cerr << "resized\n";
        width = LOCAL_FRAME_WIDTH;
        height = LOCAL_FRAME_HEIGHT;
    }
    binary_frame = color_filter(topview_frame);
*/

    lets_be_handsome(bgr_frame, gray_frame, topview_frame, binary_frame, obj_box);
    width = binary_frame.cols;
    height = binary_frame.rows;
    topview_width = topview_frame.cols;
    topview_height = topview_frame.rows;
/*
    // old algorithm:
    // split half by middle vertical line
    // run the hough-transform to find the lines in each half

    Rect left_ROI(0, 0, width / 2, height);
    Rect right_ROI(width / 2, 0, width / 2, height);
    Mat left_image = binary_frame(left_ROI);
    Mat right_image = binary_frame(right_ROI);
*/

    // new algorithm
    // approximate the split line
    // use the line to split the frame into 2 parts
    // run the hough-transform in each parts
    // if the 2 approximation-lines are close to each other, state that we've got 1 lane
    vector<Line> all = get_all_lines(binary_frame);
    filter_lines_angle(all);
    //if (all.empty()) return Line(Point(binary_frame.cols / 2, 0), Point(binary_frame.cols / 2, binary_frame.rows - 1));

    Line split_line = Line(Point(width / 2, 0), Point(width / 2, height));
    if (!all.empty()) {
        split_line = approximate(all, get_steering_angle(all));
    }
    if (invalid_split(split_line, binary_frame.cols, binary_frame.rows)) {
        cerr << "invalid" << endl;
        split_line = Line(Point(width / 2, 0), Point(width / 2, height));
    }
    Mat left_image, right_image;
    do_cut(binary_frame, split_line, left_image, right_image);
    // cerr << "left " << left_image.rows << ' ' << left_image.cols << endl;
    // cerr << "right " << right_image.rows << ' ' << right_image.cols << endl;
#ifdef SHOW_OUTPUT
    //imshow("left-image", left_image); imshow("right-image", right_image);
#endif
    vector<Line> left_lane = process_one_side(left_image);
    vector<Line> right_lane = process_one_side(right_image);
    for (int i = 0; i < right_lane.size(); ++i) {
        right_lane[i].shift(binary_frame.cols - right_image.cols);
    }
    Line left_line = approximate(left_lane, get_steering_angle(left_lane));
    Line right_line = approximate(right_lane, get_steering_angle(right_lane));
    Line res;
    bool from_all = false;
    if (abs(left_line.intersect(binary_frame.rows / 2) - right_line.intersect(binary_frame.rows / 2)) < 64) {
        res = split_line;
        res.is_left = split_line.intersect((int)binary_frame.rows) < (binary_frame.cols / 2) ? 1 : 0;
        from_all = true;
    } else {
        double sum_left = get_sum_length(left_lane);
        double sum_right = get_sum_length(right_lane);
        if (min(sum_left, sum_right) / (sum_left + sum_right) < 0.2) {
            //one line dominates
            if (sum_left > sum_right) {
                res = left_line;
                res.is_left = 1;
            } else {
                res = right_line;
                res.is_left = 0;
            }
            //res.is_left = split_line.intersect((int)binary_frame.rows) < (binary_frame.cols / 2) ? 1 : 0;
        }
        else {
            //both lane equal in length
            res.Q = Point((left_line.intersect(0) + right_line.intersect(0)) / 2, 0);
            res.P = Point((left_line.intersect(binary_frame.rows) + right_line.intersect(binary_frame.rows)) / 2, binary_frame.rows);
            res.calc_params();
            res.is_left = -1;
        }
    }
    cerr << "res.is_left = " << res.is_left << endl;
    Mat hough_frame = topview_frame.clone();

#ifdef WRITE_VIDEO
    if (from_all) {
        draw_lines(hough_frame, all);
    } else {
        draw_lines(hough_frame, left_lane);
        draw_lines(hough_frame, right_lane);
    }
    Point A = Point(res.intersect(0), 0);
    Point B = Point(res.intersect(height), height);
    line(hough_frame, A, B, Scalar(0, 255, 0), 3, 8);

    show_angle(hough_frame, res.angle());
    rectangle(hough_frame, obj_box.tl(), obj_box.br(), Scalar(0, 255, 0));
    cerr << "res.angle() = " << res.angle() << endl;
#endif
    //convert_back(hough_frame, hough_frame);

#ifdef SHOW_OUTPUT
    imshow("bgr", bgr_frame);
    //imshow("gray", gray_frame);
    line(topview_frame, Point(split_line.intersect(0), 0), Point(split_line.intersect(height - 1), height - 1), Scalar(100, 100, 100), 2, 8);
    imshow("topview", topview_frame);
    imshow("binary", binary_frame);
    imshow("hough", hough_frame);
#endif

#ifdef WRITE_VIDEO
    if (hough_frame.channels() == 1) {
        cvtColor(hough_frame, hough_frame, CV_GRAY2BGR);
    }
    cerr << "hough_frame" << hough_frame.cols << ' ' << hough_frame.rows << endl;
    //bgr_writer << hough_frame;
    Mat big_frame = Mat::zeros(hough_frame.rows * 3 + 3, bgr_frame.cols + hough_frame.cols + 1, CV_8UC3);
    cvtColor(binary_frame, binary_frame, CV_GRAY2BGR);
    bgr_frame.copyTo(big_frame(cv::Rect(0, 0, bgr_frame.cols, bgr_frame.rows)));
    topview_frame.copyTo(big_frame(cv::Rect(bgr_frame.cols + 1, 0, topview_frame.cols, topview_frame.rows)));
    binary_frame.copyTo(big_frame(cv::Rect(bgr_frame.cols + 1, topview_frame.rows + 1, binary_frame.cols, binary_frame.rows)));
    hough_frame.copyTo(big_frame(cv::Rect(bgr_frame.cols + 1, topview_frame.rows * 2 + 2, hough_frame.cols, hough_frame.rows)));
    cerr << "BIG FRAME " << big_frame.cols << ' ' << big_frame.rows << endl;
    imshow("big_frame", big_frame);
    bgr_writer << big_frame;
#endif

    return res;
}

double XLA::adjust_angle(Line previous_line, Line current_line, int topview_height, int topview_width, Rect &obj_box) {
    double multiplier = 4.0;
    Point center_point(-1, 0);

    int y = topview_height / 4;
    if (current_line.is_left != -1) {
        if (current_line.is_left == 1) { //left
            center_point.x = current_line.intersect(y) + topview_width / 10.0;
        } else { //right
            center_point.x = current_line.intersect(y) - topview_width / 10.0;
        }
    } else {
        center_point.x = current_line.intersect(y);
    }

    if (obj_box.area() > 0) {
        double middle = obj_box.x + obj_box.width / 2.0;
        if (abs(center_point.x - middle) <= obj_box.width / 2.0 + 30) {

            if (center_point.x < middle) {
                center_point.x -= obj_box.width / 2;
                cerr << "shift_left" << endl;
            } else {
                center_point.x += obj_box.width / 2;
                cerr << "shift_right" << endl;
            }
        }
    }

    Point from(topview_width / 2, topview_height);
    Line ref(from, center_point);

    double current_angle = ref.angle() / PI * 180 - 90;

    //if (current_angle > 20) current_angle = 20;
    //if (current_angle < -20) current_angle = -20;

    return current_angle * (-1) * multiplier;
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
        std::cerr << "xy = " << x << ", " << y << endl;
    }
    finp.close();
    this->bird = TopviewConverter(orig);
}

void XLA::filter_lines_angle(vector<Line> &d) {
    double sum_dist_horizontal = 0;
    double sum_dist_vertical = 0;
    for (int i = 0; i < d.size(); ++i) {
        if (min(d[i].angle(), PI - d[i].angle()) < PI / 6) {
            sum_dist_horizontal += d[i].length();
        } else {
            sum_dist_vertical += d[i].length();
        }
    }
    //if (sum_dist_horizontal / sum_dist vertical < 1.0 / 3.0)
    {
        vector<Line> temp;
        for (int i = 0; i < d.size(); ++i) {
            assert(d[i].angle() >= 0 && d[i].angle() < PI);

            if (min(d[i].angle(), PI - d[i].angle()) >= PI / 9) {//PI * 9 / 9) {
                temp.push_back(d[i]);
            }
        }
        d = temp;
    }
}

vector<Line> XLA::filter_lines(vector<Line> &d) {
    filter_lines_angle(d);
    Line ref = approximate(d, get_steering_angle(d));

    double avg_dist = 0;
    for (int i = 0; i < d.size(); ++i) {
        avg_dist += ref.dist_to_point(d[i].P);
        avg_dist += ref.dist_to_point(d[i].Q);
    }
    avg_dist /= d.size() * 2;

    vector<Line> res;
    for (int i = 0; i < d.size(); ++i) {
        if (angle_between_two_lines(d[i], ref) > PI / 3) {
            continue;
        }
        double dist = (ref.dist_to_point(d[i].P) + ref.dist_to_point(d[i].Q)) / 2;
        if (dist > avg_dist * 2) {
            continue;
        }
        res.push_back(d[i]);
    }
    return res;
}

Mat XLA::color_filter(Mat frame) {
    int low_b = 0, low_g = 0, low_r = 110;
    int high_b = 230, high_g = 255, high_r = 255;
    int morph_elem = 1;
    int morph_size = 10;
    Mat threshold, res, hsv, mor, final;
    assert(frame.channels() == 3);
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    inRange(hsv, Scalar(low_b, low_g, low_r), Scalar(high_b, high_g, high_r), threshold);
    bitwise_and(frame, frame, res, threshold);
    mor = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( res, final, 3, mor );
    if (final.channels() == 3) cvtColor(final, final, CV_BGR2GRAY);
    cv::threshold(final, final, 188, 255, THRESH_BINARY);
    return final;
}

void XLA::do_cut(const Mat &binary_image, Line split_line, Mat &left_image, Mat &right_image) {
    int up = split_line.intersect(0);
    int down = split_line.intersect(binary_image.rows - 1);
    Rect left_rect(0, 0, max(up, down), binary_image.rows);
    Rect right_rect(min(up, down), 0, binary_image.cols - min(up, down), binary_image.rows);
    left_image = Mat(left_rect.height, left_rect.width, CV_8UC1);
    right_image = Mat(right_rect.height, right_rect.width, CV_8UC1);
    for (int x = 0; x < binary_image.cols; ++x) {
        for (int y = 0; y < binary_image.rows; ++y) {
            int middle_x = split_line.intersect(y);
            if (left_rect.contains(Point(x, y))) {
                if (x <= middle_x) {
                    left_image.at<uchar>(y, x) = binary_image.at<uchar>(y, x);
                } else {
                    left_image.at<uchar>(y, x) = 0;
                }
            }
            if (right_rect.contains(Point(x, y))) {
                if (x > middle_x) {
                    right_image.at<uchar>(y, x - right_rect.x) = binary_image.at<uchar>(y, x);
                } else {
                    right_image.at<uchar>(y, x - right_rect.x) = 0;
                }
            }
        }
    }
    std::cerr << "RETURN do_cut\n";
}

bool XLA::invalid_split(Line split_line, int x, int y) {
    if (split_line.intersect(0) < 0 || split_line.intersect(0) >= x) return true;
    if (split_line.intersect(y - 1) < 0 || split_line.intersect(y - 1) >= x) return true;
    return false;
}

void XLA::filter_binary_image(Mat &topview, Mat &binary) {
    for (int y = 0; y < topview.rows; ++y) {
        int l, r;
        for (int x = 0; x < topview.cols && topview.at<uchar>(y, x) == 0; ++x) {
            l = x;
        }
        for (int x = topview.cols - 1; x >= 0 && topview.at<uchar>(y, x) == 0; --x) {
            r = x;
        }
        for (int x = 0; x < l + 10 && x < binary.cols; ++x) binary.at<uchar>(y, x) = 0;
        for (int x = binary.cols - 1; x > r - 10 && x >= 0; --x) binary.at<uchar>(y, x) = 0;
    }
}
/*
double XLA::get_stupid_angle(Mat &bgr_frame, VideoWriter &bgr_writer) {
    cerr << "ENTER stupid\n";
    int width = bgr_frame.cols;
    int height = bgr_frame.rows;

    static Mat gray_frame;
    static Mat topview_frame;
    static Mat binary_frame;

    lets_be_handsome(bgr_frame, gray_frame, topview_frame, binary_frame);
    width = binary_frame.cols;
    height = binary_frame.rows;
    // new algorithm
    // approximate the split line
    // use the line to split the frame into 2 parts
    // run the hough-transform in each parts
    // if the 2 approximation-lines are close to each other, state that we've got 1 lane
    vector<Line> all = get_all_lines(binary_frame);
    double res; //radian
    double max_min_dist = 0;
    Point from(binary_frame.cols / 2, binary_frame.rows);
    // for (double angle = -PI / 6; angle < PI / 6; angle += 0.1) {
    //     // double abs_angle = angle + PI / 2;
    //     // double min_dist = 1e18;

    //     // Line cur(from, Point(0, tan(PI - abs_angle) * binary_frame.cols / 2));
    //     // for (int i = 0; i < all.size(); ++i) {
    //     //     Point M = intersection(cur, all[i]);
    //     //     if (all[i].contain_point(M)) {
    //     //         cerr << "contain !!! " << angle << endl;
    //     //         if (min_dist > norm(from - M)) {
    //     //             min_dist = norm(from - M);
    //     //         }
    //     //     }
    //     // }

    //     // for (int x = binary_frame.cols / 2 + 1; x < binary_frame.cols / 2 + 50; ++x) {
    //     //     int y = binary_frame.rows - x * tan(abs_angle);
    //     //     if (binary_frame.at<uchar>(y, x) == 255) {
    //     //         min_dist =
    //     //     }
    //     // }
    //     // cerr << "angle, min_dist = " << angle << ' ' << min_dist << endl;
    //     // if (max_min_dist < min_dist) {
    //     //     max_min_dist = min_dist;
    //     //     res = angle;
    //     // }
    //     Point to(binary_frame.cols/2 + binary_frame.rols * tan(angle), 0);
    //     Line cur(from, to);
    //     LineIterator(binary_frame, , Point pt2)
    // }
    Point target(0,0);
    for(int y = binary_frame.rows; y>=0 ; y--){
        Point to(0,y);
        Line cur(from, to);
        double min_dist = 1e9;
        LineIterator it(binary_frame, from, to, 8);
        for(int i=0;i<it.count;i++,++it){
            Point pt = it.pos();
            if(binary_frame.at<uchar>(pt.y,pt.x)==255 || pt.y == y){
                min_dist = norm(from - pt);
                break;
            }
        }
        if(min_dist>max_min_dist){
            max_min_dist = min_dist;
            target = to;
        }
    }
    for(int x = 0; x<=binary_frame.cols ; x++){
        Point to(x,0);
        Line cur(from, to);
        double min_dist = 1e9;
        LineIterator it(binary_frame, from, to, 8);
        for(int i=0;i<it.count;i++,++it){
            Point pt = it.pos();
            if(binary_frame.at<uchar>(pt.y,pt.x)==255 || pt.x == x){
                min_dist = norm(from - pt);
                break;
            }
        }
        if(min_dist>max_min_dist){
            max_min_dist = min_dist;
            target = to;
        }
    }
    for(int y = 0; y<=binary_frame.rows ; y++){
        Point to(binary_frame.cols,y);
        Line cur(from, to);
        double min_dist = 1e9;
        LineIterator it(binary_frame, from, to, 8);
        for(int i=0;i<it.count;i++,++it){
            Point pt = it.pos();
            if(binary_frame.at<uchar>(pt.y,pt.x)==255 || pt.y == y){
                min_dist = norm(from - pt);
                break;
            }
        }
        if(min_dist>max_min_dist){
            max_min_dist = min_dist;
            target = to;
        }
    }
    Line pro(from, target);

    Mat hough_frame = topview_frame.clone();

#ifdef WRITE_VIDEO
    draw_lines(hough_frame, all);
    show_angle(hough_frame, res + PI / 2);
#endif

#ifdef SHOW_OUTPUT
    imshow("bgr", bgr_frame);
    imshow("topview", topview_frame);
    imshow("binary", binary_frame);
    imshow("hough", hough_frame);
#endif

#ifdef WRITE_VIDEO
    if (hough_frame.channels() == 1) {
        cvtColor(hough_frame, hough_frame, CV_GRAY2BGR);
    }
    bgr_writer << bgr_frame;
#endif

    return pro.angle() / PI * 180 - 90;
}
*/
