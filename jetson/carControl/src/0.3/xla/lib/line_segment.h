#ifndef LINE_SEGMENT_H
#define LINE_SEGMENT_H

#include <opencv/cv.hpp>

const double EPSILON = 1e-6;
const double PI = acos(-1);

class Line {
public:
    cv::Point P, Q; //endpoints
    double a, b, c; //ax + by = c

    Line(cv::Point P, cv::Point Q) {
        this->P = P; this->Q = Q;
        a = P.y - Q.y;
        b = Q.x - P.x;
        c = a * P.x + b * P.y;
    }

    double intersect(double y) {
        return (c - b * y) / a;
    }

    double angle() {
        double res = atan2(Q.y - P.y, Q.x - P.x);
        if (res < 0) {
            res += PI;
        }
        return res;
    }

    double dist_to_point(const cv::Point M) {
        if (contain_point(M)) {
            return 0;
        }
        return std::min(cv::norm(P - M), cv::norm(Q - M));
    }

    double length() {
        return cv::norm(P - Q);
    }

    bool contain_point(cv::Point M) {
        return fabs(cv::norm(P - M) + cv::norm(Q - M) - length()) < EPSILON;
    }
};

cv::Point intersection(const Line d1, const Line d2);
double angle_between_two_lines(Line d1, Line d2);

#endif