#include "line_segment.h"

cv::Point intersection(const Line d1, const Line d2) {
    double D  = d1.a * d2.b - d1.b * d2.a;
    double Dx = d1.c * d2.b - d1.b * d2.c;
    double Dy = d1.a * d2.c - d1.c * d2.a;
    return cv::Point(Dx / D, Dy / D);
}

double angle_between_two_lines(Line d1, Line d2) {
    double diff = d1.angle() - d2.angle();
    while (diff < 0) diff += PI * 2;
    while (diff > 2 * PI) diff -= PI * 2;
    if (diff > PI) diff = 2 * PI - diff;
    if (diff > PI / 2) diff = PI - diff;
    return diff;
}