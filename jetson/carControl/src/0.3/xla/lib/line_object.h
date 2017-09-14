//detect object using hsv
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;


int get_line_and_object(Mat frame, Mat &objectImg , Mat &lineImg);	
//frame : topview image
//objectImg : object only image
//lineImg : line only image
