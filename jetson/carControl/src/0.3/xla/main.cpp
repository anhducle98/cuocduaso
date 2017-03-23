#include <bits/stdc++.h>
#include <opencv/cv.hpp>
#include "xla.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Missing video_file_name argument\n";
        return 1;
    }

    VideoCapture video_capture(argv[1]);
    int number_of_frames = video_capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = video_capture.get(CV_CAP_PROP_FPS);
    int width = video_capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = video_capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("Process %d frames, FPS = %d, Size = %dx%d\n", number_of_frames, fps, width, height);

    //Rect ROI = Rect(0, height / 2, width, height / 2);

    VideoWriter bgr_writer;
    bgr_writer.open("bgr.avi", CV_FOURCC('X', 'V', 'I', 'D'), 10, Size(width, height));

    while (true) {
        cerr << "BEGIN #" << video_capture.get(CV_CAP_PROP_POS_FRAMES) << endl;

        static Mat bgr_frame;
        video_capture >> bgr_frame;
        if (bgr_frame.empty()) break;
        process_frame(bgr_frame, bgr_writer);
        
        cerr << "END #" << video_capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
    }

    bgr_writer.release();

    cerr << "Elapsed " << (double)clock() / CLOCKS_PER_SEC << endl;
    return 0;
}
