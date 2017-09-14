#include "line_object.h"
//detect object using color of object (red,green)

int get_line_and_object(Mat frame, Mat &objectImg , Mat &lineImg)
{
	Mat gray,hsv,bin;
    vector<Point> limitLineleft,limitLineright;

	cvtColor( frame, hsv, COLOR_BGR2HSV );	
	cvtColor( frame, gray, COLOR_BGR2GRAY );

	Mat binRedLow,binRedHigh,binGreen;
	inRange(hsv,Scalar(0,100,100,0),Scalar(10,255,255,0),binRedLow);
	inRange(hsv,Scalar(160,100,100,0),Scalar(179,255,255,0),binRedHigh);
	inRange(hsv,Scalar(35,100,100,0),Scalar(85,255,255,0),binGreen);

	Mat binWhite;
	inRange(hsv,Scalar(100,0,220,0),Scalar(180,255,255,0),binWhite);


	//********get line limited of topview image********		
	for(int i = 0;i < gray.rows; ++i)
	{
		for(int j = 0;j < gray.cols; ++j)
		{
			Scalar ref = gray.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			if (intensity != 0) 
			{
				limitLineleft.push_back(Point(j,i));
				break;
			}
		}

		for(int j = gray.cols - 1;j >= 0; --j)
		{ 
			Scalar ref = gray.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			if (intensity != 0) 
			{
				limitLineright.push_back(Point(j,i));
				break;
			}
		}
	}

	//**********Or image*********
	objectImg = Mat::zeros( frame.size(), CV_8U );
	for (int i = 0; i < binRedLow.rows; i++)
	for (int j = 0; j < binRedLow.cols; j++)
	{
		Scalar refRL = binRedLow.at<uchar>(i,j);
		Scalar refRH = binRedHigh.at<uchar>(i,j);
		Scalar refG = binGreen.at<uchar>(i,j);								
		int  intensityRL = refRL.val[0];
		int  intensityRH = refRH.val[0];
		int  intensityG = refG.val[0];
		if ((intensityRL == 255)||(intensityRH == 255)||(intensityG == 255)) objectImg.at<uchar>(i,j)= 255;
	}

	//**********filter morphology**********
	int morph_size = 3;
    Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ));
	morphologyEx( objectImg, objectImg, MORPH_DILATE, element, Point(-1,-1), 1 );


	morph_size = 3;
    element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ));
	morphologyEx( binWhite, binWhite, MORPH_CLOSE, element, Point(-1,-1), 1 );


	//**********remove small contours**********
	Mat img1 = objectImg.clone();
	Mat img2 = binWhite.clone();
	//change unuse image part limit to white for find contours
	for(int i = 0;i < img1.rows; ++i)	
	{
		for(int j = 0;j < limitLineleft[i].x+2; ++j) 
		{
			img1.at<uchar>(i,j) = 255;
			img2.at<uchar>(i,j) = 255;
		}
		for(int j = limitLineright[i].x-2;j < img1.cols ; ++j) 
		{
			img1.at<uchar>(i,j) = 255;
			img2.at<uchar>(i,j) = 255;
		}
	}
	Mat img1CT = img1.clone();
	Mat img2CT = img2.clone();
	//filter contour
	vector<vector<Point> > contours;
	findContours(img1CT, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	findContours(img2CT, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size()>0)
	{
		//bounding box
		double area;
		int ind = 0;
		Rect small_box[1000];
		for (unsigned int i = 0; i < contours.size(); i++)
		{                 			// for each contour				
			area = contourArea(contours[i]);	
			if (area < 1300) //if area of contour is small
			{	
				small_box[ind] = boundingRect(contours[i]);
				++ind;
			}
		}
		for (int k = 0;k<ind;k++)
		{
			for (int i = small_box[k].y;i < small_box[k].y + small_box[k].height;i++)
			for (int j = small_box[k].x;j < small_box[k].x + small_box[k].width;j++)
			{
				img1.at<uchar>(i,j) = 0;
				img2.at<uchar>(i,j) = 0;
			}
		}

		for(int i = 0;i < img1.rows; ++i)	
		{
			for(int j = 0;j < limitLineleft[i].x+2; ++j)
			{
				img1.at<uchar>(i,j) = 0;
				img2.at<uchar>(i,j) = 0;
			}
			for(int j = limitLineright[i].x-2;j < img1.cols ; ++j) 
			{
				img1.at<uchar>(i,j) = 0;
				img2.at<uchar>(i,j) = 0;
			}
		}
	}

	objectImg = img1.clone();
	lineImg = img2.clone();
}
