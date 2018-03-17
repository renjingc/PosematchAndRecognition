#include "StereoContour.h"

StereoContour::StereoContour(int _edgeMethed,int _lowThreshold, int _highThreshold):
edgeMethed(_edgeMethed), lowThreshold(_lowThreshold), highThreshold(_highThreshold)
{

}

StereoContour::~StereoContour()
{

}
void StereoContour::setRectInMask(Mat src, Mat& roi)
{
	Point center;
	int length = src.cols/3;
	Rect rect;
	center.x = src.cols / 2;
	center.y = src.rows / 2;
	rect.x = (center.x - length) <= 0 ? 0 : (center.x - length);
	rect.y = (center.y - length) <= 0 ? 0 : (center.y - length);
	rect.width = src.cols / 2 <= length ? src.cols - 1 : length * 2;
	rect.height = src.rows / 2 <= length ? src.rows - 1 : length * 2;
	roi = src(rect);
}
void StereoContour::monoFindContour(Mat src, vector<vector<Point> >& contours)
{
	Mat src_gray;
	if (src.channels() == 3)
		cvtColor(src, src_gray, CV_RGB2GRAY);
	else
		src.copyTo(src_gray);
	Mat roi;
	setRectInMask(src_gray, roi);
	Mat edgeImage;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	int kernel_size = 3;

	//去噪
	/// Reduce noise with a kernel 3x3
	GaussianBlur(roi, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	//边缘提取
	//canny
	if (edgeMethed==0)
	{
		Canny(roi, edgeImage, lowThreshold, lowThreshold * 3, kernel_size);
	}
	//sobel
	else if (edgeMethed==1)
	{
		/// Generate grad_x and grad_y
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		//Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
		Sobel(roi, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);//转回uint8 
		/// Gradient Y  
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(roi, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);//转回uint8 
		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edgeImage);
	}
	//Laplace
	else if (edgeMethed == 2)
	{
		Mat dst;
		Laplacian(roi, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(dst, edgeImage);//转回uint8 
	}

	//轮廓提取
	/// Detect edges using Threshold
	int thresh = 50;
	Mat threshold_output;
	vector<Vec4i> hierarchy;
	threshold(edgeImage, threshold_output, thresh, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
	RNG rng(12345);
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//if (area[i] < h * w * 0.9)
		//{
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}
	//Mat result(src_gray.size(), CV_8U, Scalar(0));
	//drawContours(result, contours,      //画出轮廓
	//	-1, // draw all contours
	//	Scalar(255), // in black
	//	2); // with a thickness of 2

	imshow("edgeImage", edgeImage);
	imshow("Contours", drawing);
	imshow("threshold_output", threshold_output);
	imshow("src", roi);
	waitKey(0);
}

void StereoContour::stereoFindContourFeature(Mat left, Mat right, vector<double>& feature)
{

}