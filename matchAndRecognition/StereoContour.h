#define _CRT_SECURE_NO_WARNINGS
#ifndef _STEREO_CONTOUR_H_
#define _STEREO_CONTOUR_H_

#include <vector>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\contrib\contrib.hpp>

using namespace std;
using namespace cv;

class StereoContour
{
public:
	StereoContour(int _edgeMethed = 1, int _lowThreshold = 50, int _highThreshold = 100);
	virtual ~StereoContour(void);

	void monoFindContour(Mat src, vector<vector<Point> >& contours);

	void stereoFindContourFeature(Mat left, Mat right, vector<double>& feature);
	
	void setRectInMask(Mat src, Mat& roi);

private:
	int edgeMethed;//0为canny,1为sobel,2为拉普拉斯
	int lowThreshold,highThreshold;
};
#endif