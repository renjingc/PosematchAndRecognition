#define _CRT_SECURE_NO_WARNINGS
#ifndef _FEATURE_MATCH_H_
#define _FEATURE_MATCH_H_

#include <memory>
#include <iostream>
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


struct Pattern
{
	cv::Mat image;
	std::vector<cv::KeyPoint>  keypoints;
	cv::Mat descriptors;

	Pattern(cv::Mat& img) :
		image(img) {}
};


class FeatureMatch
{
public:
	FeatureMatch(std::shared_ptr<Pattern> left, std::shared_ptr<Pattern> right, std::shared_ptr<cv::DescriptorMatcher> matcher,int _detectorMethod=0);

	void match(std::vector<cv::DMatch>& matches);

	void knnMatch(std::vector<cv::DMatch>& matches);

	void refineMatcheswithHomography(std::vector<cv::DMatch>& matches, double reprojectionThreshold, cv::Mat& homography);

	void refineMatchesWithFundmentalMatrix(std::vector<cv::DMatch>& matches, cv::Mat& F, double param1, double param2);

	//void showMatches(const std::vector<cv::DMatch>& matches,cv::Mat& matchesImg, const string& windowName);
	void getRidOfDist(vector<DMatch> src, vector<DMatch>& out);
private:
	std::shared_ptr<cv::DescriptorMatcher> matcher;
	std::shared_ptr<Pattern> leftPattern;
	std::shared_ptr<Pattern> rightPattern;

	int detectorMethod;//0Ϊsift,1Ϊsift
};


#endif