#include "FeatureMatch.h"
#include <opencv2\nonfree\features2d.hpp>

using namespace std;
using namespace cv;

FeatureMatch::FeatureMatch(std::shared_ptr<Pattern> left, std::shared_ptr<Pattern> right, std::shared_ptr<cv::DescriptorMatcher> matcher, int _detectorMethod) :
leftPattern(left), rightPattern(right), matcher(matcher), detectorMethod(_detectorMethod)
{

	//step1:Create detector
	int minHessian = 400;
	FeatureDetector *pDetector;
	if (detectorMethod == 0)
	{
		SiftFeatureDetector detector;
		//step2:Detecte keypoint
		detector.detect(leftPattern->image, leftPattern->keypoints);
		detector.detect(rightPattern->image, rightPattern->keypoints);

		//step3:Compute descriptor
		detector.compute(leftPattern->image, leftPattern->keypoints, leftPattern->descriptors);
		detector.compute(rightPattern->image, rightPattern->keypoints, rightPattern->descriptors);
	}
	else if (detectorMethod == 1)
	{
		SurfFeatureDetector detector(minHessian);
		//step2:Detecte keypoint
		detector.detect(leftPattern->image, leftPattern->keypoints);
		detector.detect(rightPattern->image, rightPattern->keypoints);

		//step3:Compute descriptor
		detector.compute(leftPattern->image, leftPattern->keypoints, leftPattern->descriptors);
		detector.compute(rightPattern->image, rightPattern->keypoints, rightPattern->descriptors);
	}
}

void FeatureMatch::match(vector<DMatch>& matches) 
{
	matcher->match(leftPattern->descriptors, rightPattern->descriptors, matches);
}

void FeatureMatch::knnMatch(vector<DMatch>& matches) 
{
	const float minRatio = 1.f / 1.5f;
	const int k = 2;

	vector<vector<DMatch>> knnMatches;
	matcher->knnMatch(leftPattern->descriptors, rightPattern->descriptors, knnMatches, k);

	for (size_t i = 0; i < knnMatches.size(); i++) 
	{
		const DMatch& bestMatch = knnMatches[i][0];
		const DMatch& betterMatch = knnMatches[i][1];

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
			matches.push_back(bestMatch);
	}
}

void FeatureMatch::refineMatcheswithHomography(vector<DMatch>& matches, double reprojectionThreshold, Mat& homography)
{
	const int minNumbermatchesAllowed = 8;
	if (matches.size() < minNumbermatchesAllowed)
		return;

	//Prepare data for findHomography
	vector<Point2f> srcPoints(matches.size());
	vector<Point2f> dstPoints(matches.size());

	for (size_t i = 0; i < matches.size(); i++) 
	{
		srcPoints[i] = rightPattern->keypoints[matches[i].trainIdx].pt;
		dstPoints[i] = leftPattern->keypoints[matches[i].queryIdx].pt;
	}

	//find homography matrix and get inliers mask
	vector<uchar> inliersMask(srcPoints.size());
	homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);

	vector<DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
}

void FeatureMatch::refineMatchesWithFundmentalMatrix(vector<DMatch>& matches, Mat& F,double param1,double param2) 
{
	getRidOfDist(matches, matches);
	//Align all points
	vector<KeyPoint> alignedKps1, alignedKps2;
	for (size_t i = 0; i < matches.size(); i++) {
		alignedKps1.push_back(leftPattern->keypoints[matches[i].queryIdx]);
		alignedKps2.push_back(rightPattern->keypoints[matches[i].trainIdx]);
	}

	//Keypoints to points
	vector<Point2f> ps1, ps2;
	for (unsigned i = 0; i < alignedKps1.size(); i++)
		ps1.push_back(alignedKps1[i].pt);

	for (unsigned i = 0; i < alignedKps2.size(); i++)
		ps2.push_back(alignedKps2[i].pt);

	// 分配空间
	int ptCount = (int)alignedKps1.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	// 把Keypoint转换为Mat
	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = leftPattern->keypoints[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = rightPattern->keypoints[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	//Compute fundmental matrix
	vector<uchar> status;
	F = findFundamentalMat(p1, p2, status, CV_FM_RANSAC, param1, param2);


	//优化匹配结果
	vector<KeyPoint> leftInlier;
	vector<KeyPoint> rightInlier;
	vector<DMatch> inlierMatch;

	int index = 0;
	for (unsigned i = 0; i < matches.size(); i++) 
	{
		if (status[i] != 0)
		{
			leftInlier.push_back(alignedKps1[i]);
			rightInlier.push_back(alignedKps2[i]);
			matches[i].trainIdx = index;
			matches[i].queryIdx = index;
			inlierMatch.push_back(matches[i]);
			index++;
		}
	}
	leftPattern->keypoints = leftInlier;
	rightPattern->keypoints = rightInlier;
	matches = inlierMatch;
}
void FeatureMatch::getRidOfDist(vector<DMatch> src, vector<DMatch>& out)
{
	// 分配空间
	int ptCount = (int)src.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	vector<double> dists,temp;
	// 把Keypoint转换为Mat
	Point2f pt;
	double mid;
	for (int i = 0; i<ptCount; i++)
	{
		pt = leftPattern->keypoints[src[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = rightPattern->keypoints[src[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
		dists.push_back((leftPattern->keypoints[src[i].queryIdx].pt.x - rightPattern->keypoints[src[i].trainIdx].pt.x)*(leftPattern->keypoints[src[i].queryIdx].pt.x - rightPattern->keypoints[src[i].trainIdx].pt.x) +
			(leftPattern->keypoints[src[i].queryIdx].pt.y - rightPattern->keypoints[src[i].trainIdx].pt.y)*(leftPattern->keypoints[src[i].queryIdx].pt.y - rightPattern->keypoints[src[i].trainIdx].pt.y));
	}
	temp = dists;
	sort(temp.begin(), temp.end());
	mid = temp[(int)(4*temp.size() / 5)];
	for (int i = 0; i < ptCount; i++)
	{
		if (dists[i] < mid)
			out.push_back(src[i]);
	}
}
/*void FeatureMatch::showMatches(const vector<DMatch>& matches,Mat& matchesImg, const string& windowName) 
{
	drawMatches(leftPattern->image, leftPattern->keypoints, rightPattern->image, rightPattern->keypoints, matches, matchesImg);
	namedWindow(windowName);
	imshow(windowName, matchesImg);
	waitKey();
	destroyWindow(windowName);
}*/