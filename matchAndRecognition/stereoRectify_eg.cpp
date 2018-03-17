//#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\calib3d\calib3d.hpp>
//#include <opencv2\contrib\contrib.hpp>
//#include <opencv\cv.h>
//#include <opencv\cvaux.h>
//#include <opencv\cxcore.h>
//#include <opencv\highgui.h>
//#include <iostream>
//#include <string>
//#include <vector>
//#include <direct.h>
//#include <io.h>
//#include <stdio.h>
//
//#include "StereoMatch.h"
//#include "filesTool.h"
//
//using namespace std;
//using namespace cv;
//
//string leftPath = "left";
//string rightPath = "right";
//string depthPath = "depth";
//
//string xml_filename = "calib_paras1.xml";
//string remap_filename = "remap.xml";
//string path = "训练";
//string path1 = "训练\\05行走\\left"; 
//string path2 = "训练\\05行走\\right";
//
//void F_Gray2Color(Mat gray_mat, Mat& color_mat)
//{
//	vector<Mat> channels(3);
//	Mat color(gray_mat.size(), CV_8UC3);
//	// 计算各彩色通道的像素
//	for (int i = 0; i < gray_mat.rows; i++)
//	{
//		for (int j = 0; j < gray_mat.cols; j++)
//		{
//			if ((int)gray_mat.at<uchar>(i, j) != 0)
//			{
//				color.at<cv::Vec3b>(i, j)[0] = 255 - (int)gray_mat.at<uchar>(i, j);
//				color.at<cv::Vec3b>(i, j)[2] = (int)gray_mat.at<uchar>(i, j);
//				if (gray_mat.at<uchar>(i, j) < 128)
//				{
//					color.at<cv::Vec3b>(i, j)[1] = (int)gray_mat.at<uchar>(i, j);
//				}
//				else
//				{
//					color.at<cv::Vec3b>(i, j)[1] = 255 - (int)gray_mat.at<uchar>(i, j);
//				}
//			}
//			else
//			{
//				color.at<cv::Vec3b>(i, j)[0] = 0;
//				color.at<cv::Vec3b>(i, j)[1] = 0;
//				color.at<cv::Vec3b>(i, j)[2] = 0;
//			}
//		}
//	}
//	color.copyTo(color_mat);
//}
//
//int getRightAndLeftDisp(string leftFile, string rightFile, Mat& dispColor, Mat& dispLColor, Mat& dispRColor)
//{
//
//	Mat leftImage = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
//	Mat rightImage = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
//
//	if (leftImage.empty())
//	{
//		cout << "leftImage empty" << endl;
//		return 1;
//	}
//	if (rightImage.empty())
//	{
//		cout << "rightImage empty" << endl;
//		return 1;
//	}
//
//	StereoMatch m_stereoMatcher;
//	m_stereoMatcher.init(leftImage.cols, leftImage.rows, xml_filename.c_str(), remap_filename);
//	// 开始计算图像视差
//	m_stereoMatcher.m_SGBM.disp12MaxDiff = -1;
//	m_stereoMatcher.m_SGBM.preFilterCap = 63;
//	m_stereoMatcher.m_SGBM.SADWindowSize = 3;
//	m_stereoMatcher.m_SGBM.P1 = 8 * leftImage.channels() * m_stereoMatcher.m_SGBM.SADWindowSize * m_stereoMatcher.m_SGBM.SADWindowSize;
//	m_stereoMatcher.m_SGBM.P2 = 32 * leftImage.channels() * m_stereoMatcher.m_SGBM.SADWindowSize * m_stereoMatcher.m_SGBM.SADWindowSize;
//	m_stereoMatcher.m_SGBM.minDisparity = 0;
//	m_stereoMatcher.m_SGBM.numberOfDisparities = 256;
//	m_stereoMatcher.m_SGBM.uniquenessRatio = 50;
//	m_stereoMatcher.m_SGBM.speckleWindowSize = 100;
//	m_stereoMatcher.m_SGBM.speckleRange = 32;
//	m_stereoMatcher.m_SGBM.fullDP = true;
//
//	Mat img1, img2, img1p, img2p, disp, dispL, dispR,disp8u, dispL8u, dispR8u, pointCloud;
//	m_stereoMatcher.uncalibratedSgbmMatch(leftImage, rightImage, disp, dispL, dispR, img1p, img2p, 0);
//	m_stereoMatcher.getDisparityImage(dispL, dispL8u, false);
//	m_stereoMatcher.getDisparityImage(dispR, dispR8u, false);
//	m_stereoMatcher.getDisparityImage(disp, disp8u, false);
//
//	F_Gray2Color(dispL8u, dispLColor);
//	F_Gray2Color(dispR8u, dispRColor);
//	F_Gray2Color(disp8u, dispColor);
//	return 0;
//
//	//resize(dispL8u, dispL8u, Size(640, 480));
//	//resize(dispR8u, dispR8u, Size(640, 480));
//	//resize(img1p, img1p, Size(640, 480));
//	//resize(img2p, img2p, Size(640, 480));
//	//imshow("dispL", dispL8u);
//	//imshow("dispR", dispR8u);
//	//imshow("img1p", img1p);
//	//imshow("img2p", img2p);
//	//waitKey(0);
//}
//void matchAndDepth()
//{
//
//}
//int main(int argc, char** argv)
//{
//	string filesPath;
//	vector<string> files;
//	getJustCurrentDir(path, files);
//	Mat leftDisp, rightDisp,disp;
//	for (int i = 9; i < files.size(); i++)
//	{
//		vector<string> leftFilesName, rightFilesName;
//		filesPath = path + "\\" + files[i];
//		getFilesAllName(filesPath + "\\" + leftPath, leftFilesName);
//		getFilesAllName(filesPath + "\\" + rightPath, rightFilesName);
//		sort(leftFilesName.begin(), leftFilesName.end());
//		sort(rightFilesName.begin(), rightFilesName.end());
//		for (int i = 0; i < leftFilesName.size(); i++)
//		{
//			string leftName = filesPath + "\\" + leftPath + "\\" + leftFilesName[i];
//			string rightName = filesPath + "\\" + rightPath + "\\" + rightFilesName[i];
//			leftName = "训练\\01双手平举\\left\\Left0012.bmp";
//			rightName = "训练\\01双手平举\\Right\\Right0012.bmp";
//			cout << leftName << endl;
//			if (!getRightAndLeftDisp(leftName,rightName, disp, leftDisp, rightDisp))
//			{
//				string savePath = filesPath + "\\" + depthPath;
//				_mkdir(savePath.c_str());
//				string saveLeftFile = savePath + "\\disp" + leftFilesName[i];
//				string saveRightFile = savePath + "\\disp" + rightFilesName[i];
//				//imwrite(saveLeftFile, leftDisp);
//				//imwrite(saveRightFile, rightDisp);
//				resize(leftDisp, leftDisp, Size(640, 480));
//				resize(rightDisp, rightDisp, Size(640, 480));
//				resize(disp, disp, Size(640, 480));
//				imshow("dispL", leftDisp);
//				imshow("dispR", rightDisp);
//				//imshow("disp", disp);
//				waitKey(0);
//			}
//			else
//			{
//				return 1;
//			}
//		}
//	}
//	cout << "finish" << endl;
//	return 0;
//}