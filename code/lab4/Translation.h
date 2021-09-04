#pragma once
#include "stdafx.h"
using namespace cv::ml;



Mat Translation(Mat& inputImage, int dx, int dy);
//Mat Rotate(Mat& src, int angle1);
Mat Resize(Mat& src, double scale);
Mat Perspective(Mat &src);
int OtsuAlgThreshold(const Mat image);
Mat Expand(Mat &img);
Mat Corrosion(Mat &img);
/*Mat Div_Linear_trans(Mat& src, int l, int k1);  //线性灰度变换
Mat equalize_hist(Mat& input);
Mat thinning(Mat & binaryImg); //骨架提取
Mat findcorners(Mat &src); //角点检测
void drawOnImage(const cv::Mat &binary, cv::Mat &image);
Mat findcor(Mat &srcImg);
Point2f(*choose_contour(vector<vector<Point>> contours))[2];
int license_gain(Point2f(*choose_license)[2], Mat img);
*/
//void color_rect(cv::Mat& img, cv::Mat& out);
//void tiqu(Mat& img, Mat& out);
void Recognition(Mat& img, Mat& lic, Mat& out, string &proc, char sss[], Mat NumImg[], bool &flag1, bool &flag2);

void selectRect(vector<cv::Rect>rt, vector<cv::Rect>& LRect);
string Province_Show(int Label);
