#pragma once
#include "stdafx.h"
#include <opencv2/highgui\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "opencv2/imgproc/imgproc_c.h"   //加上这个就不会报错了

class MatCImage
{
public:
	/*MatToCImage
	*简介：
	*	OpenCV的Mat转ATL/MFC的CImage，仅支持单通道灰度或三通道彩色
	*参数：
	*	mat：OpenCV的Mat
	*	cimage：ATL/MFC的CImage
	*/
	void MatToCImage(Mat& mat, CImage& cimage);


	/*CImageToMat
	*简介：
	*	ATL/MFC的CImage转OpenCV的Mat，仅支持单通道灰度或三通道彩色
	*参数：
	*	cimage：ATL/MFC的CImage
	*	mat：OpenCV的Mat
	*/
	void CImageToMat(CImage& cimage, Mat& mat);
};