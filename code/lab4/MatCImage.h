#pragma once
#include "stdafx.h"
#include <opencv2/highgui\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "opencv2/imgproc/imgproc_c.h"   //��������Ͳ��ᱨ����

class MatCImage
{
public:
	/*MatToCImage
	*��飺
	*	OpenCV��MatתATL/MFC��CImage����֧�ֵ�ͨ���ҶȻ���ͨ����ɫ
	*������
	*	mat��OpenCV��Mat
	*	cimage��ATL/MFC��CImage
	*/
	void MatToCImage(Mat& mat, CImage& cimage);


	/*CImageToMat
	*��飺
	*	ATL/MFC��CImageתOpenCV��Mat����֧�ֵ�ͨ���ҶȻ���ͨ����ɫ
	*������
	*	cimage��ATL/MFC��CImage
	*	mat��OpenCV��Mat
	*/
	void CImageToMat(CImage& cimage, Mat& mat);
};