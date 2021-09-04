#include "stdafx.h"
#include "MatCImage.h"


void MatCImage::MatToCImage(Mat& mat, CImage& cimage)
{
	if (0 == mat.total())
	{
		return;
	}


	int nChannels = mat.channels();
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = mat.cols;
	int nHeight = mat.rows;


	//重建cimage
	cimage.Destroy();
	cimage.Create(nWidth, nHeight, 8 * nChannels);


	//拷贝数据


	uchar* pucRow;									//指向数据区的行指针
	uchar* pucImage = (uchar*)cimage.GetBits();		//指向数据区的指针
	int nStep = cimage.GetPitch();					//每行的字节数,注意这个返回值有正有负


	if (1 == nChannels)								//对于单通道的图像需要初始化调色板
	{
		RGBQUAD* rgbquadColorTable;
		int nMaxColors = 256;
		rgbquadColorTable = new RGBQUAD[nMaxColors];
		cimage.GetColorTable(0, nMaxColors, rgbquadColorTable);
		for (int nColor = 0; nColor < nMaxColors; nColor++)
		{
			rgbquadColorTable[nColor].rgbBlue = (uchar)nColor;
			rgbquadColorTable[nColor].rgbGreen = (uchar)nColor;
			rgbquadColorTable[nColor].rgbRed = (uchar)nColor;
		}
		cimage.SetColorTable(0, nMaxColors, rgbquadColorTable);
		delete[]rgbquadColorTable;
	}


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				*(pucImage + nRow * nStep + nCol) = pucRow[nCol];
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					*(pucImage + nRow * nStep + nCol * 3 + nCha) = pucRow[nCol * 3 + nCha];
				}
			}
		}
	}
}

void MatCImage::CImageToMat(CImage& cimage, Mat& mat)
{
	if (true == cimage.IsNull())
	{
		return;
	}


	int nChannels = cimage.GetBPP() / 8;
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = cimage.GetWidth();
	int nHeight = cimage.GetHeight();


	//重建mat
	if (1 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC1);
	}
	else if (3 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC3);
	}


	//拷贝数据


	uchar* pucRow;									//指向数据区的行指针
	uchar* pucImage = (uchar*)cimage.GetBits();		//指向数据区的指针
	int nStep = cimage.GetPitch();					//每行的字节数,注意这个返回值有正有负


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				pucRow[nCol] = *(pucImage + nRow * nStep + nCol);
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					pucRow[nCol * 3 + nCha] = *(pucImage + nRow * nStep + nCol * 3 + nCha);
				}
			}
		}
	}
}

int OtsuAlgThreshold(const Mat image)
{
	if (image.channels() != 1)
	{
		cout << "Please input Gray-image!" << endl;
		return 0;
	}
	int T = 0; //Otsu算法阈值
	double varValue = 0; //类间方差中间值保存
	double w0 = 0; //前景像素点数所占比例
	double w1 = 0; //背景像素点数所占比例
	double u0 = 0; //前景平均灰度
	double u1 = 0; //背景平均灰度
	double Histogram[256] = { 0 }; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数
	uchar *data = image.data;
	double totalNum = image.rows*image.cols; //像素总数
	//计算灰度直方图分布，Histogram数组下标是灰度值，保存内容是灰度值对应像素点数
	for (int i = 0; i < image.rows; i++)   //为表述清晰，并没有把rows和cols单独提出来
	{
		for (int j = 0; j < image.cols; j++)
		{
			Histogram[data[i*image.step + j]]++;
		}
	}
	for (int i = 0; i < 255; i++)
	{
		//每次遍历之前初始化各变量
		w1 = 0;		u1 = 0;		w0 = 0;		u0 = 0;
		//***********背景各分量值计算**************************
		for (int j = 0; j <= i; j++) //背景部分各值计算
		{
			w1 += Histogram[j];  //背景部分像素点总数
			u1 += j * Histogram[j]; //背景部分像素总灰度和
		}
		if (w1 == 0) //背景部分像素点数为0时退出
		{
			break;
		}
		u1 = u1 / w1; //背景像素平均灰度
		w1 = w1 / totalNum; // 背景部分像素点数所占比例
		//***********背景各分量值计算**************************

		//***********前景各分量值计算**************************
		for (int k = i + 1; k < 255; k++)
		{
			w0 += Histogram[k];  //前景部分像素点总数
			u0 += k * Histogram[k]; //前景部分像素总灰度和
		}
		if (w0 == 0) //前景部分像素点数为0时退出
		{
			break;
		}
		u0 = u0 / w0; //前景像素平均灰度
		w0 = w0 / totalNum; // 前景部分像素点数所占比例
		//***********前景各分量值计算**************************

		//***********类间方差计算******************************
		double varValueI = w0 * w1*(u1 - u0)*(u1 - u0); //当前类间方差计算
		//cout << varValueI << endl;
		if (varValue < varValueI)
		{
			varValue = varValueI;
			T = i;
		}
	}
	return T;
}

