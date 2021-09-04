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


	//�ؽ�cimage
	cimage.Destroy();
	cimage.Create(nWidth, nHeight, 8 * nChannels);


	//��������


	uchar* pucRow;									//ָ������������ָ��
	uchar* pucImage = (uchar*)cimage.GetBits();		//ָ����������ָ��
	int nStep = cimage.GetPitch();					//ÿ�е��ֽ���,ע���������ֵ�����и�


	if (1 == nChannels)								//���ڵ�ͨ����ͼ����Ҫ��ʼ����ɫ��
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


	//�ؽ�mat
	if (1 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC1);
	}
	else if (3 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC3);
	}


	//��������


	uchar* pucRow;									//ָ������������ָ��
	uchar* pucImage = (uchar*)cimage.GetBits();		//ָ����������ָ��
	int nStep = cimage.GetPitch();					//ÿ�е��ֽ���,ע���������ֵ�����и�


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
	int T = 0; //Otsu�㷨��ֵ
	double varValue = 0; //��䷽���м�ֵ����
	double w0 = 0; //ǰ�����ص�����ռ����
	double w1 = 0; //�������ص�����ռ����
	double u0 = 0; //ǰ��ƽ���Ҷ�
	double u1 = 0; //����ƽ���Ҷ�
	double Histogram[256] = { 0 }; //�Ҷ�ֱ��ͼ���±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ�����ص�����
	uchar *data = image.data;
	double totalNum = image.rows*image.cols; //��������
	//����Ҷ�ֱ��ͼ�ֲ���Histogram�����±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ���ص���
	for (int i = 0; i < image.rows; i++)   //Ϊ������������û�а�rows��cols���������
	{
		for (int j = 0; j < image.cols; j++)
		{
			Histogram[data[i*image.step + j]]++;
		}
	}
	for (int i = 0; i < 255; i++)
	{
		//ÿ�α���֮ǰ��ʼ��������
		w1 = 0;		u1 = 0;		w0 = 0;		u0 = 0;
		//***********����������ֵ����**************************
		for (int j = 0; j <= i; j++) //�������ָ�ֵ����
		{
			w1 += Histogram[j];  //�����������ص�����
			u1 += j * Histogram[j]; //�������������ܻҶȺ�
		}
		if (w1 == 0) //�����������ص���Ϊ0ʱ�˳�
		{
			break;
		}
		u1 = u1 / w1; //��������ƽ���Ҷ�
		w1 = w1 / totalNum; // �����������ص�����ռ����
		//***********����������ֵ����**************************

		//***********ǰ��������ֵ����**************************
		for (int k = i + 1; k < 255; k++)
		{
			w0 += Histogram[k];  //ǰ���������ص�����
			u0 += k * Histogram[k]; //ǰ�����������ܻҶȺ�
		}
		if (w0 == 0) //ǰ���������ص���Ϊ0ʱ�˳�
		{
			break;
		}
		u0 = u0 / w0; //ǰ������ƽ���Ҷ�
		w0 = w0 / totalNum; // ǰ���������ص�����ռ����
		//***********ǰ��������ֵ����**************************

		//***********��䷽�����******************************
		double varValueI = w0 * w1*(u1 - u0)*(u1 - u0); //��ǰ��䷽�����
		//cout << varValueI << endl;
		if (varValue < varValueI)
		{
			varValue = varValueI;
			T = i;
		}
	}
	return T;
}

