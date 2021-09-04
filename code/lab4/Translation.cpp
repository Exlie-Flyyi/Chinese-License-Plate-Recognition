#include "stdafx.h"
#include "lab4Dlg.h"
#include "Translation.h" 
//ȫ�ֱ�������
//Mat SrcImg;
//��ɫֵ����
int h_lower[2] = { 100 ,15};	
int s_lower = 52;
int v_lower = 46;
int h_upper[2] = { 120 ,40};
int s_upper = 255;
int v_upper = 255;
vector<cv::RotatedRect>ColorRect;
vector<int>CarLabel; //�ñ�ǩ���������
Mat Translation(Mat& inputImage, int dx, int dy) {
	CV_Assert(inputImage.depth() == CV_8U);
	int rows = inputImage.rows + abs(dy);   //���ͼ������
	int cols = inputImage.cols + abs(dx);   //���ͼ������
	Mat outputImage;
	outputImage.create(rows, cols, inputImage.type()); //�����µ��������� �������Ҫ����

	if (inputImage.channels() == 1)    //��ͨ���Ҷ�ͼ
	{
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				//����������ͼ���λ��
				int x = j - dx;
				int y = i - dy;
				if (x >= 0 && y >= 0 && x < inputImage.cols && y < inputImage.rows) {//��֤��ԭͼ�����ط�Χ��
					outputImage.at<uchar>(i, j) = inputImage.at<uchar>(y, x);
				}
			}
		}
	}

	if (inputImage.channels() == 3) {   //��ͨ����ͼ
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				int x = j - dx;
				int y = i - dy;
				if (x >= 0 && y >= 0 && x < inputImage.cols && y < inputImage.rows) {
					outputImage.at<cv::Vec3b>(i, j)[0] = inputImage.at<cv::Vec3b>(y, x)[0];
					outputImage.at<cv::Vec3b>(i, j)[1] = inputImage.at<cv::Vec3b>(y, x)[1];
					outputImage.at<cv::Vec3b>(i, j)[2] = inputImage.at<cv::Vec3b>(y, x)[2];
				}
			}
		}
	}
	return outputImage;
}
Mat Resize(Mat& src, double scale) {   //��������
	Mat out = Mat::zeros(src.size(), src.type());
	src.copyTo(out);
	resize(src, out, Size(src.cols * scale, src.rows * scale), 0, 0, INTER_LINEAR);
	return out;
}
Mat Perspective(Mat &src) {
	vector<Point2f> corners(4);
	int img_width = src.cols;
	int img_height = src.rows;
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(img_height - 1, 0);
	corners[2] = Point2f(0, img_width - 1);
	corners[3] = Point2f(img_width - 1, img_height - 1);
	vector<Point2f> corners_trans(4);
	corners_trans[0] = Point2f(50, 50);
	corners_trans[1] = Point2f(img_height - 1, 0);
	corners_trans[2] = Point2f(0, img_width - 1);
	corners_trans[3] = Point2f(img_width - 50, img_height - 60);

	Mat transform = getPerspectiveTransform(corners, corners_trans);
	Mat resultImage;
	warpPerspective(src, resultImage, transform, Size(img_width, img_height), INTER_LINEAR);
	return resultImage;
}


Mat Expand(Mat &img) {  //����
	Mat dilated_dst;
	Mat element_1 = getStructuringElement(MORPH_RECT, Size(4, 4));
	Mat element_2 = getStructuringElement(MORPH_CROSS, Size(15, 15));
	Mat element_3 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));

	dilate(img, dilated_dst, element_1);
	return dilated_dst;
}
Mat Corrosion(Mat &img) {  //��ʴ
	//�Զ��巽��
	Mat element_1 = getStructuringElement(MORPH_RECT, Size(4, 4));
	Mat element_2 = getStructuringElement(MORPH_CROSS, Size(15, 15));
	Mat element_3 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));

	Mat eroded_my, eroded_cv;
	erode(img, eroded_cv, element_1);
	return eroded_cv;
}


string Province_Show(int Label) {
	string pv = "";
	switch (Label)
	{
	case 0:pv = "��";
		break;
	case 1:pv = "��";
		break;
	case 2:pv = "��";
		break;
	case 3:pv = "��";
		break;
	case 4:pv = "��";
		break;
	case 5:pv = "��";
		break;
	case 6:pv = "��";
		break;
	case 7:pv = "��";
		break;
	case 8:pv = "��";
		break;
	case 9:pv = "��";
		break;
	case 10:pv = "��";
		break;
	case 11:pv = "ԥ";
		break;
	case 12:pv = "��";
		break;
	case 13:pv = "��";
		break;
	case 14:pv = "��";
		break;
	case 15:pv = "��";
		break;
	case 16:pv = "��";
		break;
	case 17:pv = "��";
		break;
	case 18:pv = "��";
		break;
	case 19:pv = "��";
		break;
	case 20:pv = "��";
		break;
	case 21:pv = "³";
		break;
	case 22:pv = "��";
		break;
	case 23:pv = "��";
		break;
	case 24:pv = "��";
		break;
	case 25:pv = "��";
		break;
	case 26:pv = "��";
		break;
	case 27:pv = "��";
		break;
	case 28:pv = "��";
		break;
	default:
		break;
	}
	return pv;
}

void selectRect(vector<cv::Rect>rt, vector<cv::Rect>& LRect)
{
	LRect.resize(2);
	cv::Rect temp;

	//numRect->LRect
	//�����������(�Ӵ�С)
	int Averarea, AverWidth, AverHeight;
	for (int i = 0; i < rt.size(); i++)
	{
		for (int j = i; j < rt.size(); j++)
		{
			if (rt[i].area() <= rt[j].area())
			{
				temp = rt[i];
				rt[i] = rt[j];
				rt[j] = temp;
			}
		}
	}
	//new->1
	Averarea = (rt[0].area() + rt[1].area()) / 2;
	AverHeight = (rt[0].height + rt[1].height) / 2;
	AverWidth = (rt[0].width + rt[1].width) / 2;

	//�������ȷ��ƽ��ֵ
	int AverHeight1 = 0, Averarea1 = 0, AverWidth1 = 0;
	int add = 0;
	for (int q = 0; q < rt.size(); q++)
	{
		if (rt[q].area() > 0.4 * Averarea && rt[q].height > AverHeight * 0.7)
		{
			AverHeight1 += rt[q].height;
			AverWidth1 += rt[q].width;
			Averarea1 += rt[q].area();
			add++;
		}
	}

	AverHeight1 /= add;
	AverWidth1 /= add;
	Averarea1 /= add;

	//����x��������˳��
	for (int i = 0; i < rt.size(); i++)
	{
		for (int j = i; j < rt.size(); j++)
		{
			if (rt[i].x >= rt[j].x) {
				temp = rt[i];
				rt[i] = rt[j];
				rt[j] = temp;
			}
		}
	}

	//�жϾ��ε�λ��
	//sum_one->Sumof1
	//ForSrect->random
	int Sumof1 = 0; //��������1�ĸ��������֡�������
	cv::Rect random;    //��һ�����ߵڶ�����ĸ
	int total = 0;  //randomǰ��ĵ�����ܺ�
	int minX = rt[0].x;//��߲�Ϊ�����Сֵ

	//��ʼ����ǰ�����������Ƿ�Ϊ��������������������
	for (int h = 0; h < rt.size() - 4; h++)
	{
		if (rt[h].x <= 2 + AverWidth1 / 20)
		{
			minX = rt[h + 1].x;
			continue;
		}
		total += rt[h].area();

		//�ж��Ƿ��Ǵ�
		if (rt[h].area() < Averarea1 * 0.5 && rt[h].height > AverHeight1*0.60)
		{
			Sumof1++;
		}

		if (rt[h].area() > Averarea1 * 0.8)
		{
			random = rt[h];
			total -= random.area();


			if (total > Averarea1 * 0.4 || Sumof1 >= 3)
			{
				//��
				//�ڶ���
				LRect[1] = random;
				//���һ��
				//ˮƽ����
				int maxX = rt[h - 1].x + rt[h - 1].width;
				//��ֱ����
				int minY = 1000;
				int maxY = 0;
				for (int w = 0; w < h; w++)
				{
					if (minY >= rt[w].y)
						minY = rt[w].y;
					if (maxY <= rt[w].y + rt[w].height)
						maxY = rt[w].y + rt[w].height;
				}
				LRect[0] = cv::Rect(cv::Point(minX, minY), cv::Point(maxX, maxY));
			}


			else//��
			{

				//��һ��
				LRect[0] = random;
				//��ڶ���
				h++;
				while (rt[h].area() < Averarea1 * 0.7 && h < (rt.size() - 2))
				{
					h++;
				}
				LRect[1] = rt[h];

			}
			h++;


			for (int d = h; d < rt.size(); d++)
			{
				if (rt[d].height > AverHeight1 * 0.6)
				{
					if (rt[d].width < AverWidth1 * 0.5) {
						rt[d].x = rt[d].x - (AverWidth1 - rt[d].width) / 2;
						rt[d].width = AverWidth1;
					}
					LRect.push_back(rt[d]);
				}
			}
			break;
		}
		
	}
}



void Recognition(Mat& img, Mat& lic ,Mat& out, string &proc, char sss[], Mat NumImg[], bool &flag1, bool &flag2)
{
	//����ʶ��һ��ͼƬ��ʱ��
	int CatchSuccess = 0;
	//����ͼƬ��СΪ900*900	
	//cout << img.rows << " " << img.cols << endl;
	if (img.rows > 900 || img.cols > 900)
	{
		int Wide = max(img.rows, img.cols);
		resize(img, img, cv::Size(), (double)900 / Wide, (double)900 / Wide);
		//std::cout << "�Ѿ�����ͼƬ��С" << endl;
	}
	for (int hi = 0; hi < 2; hi++) {
		flag1 = 0, flag2 = 0;
		if (CatchSuccess == 0)
		{
			//cout << v_lower << endl;
			//v_lower = 130;
			//hsv��ɫ��λ
			cv::Mat HsvImg;
			cv::Mat BinHsvImg;
			cvtColor(img, HsvImg, CV_BGR2HSV); //ת��HSVͼ��
			medianBlur(HsvImg, HsvImg, 3);
			medianBlur(HsvImg, HsvImg, 3);
			//GaussianBlur(HsvImg, HsvImg, Size(3, 3), 0, 0);
			inRange(HsvImg, cv::Scalar(h_lower[hi], s_lower, v_lower), cv::Scalar(h_upper[hi], s_upper, v_upper), BinHsvImg);  //��ֵ����ֵ��hsv��Χ������
			//cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
			//cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
			//morphologyEx(BinHsvImg, BinHsvImg, cv::MORPH_CLOSE, kernel1); //������
			//morphologyEx(BinHsvImg, BinHsvImg, cv::MORPH_OPEN, kernel2);  //������*/
			//������
			//imshow("���ͼƬ", BinHsvImg);
			BinHsvImg = Expand(BinHsvImg); BinHsvImg = Corrosion(BinHsvImg);
			//������
			BinHsvImg = Corrosion(BinHsvImg); BinHsvImg = Expand(BinHsvImg);
			//imshow("��̬ѧ���ͼƬ", BinHsvImg);
			//imwrite("stander.png", BinHsvImg);

			//�������
			vector<vector<cv::Point>>ColorContours;//���������ĵ����꣬�������������
			findContours(BinHsvImg, ColorContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//�������������,ֻ�������Χ����,����յ���Ϣ,��������߽������������������㵽contours������
			vector<cv::RotatedRect>ColorRect;//��ת������
			vector<int>CarLabel; //�ñ�ǩ���������

			//cout << "�Ѿ��ҵ��ľ��ο������" << ColorContours.size() << endl;

			for (int i = 0; i < ColorContours.size(); i++)//����ÿһ������
			{
				cv::RotatedRect rt;
				rt = minAreaRect(ColorContours[i]);  //����Ϊ�����Χ�㼯����С��ת����
				double width, height;
				width = max(rt.size.width, rt.size.height);
				height = min(rt.size.width, rt.size.height);//�õ����εĳ��ȺͿ��
				if (width * height > 300 && width * height < img.rows * img.cols / 10 && width*1.0 / height> 2.5 && width*1.0 / height < 3.6)
				{
					ColorRect.push_back(rt);
				}
			}

			//std::cout << "�Ѿ�ɸѡ�ĵ������" << ColorRect.size() << endl;
			//͸�ӱ任 
			vector<cv::Mat>ColorROI(ColorRect.size());
			for (int i = 0; i < ColorRect.size(); i++)
			{
				cv::Point2f SrcVertices[4];
				cv::Point2f Midpts;//�����м���λ��
				ColorRect[i].points(SrcVertices);
				//����-����-����-����
				for (int k = 0; k < 4; k++) {  //�Ȱ�yֵ���ϵ�������
					for (int l = k; l < 4; l++) {
						if (SrcVertices[k].y >= SrcVertices[l].y) {
							Midpts = SrcVertices[k];
							SrcVertices[k] = SrcVertices[l];
							SrcVertices[l] = Midpts;
						}
					}
				}

				//�ж����������������ǲ������
				if (pow(abs(SrcVertices[0].x - SrcVertices[1].x), 2) < pow(abs(SrcVertices[0].x - SrcVertices[2].x), 2))
				{
					Midpts = SrcVertices[1];
					SrcVertices[1] = SrcVertices[2];
					SrcVertices[2] = Midpts;
				}
				if (SrcVertices[0].x > SrcVertices[1].x)
				{
					Midpts = SrcVertices[1];
					SrcVertices[1] = SrcVertices[0];
					SrcVertices[0] = Midpts;
				}
				if (SrcVertices[2].x < SrcVertices[3].x)
				{
					Midpts = SrcVertices[2];
					SrcVertices[2] = SrcVertices[3];
					SrcVertices[3] = Midpts;
				}

				cv::Point2f DstVertices[4];
				double CarWidth = max(ColorRect[i].size.width, ColorRect[i].size.height);
				double CarHeight = min(ColorRect[i].size.width, ColorRect[i].size.height);

				DstVertices[0] = cv::Point(0, 0);
				DstVertices[1] = cv::Point(CarWidth, 0);
				DstVertices[2] = cv::Point(CarWidth, CarHeight);
				DstVertices[3] = cv::Point(0, CarHeight);

				cv::Mat H = getPerspectiveTransform(SrcVertices, DstVertices);
				ColorROI[i] = cv::Mat(cv::Size(CarWidth, CarHeight), CV_8UC3);
				warpPerspective(img, ColorROI[i], H, ColorROI[i].size(), 1);
				//cout << "�Ѿ���ת�ĳ�������Ϊ" << ColorROI.size() << endl;
			}

			//�Կ������ĳ��ƽ����ٴ���
			vector<cv::Mat>NewColorROI;
			for (int i = 0; i < ColorROI.size(); i++)
			{
				cv::Mat GrayImg;
				cvtColor(ColorROI[i], GrayImg, CV_BGR2GRAY);

				cv::imshow("�ҶȻ��ĳ���", GrayImg);


				cv::Mat TempImg;
				cv::Mat Element;
				Canny(GrayImg, TempImg, 100, 150, 3);
				//cv::imshow("can�ĳ���", TempImg);
				threshold(TempImg, TempImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
				//cv::imshow("��ֵ�ĳ���", TempImg);
				Element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ColorROI[i].cols / 6, 1));
				morphologyEx(TempImg, TempImg, cv::MORPH_CLOSE, Element, cv::Point(-1, -1));
				//cv::imshow("�Ѿ����´���ĳ���", TempImg);

				//��һ����ȡ����
				vector<vector<cv::Point>>LittleContours;
				findContours(TempImg, LittleContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

				for (int j = 0; j < LittleContours.size(); j++) {
					cv::Rect rt;
					rt = boundingRect(LittleContours[j]);
					if (rt.width > ColorROI[i].cols * 0.8 && rt.height > ColorROI[i].rows * 0.8) {
						NewColorROI.push_back(ColorROI[i]);
						CarLabel.push_back(i);
						break;
					}
				}
			}

			if (NewColorROI.size() == 0)
			{
				//std::cout << "��ɫ��ȡ����ʧ��" << endl;
				flag1 = 1;
				if(hi == 1) return;
			}

			//�Գ���ROI���������и�
			for (int num = 0; num < NewColorROI.size(); num++)
			{
				//ͳһ��С
				//int Radio = 6000 * NewColorROI[num].cols / 126;//����ɸѡ��ĸ
				int Radio = 6000 * NewColorROI[num].cols / 126;//����ɸѡ��ĸ
				cv::Mat GrayCarImg, BinCarImg;
				cvtColor(NewColorROI[num], GrayCarImg, CV_BGR2GRAY);
				//cv::imshow("�����ٴ���֮���ͼ��", GrayCarImg);
				threshold(GrayCarImg, BinCarImg, 0, 255, cv::THRESH_OTSU);

				if (hi == 1) {
					for (int kk = 0; kk < BinCarImg.rows; kk++) 
						for(int jj = 0; jj < BinCarImg.cols; jj++){
							if(BinCarImg.at<uchar>(kk, jj) == 0) BinCarImg.at<uchar>(kk, jj) = 255;
							else BinCarImg.at<uchar>(kk, jj) = 0;
						}
				}
				//cv::imshow("�����ٴ���֮���ͼ��", BinCarImg);
				lic = BinCarImg;
				//����x�����ͶӰ�г���ĸ���¶���Ĳ���
				cv::Mat RowSum;
				reduce(BinCarImg, RowSum, 1, CV_REDUCE_SUM, CV_32SC1);//�ϲ������� ��ʾ�е�����
				//imshow("s", RowSum);
				cv::Point CutNumY;//Y������и��
				int Sign = 1;
				int Lenth = 0;
				for (int j = 0; j < RowSum.rows; j++)
				{

					if (RowSum.ptr<int>(0)[0] > Radio && Sign == 0)
					{
						while (RowSum.ptr<int>(Lenth)[0] >= Radio)
						{
							Lenth++;
							if (Lenth == RowSum.rows - 1)
								break;
						}
						if (Lenth > RowSum.rows * 0.6)
						{
							CutNumY = cv::Point(0, Lenth);
							break;
						}
						else {
							Sign = 1;
							j = Lenth;
							Lenth = 0;
						}
					}

					else
					{
						Sign = 1;
					}

					if (RowSum.ptr<int>(j)[0] > Radio && Sign == 1)
					{
						Lenth++;
						if (j == RowSum.rows - 1)
						{
							if (Lenth > RowSum.rows * 0.6)
							{
								CutNumY = cv::Point(j - Lenth, j);
							}
						}
						else {
							CutNumY = cv::Point(RowSum.rows * 0.15, RowSum.rows * 0.85);
						}
					}

					else if (RowSum.ptr<int>(j)[0] <= Radio && Sign == 1) {
						if (Lenth > RowSum.rows * 0.6) {
							CutNumY = cv::Point(j - Lenth, j);
							break;
						}
						else if (j == RowSum.rows - 1) {
							CutNumY = cv::Point(RowSum.rows * 0.15, RowSum.rows * 0.85);
						}
						else {
							Lenth = 0;
						}
					}
				}

				cv::Mat NewBinCarImg;//��ĸ���±��и���ͼ��
				NewBinCarImg = BinCarImg(cv::Rect(cv::Point(0, CutNumY.x), cv::Point(BinCarImg.cols, CutNumY.y)));
				//imshow("qir", NewBinCarImg);

				//�߸��ַ��ָ�
				cv::Mat MidBinCarImg = cv::Mat::zeros(cv::Size(NewBinCarImg.cols + 4, NewBinCarImg.rows + 4), CV_8UC1);//������4�����ص�
				for (int row = 2; row < MidBinCarImg.rows - 2; row++)
				{
					for (int col = 2; col < MidBinCarImg.cols - 2; col++)
					{
						MidBinCarImg.ptr<uchar>(row)[col] = NewBinCarImg.ptr<uchar>(row - 2)[col - 2];
					}
				}
				vector<vector<cv::Point>>NumContours;
				findContours(MidBinCarImg, NumContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				//��С��Ӿ���
				vector<cv::Rect>LittleRect;
				for (int k = 0; k < NumContours.size(); k++)
				{
					LittleRect.push_back(boundingRect(NumContours[k]));
				}

				//midbin_carImg->MidBinCarImg
				//num_rect->LittleRect
				//���ο�Ĵ���
				//real_numRect->AlgRect
				vector<cv::Rect>AlgRect;
				if (LittleRect.size() < 7)
				{
					continue;
				}

				selectRect(LittleRect, AlgRect);//������ĺ�����
				if (AlgRect.size() >= 7)
				{
					//std::cout << "�ַ���ȡ�ɹ�" << endl;
				}
				else {
					//std::cout << "�ַ���ȡʧ��" << endl;
					flag2 = 1;
					return;
					//continue;
				}

				//��ֹ���߸�Խ��
				if (AlgRect[6].x + AlgRect[6].width > MidBinCarImg.cols)
				{
					AlgRect[6].width = AlgRect[6].x + AlgRect[6].width - MidBinCarImg.cols;
				}

				//���ƴ������ַ�ͼ��
				//cv::Mat NumImg[7];
				for (int i = 0; i < 7; i++)
				{
					NumImg[i] = MidBinCarImg(cv::Rect(AlgRect[i].x, AlgRect[i].y, AlgRect[i].width, AlgRect[i].height));
				}

				ostringstream CarNumber;
				string Character;

				//carnumber->CarNumber;
				//charater->Character;
				//testDetector->TestDetector
				//����
				Ptr<ml::SVM>SVM_paramsH = ml::SVM::load("..//HOG��svm.xml");

				//bin_character->BinCharacter

				cv::Mat Input;
				cv::Mat BinCharacter;
				resize(NumImg[0], BinCharacter, cv::Size(16, 32), 0, 0);
				cv::HOGDescriptor TestDetector(cv::Size(16, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);

				//testDescriptor->TestDescriptor
				vector<float>TestDescriptor;
				TestDetector.compute(BinCharacter, TestDescriptor, cv::Size(0, 0), cv::Size(0, 0));
				Input.push_back(static_cast<cv::Mat>(TestDescriptor).reshape(1, 1));

				int r = SVM_paramsH->predict(Input);//�������н���Ԥ��
				Character = Province_Show(r);
				//std::cout << "ʶ������" << Province_Show(r) << endl;
				proc = Province_Show(r);

				//imshow(proc, Input);
				//bin_num->BinNum
				//midBin_num->MidBinNum
				//������ĸʶ��
				Ptr<ml::SVM>SVM_paramsZ = ml::SVM::load("..//HOG������ĸsvm.xml");
				for (int i = 1; i < 7; i++)
				{
					cv::Mat BinNum = cv::Mat::zeros(cv::Size(16, 32), CV_8UC1);
					cv::Mat MidBinNum;
					resize(NumImg[i], BinNum, cv::Size(16, 32), 0, 0);
					cv::Mat input;
					cv::HOGDescriptor Detector(cv::Size(16, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);

					vector<float>Descriptors;
					Detector.compute(BinNum, Descriptors, cv::Size(0, 0), cv::Size(0, 0));
					input.push_back(static_cast<cv::Mat>(Descriptors).reshape(1, 1));//���л����ͼƬ���δ��룬������һ��row
					float r = SVM_paramsZ->predict(input);//�������н���Ԥ��

					//��0��D����������
					if (r == 0 || r == 'D')
					{
						if (BinNum.ptr<uchar>(0)[0] == 255 || BinNum.ptr<uchar>(31)[0] == 255)
							r = 'D';
						else
							r = 0;
					}
					if (r > 9)
					{
						//std::cout << "ʶ����" << (char)r << endl;
						sss[i - 1] = (char)r;
						CarNumber << (char)r;
					}
					else
					{
						//std::cout << "ʶ����" << r << endl;
						int k = (int)r;
						sss[i - 1] = k + '0';
						CarNumber << r;
					}

				}
				img.copyTo(out);
				//��ԭͼ����ʾ���ƺ���
				Character = Character + CarNumber.str();
				//proc = Character;
				putTextZH(out, &Character[0], cv::Point(ColorRect[CarLabel[num]].boundingRect().x, abs(ColorRect[CarLabel[num]].boundingRect().y - 30)),
					cv::Scalar(0, 0, 255), 30, "����");

				cv::Point2f pts[4];
				ColorRect[CarLabel[num]].points(pts);
				for (int j = 0; j < 4; j++)
				{
					line(out, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
				}
				CatchSuccess = 1;
			}

		}
	}

}