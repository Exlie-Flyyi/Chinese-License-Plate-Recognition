#include "stdafx.h"
#include "lab4Dlg.h"
#include "Translation.h" 
//全局变量声明
//Mat SrcImg;
//颜色值变量
int h_lower[2] = { 100 ,15};	
int s_lower = 52;
int v_lower = 46;
int h_upper[2] = { 120 ,40};
int s_upper = 255;
int v_upper = 255;
vector<cv::RotatedRect>ColorRect;
vector<int>CarLabel; //用标签来定义矩形
Mat Translation(Mat& inputImage, int dx, int dy) {
	CV_Assert(inputImage.depth() == CV_8U);
	int rows = inputImage.rows + abs(dy);   //输出图像行数
	int cols = inputImage.cols + abs(dx);   //输出图像列数
	Mat outputImage;
	outputImage.create(rows, cols, inputImage.type()); //分配新的阵列数据 （如果需要）。

	if (inputImage.channels() == 1)    //单通道灰度图
	{
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				//计算在输入图像的位置
				int x = j - dx;
				int y = i - dy;
				if (x >= 0 && y >= 0 && x < inputImage.cols && y < inputImage.rows) {//保证在原图像像素范围内
					outputImage.at<uchar>(i, j) = inputImage.at<uchar>(y, x);
				}
			}
		}
	}

	if (inputImage.channels() == 3) {   //三通道彩图
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
Mat Resize(Mat& src, double scale) {   //比例缩放
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


Mat Expand(Mat &img) {  //膨胀
	Mat dilated_dst;
	Mat element_1 = getStructuringElement(MORPH_RECT, Size(4, 4));
	Mat element_2 = getStructuringElement(MORPH_CROSS, Size(15, 15));
	Mat element_3 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));

	dilate(img, dilated_dst, element_1);
	return dilated_dst;
}
Mat Corrosion(Mat &img) {  //腐蚀
	//自定义方法
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
	case 0:pv = "皖";
		break;
	case 1:pv = "京";
		break;
	case 2:pv = "渝";
		break;
	case 3:pv = "闽";
		break;
	case 4:pv = "甘";
		break;
	case 5:pv = "粤";
		break;
	case 6:pv = "桂";
		break;
	case 7:pv = "贵";
		break;
	case 8:pv = "琼";
		break;
	case 9:pv = "翼";
		break;
	case 10:pv = "黑";
		break;
	case 11:pv = "豫";
		break;
	case 12:pv = "鄂";
		break;
	case 13:pv = "湘";
		break;
	case 14:pv = "苏";
		break;
	case 15:pv = "赣";
		break;
	case 16:pv = "吉";
		break;
	case 17:pv = "辽";
		break;
	case 18:pv = "蒙";
		break;
	case 19:pv = "宁";
		break;
	case 20:pv = "青";
		break;
	case 21:pv = "鲁";
		break;
	case 22:pv = "晋";
		break;
	case 23:pv = "陕";
		break;
	case 24:pv = "川";
		break;
	case 25:pv = "津";
		break;
	case 26:pv = "新";
		break;
	case 27:pv = "云";
		break;
	case 28:pv = "浙";
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
	//按照面积排序(从大到小)
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

	//计算更精确的平均值
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

	//按照x坐标排列顺序
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

	//判断矩形的位置
	//sum_one->Sumof1
	//ForSrect->random
	int Sumof1 = 0; //用来计算1的个数，区分“川”字
	cv::Rect random;    //第一个或者第二个字母
	int total = 0;  //random前面的的面积总和
	int minX = rt[0].x;//左边不为零的最小值

	//开始遍历前三个，鉴别是否为“川”这样的特殊字形
	for (int h = 0; h < rt.size() - 4; h++)
	{
		if (rt[h].x <= 2 + AverWidth1 / 20)
		{
			minX = rt[h + 1].x;
			continue;
		}
		total += rt[h].area();

		//判断是否是川
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
				//？
				//第二个
				LRect[1] = random;
				//求第一个
				//水平方向
				int maxX = rt[h - 1].x + rt[h - 1].width;
				//竖直方向
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


			else//？
			{

				//第一个
				LRect[0] = random;
				//求第二个
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
	//计算识别一张图片的时间
	int CatchSuccess = 0;
	//重置图片大小为900*900	
	//cout << img.rows << " " << img.cols << endl;
	if (img.rows > 900 || img.cols > 900)
	{
		int Wide = max(img.rows, img.cols);
		resize(img, img, cv::Size(), (double)900 / Wide, (double)900 / Wide);
		//std::cout << "已经重置图片大小" << endl;
	}
	for (int hi = 0; hi < 2; hi++) {
		flag1 = 0, flag2 = 0;
		if (CatchSuccess == 0)
		{
			//cout << v_lower << endl;
			//v_lower = 130;
			//hsv颜色定位
			cv::Mat HsvImg;
			cv::Mat BinHsvImg;
			cvtColor(img, HsvImg, CV_BGR2HSV); //转成HSV图像
			medianBlur(HsvImg, HsvImg, 3);
			medianBlur(HsvImg, HsvImg, 3);
			//GaussianBlur(HsvImg, HsvImg, Size(3, 3), 0, 0);
			inRange(HsvImg, cv::Scalar(h_lower[hi], s_lower, v_lower), cv::Scalar(h_upper[hi], s_upper, v_upper), BinHsvImg);  //二值化阈值（hsv范围）操作
			//cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
			//cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
			//morphologyEx(BinHsvImg, BinHsvImg, cv::MORPH_CLOSE, kernel1); //闭运算
			//morphologyEx(BinHsvImg, BinHsvImg, cv::MORPH_OPEN, kernel2);  //开运算*/
			//闭运算
			//imshow("后的图片", BinHsvImg);
			BinHsvImg = Expand(BinHsvImg); BinHsvImg = Corrosion(BinHsvImg);
			//开运算
			BinHsvImg = Corrosion(BinHsvImg); BinHsvImg = Expand(BinHsvImg);
			//imshow("形态学后的图片", BinHsvImg);
			//imwrite("stander.png", BinHsvImg);

			//轮廓检测
			vector<vector<cv::Point>>ColorContours;//所有轮廓的点坐标，子容器里面存下
			findContours(BinHsvImg, ColorContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//检测出物体的轮廓,只检测最外围轮廓,保存拐点信息,保存物体边界上所有连续的轮廓点到contours向量内
			vector<cv::RotatedRect>ColorRect;//旋转矩形类
			vector<int>CarLabel; //用标签来定义矩形

			//cout << "已经找到的矩形框的数量" << ColorContours.size() << endl;

			for (int i = 0; i < ColorContours.size(); i++)//遍历每一个矩形
			{
				cv::RotatedRect rt;
				rt = minAreaRect(ColorContours[i]);  //作用为计算包围点集的最小旋转矩阵
				double width, height;
				width = max(rt.size.width, rt.size.height);
				height = min(rt.size.width, rt.size.height);//得到矩形的长度和宽度
				if (width * height > 300 && width * height < img.rows * img.cols / 10 && width*1.0 / height> 2.5 && width*1.0 / height < 3.6)
				{
					ColorRect.push_back(rt);
				}
			}

			//std::cout << "已经筛选的点的数量" << ColorRect.size() << endl;
			//透视变换 
			vector<cv::Mat>ColorROI(ColorRect.size());
			for (int i = 0; i < ColorRect.size(); i++)
			{
				cv::Point2f SrcVertices[4];
				cv::Point2f Midpts;//调整中间点的位置
				ColorRect[i].points(SrcVertices);
				//左上-右上-右下-左下
				for (int k = 0; k < 4; k++) {  //先按y值从上到下排列
					for (int l = k; l < 4; l++) {
						if (SrcVertices[k].y >= SrcVertices[l].y) {
							Midpts = SrcVertices[k];
							SrcVertices[k] = SrcVertices[l];
							SrcVertices[l] = Midpts;
						}
					}
				}

				//判断最上面那两个点是不是最长边
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
				//cout << "已经旋转的车牌数量为" << ColorROI.size() << endl;
			}

			//对扣下来的车牌进行再处理
			vector<cv::Mat>NewColorROI;
			for (int i = 0; i < ColorROI.size(); i++)
			{
				cv::Mat GrayImg;
				cvtColor(ColorROI[i], GrayImg, CV_BGR2GRAY);

				cv::imshow("灰度化的车牌", GrayImg);


				cv::Mat TempImg;
				cv::Mat Element;
				Canny(GrayImg, TempImg, 100, 150, 3);
				//cv::imshow("can的车牌", TempImg);
				threshold(TempImg, TempImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
				//cv::imshow("二值的车牌", TempImg);
				Element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ColorROI[i].cols / 6, 1));
				morphologyEx(TempImg, TempImg, cv::MORPH_CLOSE, Element, cv::Point(-1, -1));
				//cv::imshow("已经重新处理的车牌", TempImg);

				//进一步扣取车牌
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
				//std::cout << "颜色提取车牌失败" << endl;
				flag1 = 1;
				if(hi == 1) return;
			}

			//对车牌ROI进行上下切割
			for (int num = 0; num < NewColorROI.size(); num++)
			{
				//统一大小
				//int Radio = 6000 * NewColorROI[num].cols / 126;//用来筛选字母
				int Radio = 6000 * NewColorROI[num].cols / 126;//用来筛选字母
				cv::Mat GrayCarImg, BinCarImg;
				cvtColor(NewColorROI[num], GrayCarImg, CV_BGR2GRAY);
				//cv::imshow("车牌再处理之后的图像", GrayCarImg);
				threshold(GrayCarImg, BinCarImg, 0, 255, cv::THRESH_OTSU);

				if (hi == 1) {
					for (int kk = 0; kk < BinCarImg.rows; kk++) 
						for(int jj = 0; jj < BinCarImg.cols; jj++){
							if(BinCarImg.at<uchar>(kk, jj) == 0) BinCarImg.at<uchar>(kk, jj) = 255;
							else BinCarImg.at<uchar>(kk, jj) = 0;
						}
				}
				//cv::imshow("车牌再处理之后的图像", BinCarImg);
				lic = BinCarImg;
				//进行x方向的投影切除字母上下多余的部分
				cv::Mat RowSum;
				reduce(BinCarImg, RowSum, 1, CV_REDUCE_SUM, CV_32SC1);//合并行向量 显示行的数量
				//imshow("s", RowSum);
				cv::Point CutNumY;//Y方向的切割点
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

				cv::Mat NewBinCarImg;//字母上下被切割后的图像
				NewBinCarImg = BinCarImg(cv::Rect(cv::Point(0, CutNumY.x), cv::Point(BinCarImg.cols, CutNumY.y)));
				//imshow("qir", NewBinCarImg);

				//七个字符分割
				cv::Mat MidBinCarImg = cv::Mat::zeros(cv::Size(NewBinCarImg.cols + 4, NewBinCarImg.rows + 4), CV_8UC1);//宽增加4个像素点
				for (int row = 2; row < MidBinCarImg.rows - 2; row++)
				{
					for (int col = 2; col < MidBinCarImg.cols - 2; col++)
					{
						MidBinCarImg.ptr<uchar>(row)[col] = NewBinCarImg.ptr<uchar>(row - 2)[col - 2];
					}
				}
				vector<vector<cv::Point>>NumContours;
				findContours(MidBinCarImg, NumContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				//最小外接矩形
				vector<cv::Rect>LittleRect;
				for (int k = 0; k < NumContours.size(); k++)
				{
					LittleRect.push_back(boundingRect(NumContours[k]));
				}

				//midbin_carImg->MidBinCarImg
				//num_rect->LittleRect
				//矩形框的处理
				//real_numRect->AlgRect
				vector<cv::Rect>AlgRect;
				if (LittleRect.size() < 7)
				{
					continue;
				}

				selectRect(LittleRect, AlgRect);//在下面的函数中
				if (AlgRect.size() >= 7)
				{
					//std::cout << "字符提取成功" << endl;
				}
				else {
					//std::cout << "字符提取失败" << endl;
					flag2 = 1;
					return;
					//continue;
				}

				//防止第七个越界
				if (AlgRect[6].x + AlgRect[6].width > MidBinCarImg.cols)
				{
					AlgRect[6].width = AlgRect[6].x + AlgRect[6].width - MidBinCarImg.cols;
				}

				//车牌从左到右字符图像
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
				//汉字
				Ptr<ml::SVM>SVM_paramsH = ml::SVM::load("..//HOG字svm.xml");

				//bin_character->BinCharacter

				cv::Mat Input;
				cv::Mat BinCharacter;
				resize(NumImg[0], BinCharacter, cv::Size(16, 32), 0, 0);
				cv::HOGDescriptor TestDetector(cv::Size(16, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);

				//testDescriptor->TestDescriptor
				vector<float>TestDescriptor;
				TestDetector.compute(BinCharacter, TestDescriptor, cv::Size(0, 0), cv::Size(0, 0));
				Input.push_back(static_cast<cv::Mat>(TestDescriptor).reshape(1, 1));

				int r = SVM_paramsH->predict(Input);//对所有行进行预测
				Character = Province_Show(r);
				//std::cout << "识别结果：" << Province_Show(r) << endl;
				proc = Province_Show(r);

				//imshow(proc, Input);
				//bin_num->BinNum
				//midBin_num->MidBinNum
				//数字字母识别
				Ptr<ml::SVM>SVM_paramsZ = ml::SVM::load("..//HOG数字字母svm.xml");
				for (int i = 1; i < 7; i++)
				{
					cv::Mat BinNum = cv::Mat::zeros(cv::Size(16, 32), CV_8UC1);
					cv::Mat MidBinNum;
					resize(NumImg[i], BinNum, cv::Size(16, 32), 0, 0);
					cv::Mat input;
					cv::HOGDescriptor Detector(cv::Size(16, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);

					vector<float>Descriptors;
					Detector.compute(BinNum, Descriptors, cv::Size(0, 0), cv::Size(0, 0));
					input.push_back(static_cast<cv::Mat>(Descriptors).reshape(1, 1));//序列化后的图片依次存入，放在下一个row
					float r = SVM_paramsZ->predict(input);//对所有行进行预测

					//对0和D进行再区分
					if (r == 0 || r == 'D')
					{
						if (BinNum.ptr<uchar>(0)[0] == 255 || BinNum.ptr<uchar>(31)[0] == 255)
							r = 'D';
						else
							r = 0;
					}
					if (r > 9)
					{
						//std::cout << "识别结果" << (char)r << endl;
						sss[i - 1] = (char)r;
						CarNumber << (char)r;
					}
					else
					{
						//std::cout << "识别结果" << r << endl;
						int k = (int)r;
						sss[i - 1] = k + '0';
						CarNumber << r;
					}

				}
				img.copyTo(out);
				//在原图中显示车牌号码
				Character = Character + CarNumber.str();
				//proc = Character;
				putTextZH(out, &Character[0], cv::Point(ColorRect[CarLabel[num]].boundingRect().x, abs(ColorRect[CarLabel[num]].boundingRect().y - 30)),
					cv::Scalar(0, 0, 255), 30, "宋体");

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