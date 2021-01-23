#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <Windows.h>
#include<time.h>
#define WIDTH_INDEX 3
#define HEIGHT_INDEX 4
using namespace std;
using namespace cv;
int main() {
	//读取模板  注意更改文件路径
	Mat image_at;
	image_at = imread("C:\\Users\\user\\Desktop\\at.jpg", CV_8UC1);
	/*imshow("at", image_at);*/
	Mat image_on;
	image_on = imread("C:\\Users\\user\\Desktop\\on.jpg", CV_8UC1);
	//imshow("on", image_on);
	Mat image_on1;
	image_on1 = imread("C:\\Users\\user\\Desktop\\on1.jpg", CV_8UC1);
	//imshow("on1", image_on1);
	Mat image_on2;
	image_on2 = imread("C:\\Users\\user\\Desktop\\on2.jpg", CV_8UC1);
	//imshow("on2", image_on2);
	Mat image_on3;
	image_on3 = imread("C:\\Users\\user\\Desktop\\on3.jpg", CV_8UC1);
	//imshow("on3", image_on3);
	Mat image_on4;
	image_on4 = imread("C:\\Users\\user\\Desktop\\on4.jpg", CV_8UC1);
	//imshow("on4", image_on4);
	Mat image_on5;
	image_on5 = imread("C:\\Users\\user\\Desktop\\on5.jpg", CV_8UC1);
	//imshow("on5", image_on5);
	//打开摄像头
	VideoCapture capture(0);

	if (!capture.isOpened())
	{
		cout << "----------------------------------error to open camera--------------------------------------" << endl;
		return -1;
	}
	//获取当前摄像头的视频宽高信息
	Size S = cv::Size((int)capture.get(WIDTH_INDEX),
		(int)capture.get(HEIGHT_INDEX));
	Mat frame;
	while (true)
	{   //读取当前帧
		capture >> frame;
		//imshow("读取视频", frame);
		Mat image_RGB = frame.clone();
		flip(frame, image_RGB, 1);
		//HSV
		int h = 0;
		int s = 1;
		int v = 2;
		Mat image_HSV;
		Mat image_skinHSV = image_RGB.clone();
		image_skinHSV.setTo(0);
		cvtColor(image_RGB, image_HSV, COLOR_BGR2HSV);
		//vote
		Mat image_skin = image_RGB.clone();
	// HSV颜色空间H范围筛选法  来源 https://blog.csdn.net/Gavinmiaoc/article/details/84034183
		image_skin.setTo(0);
		for (int i = 0; i < image_RGB.rows; i++) {
			for (int j = 0; j < image_RGB.cols; j++) {
				//HSV空间判别
				uchar* p_srcHSV = image_HSV.ptr<uchar>(i, j);
				uchar* p_skin = image_skin.ptr<uchar>(i, j);
				if (p_srcHSV[h] >= 0   
					&& p_srcHSV[h] <= 20
					&& p_srcHSV[s] >= 48
					&& p_srcHSV[v] >= 50) {
					p_skin[0] = 255;
					p_skin[1] = 255;
					p_skin[2] = 255;
				}
			}
		}
		//imshow("skinRGB", image_skinRGB);
		//imshow("skinEllipse", image_skinEllipse);
		//imshow("skinHSV", image_skinHSV);
		//imshow("skin", image_skin);

		//灰度
		Mat image_grey = image_RGB.clone();
		cvtColor(image_skin, image_grey, COLOR_BGR2GRAY);
		//imshow("grey", image_grey);
		//模糊
		Mat image_blur = image_grey.clone();
		blur(image_grey, image_blur, Size(5, 5));
		/*imshow("blur", image_blur);*/

		Mat result;
		int result_cols = image_RGB.cols - image_at.cols + 1;
		int result_rows = image_RGB.rows - image_at.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_at, result, TM_CCOEFF_NORMED);
		//normalize(result, result, 0, 1, 32);
		Point minLoc;
		Point maxLoc;
		double minVal = 0;
		double maxVal = 0;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		matchLoc = maxLoc;

		Mat mask = image_RGB.clone();
		if (maxVal > 0.75) {
			rectangle(mask, matchLoc, Point(matchLoc.x + image_at.cols, matchLoc.y + image_at.rows), Scalar(0, 255, 0), 2, 8, 0);
			SetCursorPos(0.01*matchLoc.x*abs(matchLoc.x), 0.01*matchLoc.y*abs(matchLoc.y));
		}

		//on
		result_cols = image_RGB.cols - image_on.cols + 1;
		result_rows = image_RGB.rows - image_on.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_on, result, TM_CCOEFF_NORMED);


		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		//on1
		result_cols = image_RGB.cols - image_on1.cols + 1;
		result_rows = image_RGB.rows - image_on1.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_on1, result, TM_CCOEFF_NORMED);

		double maxVal1;
		minMaxLoc(result, &minVal, &maxVal1, &minLoc, &maxLoc, Mat());
		//cout << matchLoc.x << " " << matchLoc.y << " " << maxVal1 << endl;

		//on2
		result_cols = image_RGB.cols - image_on2.cols + 1;
		result_rows = image_RGB.rows - image_on2.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_on2, result, TM_CCOEFF_NORMED);

		double maxVal2;
		minMaxLoc(result, &minVal, &maxVal2, &minLoc, &maxLoc, Mat());
		//cout << matchLoc.x << " " << matchLoc.y << " " << maxVal2 << endl;

		//on3
		result_cols = image_RGB.cols - image_on3.cols + 1;
		result_rows = image_RGB.rows - image_on3.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_on3, result, TM_CCOEFF_NORMED);

		double maxVal3;
		minMaxLoc(result, &minVal, &maxVal3, &minLoc, &maxLoc, Mat());
		//cout << matchLoc.x << " " << matchLoc.y << " " << maxVal3 << endl;

		//on4
		result_cols = image_RGB.cols - image_on4.cols + 1;
		result_rows = image_RGB.rows - image_on4.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_on4, result, TM_CCOEFF_NORMED);

		double maxVal4;
		minMaxLoc(result, &minVal, &maxVal4, &minLoc, &maxLoc, Mat());
		//cout << matchLoc.x << " " << matchLoc.y << " " << maxVal4 << endl;

		//on5
		result_cols = image_RGB.cols - image_on5.cols + 1;
		result_rows = image_RGB.rows - image_on5.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(image_blur, image_on5, result, TM_CCOEFF_NORMED);

		double maxVal5;
		minMaxLoc(result, &minVal, &maxVal5, &minLoc, &maxLoc, Mat());
		/*cout << matchLoc.x << " " << matchLoc.y << " " << maxVal5 << endl;*/
	
		if (maxVal > 0.9|| maxVal1 > 0.9|| maxVal2 > 0.89||maxVal3 > 0.9 || maxVal4 > 0.9 || maxVal5 > 0.90) {
			mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);//按下左键
			mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);//松开左键		}
		}
		imshow("mask", mask);

		if (char(waitKey(1)) == 'q') break;//注意：鼠标必须激活当前窗口, 即鼠标要点一下窗口（图像），
		//不然要是放在cmd窗口，无法键入字符。
	}
	return 0;
}