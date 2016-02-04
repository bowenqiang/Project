#include<iostream>
#include<highgui.h>
#include<cv.h>
#include "MSAC.h"
#include "VP.h"
#include "InversePerspectiveMapping.h"
#include "LaneDetector.h"
#include "CameraInfoOpt.h"
#include<fstream>
//#include "mcv.hh"
//#define DEBUG
using namespace std;
using namespace cv;
using namespace LaneDetector;
const char* _wndname0 = "Lane Detection Demo";
const char* _wndname1 = "IPM";
const char* _wndname2 = "Lane Detection Demo";
int _laneWidth = 5;
int _houghMinLength = 50;
int _numVps = 1;
int _scale = 1;

float _focalLengthX=309.4362;
float _focalLengthY = 344.2161;
float _opticalCenterX = 317.9034;
float _opticalCenterY = 256.5352;
float _dist = 2179.8;
float _pitch = 14 * CV_PI / 180;
float _yaw = 0;





void process(Mat imageOrigin,IPMInfo &ipmInfo, CameraInfo &cameraInfo, LaneDetectorConf * LineConf)
{


}

int main()
{
	//namedWindow(_wndname0, WINDOW_AUTOSIZE);
	//namedWindow(_wndname1, WINDOW_AUTOSIZE);
	IPMInfo ipmInfo;
	ipmInfo.vpPortion = 0;
	ipmInfo.ipmLeft = 56;
	ipmInfo.ipmRight = 573;
	//ipmInfo.ipmTop = 185;
	ipmInfo.ipmTop = 220;
	ipmInfo.ipmBottom = 345;
	ipmInfo.ipmInterpolation = 0;

	CameraInfo cameraInfo;
	//cameraInfo.pitch = _pitch;

	cameraInfo.cameraHeight = _dist;
	cameraInfo.focalLength.x = _focalLengthX;
	cameraInfo.focalLength.y = _focalLengthY;
	cameraInfo.opticalCenter.x = _opticalCenterX;
	cameraInfo.opticalCenter.y = _opticalCenterY;

	LaneDetectorConf *LineConf = new LaneDetectorConf;
	LineConf->rStep = 1;
	LineConf->thetaMin = -5 * CV_PI / 180;
	LineConf->thetaMax = 5 * CV_PI / 180;
	LineConf->thetaStep = 1 * CV_PI / 180;
	LineConf->group = true;
	LineConf->groupThreshold = 50;
	LineConf->overlapThreshold = 0.3;
	LineConf->ransacLineWindow = 15;
	LineConf->ransacLineNumSamples =4 ;
	LineConf->ransacLineThreshold =0.2;
	LineConf->ransacLineScoreThreshold=10;
	LineConf->checkLaneWidthMean = 25;
	LineConf->checkLaneWidthStd = 5;
	LineConf->ransacLineBinarize=1;
	LineConf->ransacSplineDegree;
	LineConf->ransacSplineWindow=15;
	LineConf->ransacSplineStep=0.1;
	LineConf->ransacLineNumGoodFit = 5;

	LineConf->ransacSplineNumSamples = 4;
	LineConf->ransacSplineNumIterations = 40;
	LineConf->ransacSplineThreshold = 0.2;
	LineConf->ransacSplineScoreThreshold = 1;
	LineConf->ransacSplineNumGoodFit = 10;
	LineConf->ransacSplineDegree = 3;
	LineConf->splineScoreJitter = 2;
	LineConf->splineScoreLengthRatio=1.5;
	LineConf->splineScoreAngleRatio = 1.2;
	LineConf->splineScoreStep = 0.01;
	LineConf->ransacSplineBinarize = 1;
	LineConf->ransacSplineWindow = 10;
	LineConf->ransacLine = 1;


	//LineConf->getEndPoints;




	ifstream fin("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\cordova1\\imagelist.txt");
	char str[250][256];//存300个图像路径的字符数组
	int n = 0;
	Mat imageOrigin;
	Mat imageResize, imageGrey;
	VP vp(_laneWidth, _houghMinLength, _numVps);
	while (fin.getline(str[n], 256) && n<250)   // 获取每个图像的路径
	{
		imageOrigin = imread(str[n], 1);
		//imshow(_wndname0, imageOrigin);

		resize(imageOrigin, imageResize, Size(), _scale, _scale, INTER_AREA);
		cvtColor(imageResize, imageGrey, CV_RGB2GRAY);
		//vp.findVinshingPoints(imageGrey);

		Size size = imageResize.size();
		cameraInfo.imageWidth = size.width;
		cameraInfo.imageHeight = size.height;
		cameraInfo.pitch = _pitch;
		//cameraInfo.pitch = atan((_opticalCenterY - vp.m_vp.y) / _focalLengthY);
		cameraInfo.yaw = _yaw;
		//cameraInfo.yaw = atan((_opticalCenterX - vp.m_vp.x) / _focalLengthX);

		imageGrey.convertTo(imageGrey, CV_32F);
		const CvMat cvImage = imageGrey;
		const CvMat clrImage = imageResize;
		CvMat *ipm;
		ipm = cvCreateMat(cvImage.height, cvImage.width, cvImage.type);
		mcvGetIPM(&cvImage, ipm, &ipmInfo, &cameraInfo);  //image after IPM
		CvMat *ipm_clone = cvCloneMat(ipm);

		//SHOW_IMAGE(ipm, "IPM", 10);
		float sigmax = 76.5 * ipmInfo.xScale;
		//float sigmax = 1;
		float sigmay = 1500 * ipmInfo.yScale;
		//float sigmay = 1;
		mcvFilterLines(ipm, ipm, 4, 4, sigmax, sigmay, LINE_HORIZONTAL);
		mcvFilterLines(ipm, ipm, 2, 2, sigmax, sigmay, LINE_VERTICAL);  //IPM image after filtering and thresholding
		//SHOW_IMAGE(ipm, "IPM after Filtering", 10);

		//zero out points outside the image in IPM view
		list<CvPoint> outPixels;
		list<CvPoint>::iterator outPixelsi;
		for (outPixelsi = outPixels.begin(); outPixelsi != outPixels.end(); outPixelsi++)
			CV_MAT_ELEM(*ipm, float, (*outPixelsi).y, (*outPixelsi).x) = 0;
		outPixels.clear();


		if (LineConf->ipmWindowClear)
		{
			//check to blank out other periferi of the image
			//blank from 60->100 (width 40)
			CvRect mask = cvRect(LineConf->ipmWindowLeft, 0,
				LineConf->ipmWindowRight -
				LineConf->ipmWindowLeft + 1,
				ipm->height);
			mcvSetMat(ipm, mask, 0);
		}

		//zero out negative values
		mcvThresholdLower(ipm, ipm, 0);
		//compute quantile: .985
		float lowerQuantile = 0.975;
		FLOAT qtileThreshold = mcvGetQuantile(ipm, lowerQuantile);
		mcvThresholdLower(ipm, ipm, qtileThreshold);
		//SHOW_IMAGE(ipm, "Before Hough", 1);


		////morphologyex   
		//CvMat *ipm_temp;
		//cvThreshold(ipm, ipm, 0, 255, CV_THRESH_BINARY); //0.05
		//Mat ipm_mat0 = Mat(ipm, true);
		//imshow("binary", ipm_mat0);
		//dilate(ipm_mat0, ipm_mat0, Mat());
		//morphologyEx(ipm_mat0, ipm_mat0, MORPH_CLOSE, Mat(5, 5, CV_32F));
		//morphologyEx(ipm_mat0, ipm_mat0, MORPH_OPEN, Mat(3, 3, CV_32F));
		//imshow("after morphologyex", ipm_mat0);



		vector<float> lineScores, splineScores;
		vector<Spline>Splines;
		vector<Line> Lines;
		///////////Hough Grouping Gonfig
		LineConf->rMin = 0.2*ipm->height;
		LineConf->rMax = 0.8*ipm->height;
		mcvGetHoughTransformLines(ipm, &Lines, &lineScores, LineConf->rMin, LineConf->rMax, LineConf->rStep, LineConf->thetaMin, LineConf->thetaMax, LineConf->thetaStep, LineConf->binarize, LineConf->localMaxima, LineConf->detectionThreshold, LineConf->smoothScores, LineConf->group, LineConf->groupThreshold);
		//mcvCheckLaneWidth(Lines, lineScores,LineConf->checkLaneWidthMean,LineConf->checkLaneWidthStd);
		//mcvGetRansacLines(ipm,Lines,lineScores,LineConf,LINE_VERTICAL);
		LineState *state = new LineState;
		mcvGetRansacSplines(ipm,Lines,lineScores,LineConf,LINE_VERTICAL,Splines,splineScores,state);
		CvMat imDisplay = imageResize;
		CvSize inSize = cvSize(imDisplay.width - 1, imDisplay.height - 1);
		vector<Spline> Splines_ipm = Splines;
		mcvSplinesImIPM2Im(Splines,ipmInfo,cameraInfo,inSize);

		for (int i = 0; i < Splines.size(); i++)
		{
			mcvDrawSpline(ipm_clone, Splines_ipm[i], CV_RGB(255, 0, 0), 1);
			mcvDrawSpline(&imDisplay, Splines[i], CV_RGB(0, 255, 0), 3);
		}
		SHOW_IMAGE(ipm_clone,"Detected Lanes_IPM",1);
		SHOW_IMAGE(&imDisplay, "Detected Lanes", 1);


		waitKey(1);
	}




	return 0;
}
