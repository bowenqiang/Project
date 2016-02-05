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

int _laneWidth = 5;
int _houghMinLength = 50;
int _numVps = 1;
float _scale = 0.5;

//float _focalLengthX=309.4362;
//float _focalLengthY = 344.2161;
//float _opticalCenterX = 317.9034;
//float _opticalCenterY = 256.5352;
//float _dist = 2179.8;
//float _pitch = 14 * CV_PI / 180;
//float _yaw = 0;

float _focalLengthX = 2013.6773;
float _focalLengthY = 2004.6982;
float _opticalCenterX = 824.9676;
float _opticalCenterY = 508.4846;
float _dist = 1500;
float _pitch = -2* CV_PI / 180;
float _yaw = 0;

float _left = 0;
float _right = 1920;
float _top = 670;
float _button = 1080;


int main()
{
	IPMInfo ipmInfo;
	ipmInfo.vpPortion = 0;
	ipmInfo.ipmLeft = _left * _scale;
	ipmInfo.ipmRight = _right * _scale;
	ipmInfo.ipmTop = _top * _scale;
	ipmInfo.ipmBottom = _button * _scale;
	ipmInfo.ipmInterpolation = 0;

	CameraInfo cameraInfo;
	cameraInfo.cameraHeight = _dist*_scale;
	cameraInfo.focalLength.x = _focalLengthX*_scale;
	cameraInfo.focalLength.y = _focalLengthY*_scale;
	cameraInfo.opticalCenter.x = _opticalCenterX*_scale;
	cameraInfo.opticalCenter.y = _opticalCenterY*_scale;

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
	LineConf->ransacSplineNumSamples = 5;
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
	LineConf->ransacLine = 0;
	LineConf->ransacSpline = 1;

	ifstream fin("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\video0\\imagelist.txt");
	char str[250][256];//存300个图像路径的字符数组
	int n = 0;
	Mat imageOrigin;
	//Size dsize = Size(imageOrigin.cols*_scale,imageOrigin.rows*_scale);
	Mat imageResize;
	Mat imageGrey;
	//VP vp(_laneWidth, _houghMinLength, _numVps);
	while (fin.getline(str[n], 256) && n<250)   // 获取每个图像的路径
	{
		imageOrigin = imread(str[n], 1);
		if (!imageOrigin.data)
		{
			cout << "can't read data!" << endl;
			return -1;
		}
		resize(imageOrigin, imageResize, Size(), _scale, _scale, INTER_AREA);
		cvtColor(imageResize, imageGrey, CV_RGB2GRAY);
		Size size = imageResize.size();
		cameraInfo.imageWidth = size.width;
		cameraInfo.imageHeight = size.height;
		cameraInfo.pitch = _pitch;
		//cameraInfo.pitch = atan((_opticalCenterY - vp.m_vp.y) / _focalLengthY);
		cameraInfo.yaw = _yaw;
		//cameraInfo.yaw = atan((_opticalCenterX - vp.m_vp.x) / _focalLengthX);

		imageGrey.convertTo(imageGrey, CV_32F);
		const CvMat cvImage = imageGrey;
		const CvMat clrImage = imageResize.clone();
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
		CvMat *fipm = cvCloneMat(ipm);
		//mcvPostprocessLines(&cvImage, &clrImage, fipm, ipm, Lines, lineScores,Splines, splineScores,LineConf, state, ipmInfo, cameraInfo);

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
