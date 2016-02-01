#include<iostream>
#include<highgui.h>
#include<cv.h>
#include "MSAC.h"
#include "VP.h"
#include "InversePerspectiveMapping.h"
#include "LaneDetector.h"
#include "CameraInfoOpt.h"
//#include "mcv.hh"

using namespace std;
using namespace cv;
using namespace LaneDetector;
const char* _wndname = "Lane Detection Demo";
int _laneWidth = 5;
int _houghMinLength = 50;
int _numVps = 1;
int _scale = 1;

float _focalLengthX=309.4362;
float _focalLengthY = 344.2161;
float _opticalCenterX = 317.9034;
float _opticalCenterY = 256.5352;
float _dist = 2179.8;
//float _pitch = 14*CV_PI/180;
//float _yaw = 0*CV_PI/180;
//Rect _roi = Rect(56, 185, 510, 160);


int main()
{
	namedWindow(_wndname, WINDOW_AUTOSIZE);
	Mat imageOrigin,imageResize,imageGrey;
	imageOrigin = imread("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\f00002.png", 1);
	resize(imageOrigin, imageResize, Size(), _scale, _scale, INTER_AREA);
	cvtColor(imageResize, imageGrey, CV_RGB2GRAY);
	VP vp(_laneWidth, _houghMinLength, _numVps);
	vp.findVinshingPoints(imageGrey);


	//Mat transfo = findTransformationMatrix(vp.m_vp, imageResize.size());
	Size size = imageResize.size();
	CameraInfo cameraInfo;
	//mcvInitCameraInfo("C:\\Users\\bowen\\Downloads\\caltech-lane-detection-master\\caltech-lane-detection-master\\src\\CameraInfo.conf",&cameraInfo);
	//cameraInfo.pitch = _pitch;
	cameraInfo.pitch = atan((_opticalCenterY - vp.m_vp.y) / _focalLengthY);
	//cameraInfo.yaw = _yaw;
	cameraInfo.yaw = atan((_opticalCenterX - vp.m_vp.x) / _focalLengthX);
	cameraInfo.cameraHeight = _dist;
	cameraInfo.focalLength.x = _focalLengthX;
	cameraInfo.focalLength.y = _focalLengthY;
	cameraInfo.opticalCenter.x = _opticalCenterX;
	cameraInfo.opticalCenter.y = _opticalCenterY;
	cameraInfo.imageWidth = size.width;
	cameraInfo.imageHeight = size.height;
	


	IPMInfo ipmInfo;
	ipmInfo.vpPortion = 0;
	ipmInfo.ipmLeft = 56;
	ipmInfo.ipmRight =573;
	//ipmInfo.ipmTop = 185;
	ipmInfo.ipmTop = 200;
	ipmInfo.ipmBottom = 345;
	ipmInfo.ipmInterpolation = 0;

	imageGrey.convertTo(imageGrey,CV_32F);
	const CvMat cvImage=imageGrey;
	const CvMat clrImage = imageResize;
	CvMat *ipm;
	ipm = cvCreateMat(cvImage.height, cvImage.width, cvImage.type);
	mcvGetIPM(&cvImage, ipm, &ipmInfo, &cameraInfo);  //image after IPM
	Mat dst_IPM = Mat(ipm, true);
	imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\IPM.jpg", dst_IPM);
	float sigmax = 76.5 * ipmInfo.xScale;
	//float sigmax = 1;
	float sigmay = 1500*ipmInfo.yScale;
	//float sigmay = 1;
	//mcvFilterLines(ipm, ipm, 4, 4, sigmax, sigmay, LINE_HORIZONTAL);
	mcvFilterLines(ipm, ipm, 2, 2, sigmax, sigmay, LINE_VERTICAL);  //IPM image after filtering and thresholding

	Mat dst_filtered = Mat(ipm, true);
	imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\filtered.jpg", dst_filtered);

	//zero out points outside the image in IPM view
	list<CvPoint> outPixels;
	list<CvPoint>::iterator outPixelsi;
	for (outPixelsi = outPixels.begin(); outPixelsi != outPixels.end(); outPixelsi++)
		CV_MAT_ELEM(*ipm, float, (*outPixelsi).y, (*outPixelsi).x) = 0;
	outPixels.clear();

	LaneDetectorConf *LineConf = new LaneDetectorConf;
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
	Mat dst_zeroout = Mat(ipm, true);
	imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\zeroout.jpg", dst_zeroout);

	//compute quantile: .985
	float lowerQuantile = 0.975;
	FLOAT qtileThreshold = mcvGetQuantile(ipm, lowerQuantile);
	mcvThresholdLower(ipm, ipm, qtileThreshold);
	Mat image_output= Mat(ipm, true);
	imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\quantile.jpg", image_output);

	vector<Line> Lines;
	vector<float> lineScores;
	///////Hough Grouping Gonfig

	LineConf->rMin=0.05*ipm->height;
	LineConf->rMax=0.4*ipm->height;
	LineConf->rStep=1;
	LineConf->thetaMin=88*CV_PI/180;
	LineConf->thetaMax=92*CV_PI/180;
	LineConf->thetaStep = 1 * CV_PI / 180;
	LineConf->group=false;
	LineConf->groupThreshold=1;
	Mat ipm_mat = Mat(ipm, true);
	//threshold(ipm_mat, ipm_mat, 10, 255, THRESH_BINARY);
	const CvMat ipm0 = ipm_mat;
	mcvGetHoughTransformLines(&ipm0, &Lines, &lineScores, LineConf->rMin, LineConf->rMax, LineConf->rStep, LineConf->thetaMin, LineConf->thetaMax, LineConf->thetaStep, LineConf->binarize, LineConf->localMaxima, LineConf->detectionThreshold, LineConf->smoothScores, LineConf->group, LineConf->groupThreshold);
	return 0;
}
