#include<iostream>
#include<highgui.h>
#include<cv.h>
#include "MSAC.h"
#include "VP.h"
#include "InversePerspectiveMapping.h"
#include "LaneDetector.h"
#include "CameraInfoOpt.h"
#include<fstream>
//#include "Main.h"
//#include "mcv.hh"
//#define DEBUG
using namespace std;
using namespace cv;
using namespace LaneDetector;

int _laneWidth = 20;
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
float _pitch = -2.5* CV_PI / 180;
float _yaw = 1 * CV_PI / 180;

float _left = 150;
float _right = 1450;
float _top = 670;
float _button = 1920;


void laneMarkingsDetector(cv::Mat &srcGRAY, cv::Mat &dstGRAY, int tau)
{
	Mat src_clone = srcGRAY.clone();
	dstGRAY.setTo(0);
	int aux = 0;
	for (int j = 0; j < srcGRAY.rows; ++j)
	{
		unsigned char *ptRowSrc = src_clone.ptr<uchar>(j);
		unsigned char *ptRowDst = dstGRAY.ptr<uchar>(j);
		for (int i = tau; i < src_clone.cols - tau; ++i)
		{
			if (ptRowSrc[i] != 0)
			{
				aux = 2 * ptRowSrc[i];
				aux += -ptRowSrc[i - tau];
				aux += -ptRowSrc[i + tau];
				aux += -abs((int)(ptRowSrc[i - tau] - ptRowSrc[i + tau]));
				aux = (aux < 0) ? (0) : (aux);
				aux = (aux > 255) ? (255) : (aux);
				ptRowDst[i] = (unsigned char)aux;
			}
		}
	}
}


bool mycmp(Line line1,Line line2)
{
	return line1.startPoint.x < line2.startPoint.x;
}


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
	LineConf->ransacLineNumSamples =2 ;
	LineConf->ransacLineThreshold =0.2;
	LineConf->ransacLineScoreThreshold=10;
	LineConf->checkLaneWidthMean = 250*_scale;
	LineConf->checkLaneWidthStd = 50*_scale;
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
	LineConf->ipmWindowClear = 1;
	LineConf->ipmWindowLeft=350;
	LineConf->ipmWindowRight=550;

	ifstream fin("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\video0\\imagelist.txt");
	char str[256];//存300个图像路径的字符数组
	int n = 0;

	//kalmanfilter
	KalmanFilter KF(4, 4, 0);
	KF.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	Mat_<float> measurement(4, 1);
	measurement.setTo(Scalar(0));
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	//VP vp(_laneWidth, _houghMinLength, _numVps);
	int frameidx = 1;
	vector<Line> linesPre;
	LineState *state = new LineState;
	vector<Line> stateLine;
	while (fin.getline(str, 256))   // 获取每个图像的路径
	{
		Mat imageOrigin = imread(str, 1);
		if (!imageOrigin.data)
		{
			cout << "can't read data!" << endl;
			return -1;
		}
		Mat imageResize;
		Mat imageGrey;
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
		const CvMat clrImage = imageResize;
		CvMat *ipm;
		ipm = cvCreateMat(cvImage.height, cvImage.width, cvImage.type);
		mcvGetIPM(&cvImage, ipm, &ipmInfo, &cameraInfo);  //image after IPM 
		SHOW_IMAGE(ipm,"ipm",1);
		CvMat *ipm_clone = cvCloneMat(ipm);
		Mat ipm_mat = Mat(ipm, true);
		ipm_mat.convertTo(ipm_mat, CV_8U);
		laneMarkingsDetector(ipm_mat, ipm_mat, _laneWidth*_scale);
		threshold(ipm_mat, ipm_mat, 50,255,THRESH_BINARY);
		morphologyEx(ipm_mat, ipm_mat, MORPH_OPEN, Mat(3,3, CV_8U),Point(-1,-1),1);
		ipm_mat.convertTo(ipm_mat, CV_32F);
		ipm = &CvMat(ipm_mat);

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

		vector<float> lineScores, splineScores;
		vector<Spline>Splines;
		vector<Line> Lines;
		LineConf->rMin = 0.5*ipm->height;
		LineConf->rMax = 1*ipm->height;
		//detect the lines;
		mcvGetHoughTransformLines(ipm, &Lines, &lineScores, LineConf->rMin, LineConf->rMax, LineConf->rStep, LineConf->thetaMin, LineConf->thetaMax, LineConf->thetaStep, LineConf->binarize, LineConf->localMaxima, LineConf->detectionThreshold, LineConf->smoothScores, LineConf->group, LineConf->groupThreshold);		
		mcvCheckLaneWidth(Lines, lineScores,LineConf->checkLaneWidthMean,LineConf->checkLaneWidthStd);

		sort(Lines.begin(), Lines.end(), mycmp);
		//mcvGetRansacLines(ipm, Lines, lineScores, LineConf, LINE_VERTICAL);
		//add kalman filter here:

		if (linesPre.size()<2)////////
		{
			if (Lines.size() == 2)
			{
				linesPre = Lines;
				KF.statePre.at<float>(0) = linesPre[0].startPoint.x;
				KF.statePre.at<float>(1) = linesPre[0].endPoint.x;
				KF.statePre.at<float>(2) = linesPre[1].startPoint.x;
				KF.statePre.at<float>(3) = linesPre[1].endPoint.x;
				stateLine = linesPre;
			}
		}
		else
		{
			if (Lines.empty())
				Lines=linesPre;
			if (Lines.size() == 1)
			{
				if (fabs(Lines[0].startPoint.x - linesPre[0].startPoint.x) > (LineConf->checkLaneWidthMean / 4))
				{
					Lines.push_back(linesPre[0]);
					sort(Lines.begin(), Lines.end(), mycmp);
				}
				else
				{
					Lines.push_back(linesPre[1]);
				}

			}
			Mat prediction = KF.predict();
			//Point predictPt(prediction.at<float>(0), prediction.at<float>(1));
			measurement(0) = Lines[0].startPoint.x;
			measurement(1) = Lines[0].endPoint.x;
			measurement(2) = Lines[1].startPoint.x;
			measurement(3) = Lines[1].endPoint.x;
			Mat estimated = KF.correct(measurement);
			stateLine[0].startPoint.x = estimated.at<float>(0);
			stateLine[0].startPoint.y = 0;
			stateLine[0].endPoint.x = estimated.at<float>(1);
			stateLine[0].endPoint.y = 540;
			stateLine[1].startPoint.x = estimated.at<float>(2);
			stateLine[1].startPoint.y = 0;
			stateLine[1].endPoint.x = estimated.at<float>(3);
			stateLine[1].endPoint.y = 540;
			linesPre = Lines;
		}



		//mcvGetRansacSplines(ipm, Lines, lineScores, LineConf, LINE_VERTICAL, Splines, splineScores, state);
		//mcvGetSplinesBoundingBoxes(splines, lineType,cvSize(image->width, image->height),state->ipmBoxes);

		//mcvPostprocessLines(&cvImage, &clrImage, fipm, ipm, Lines, lineScores,Splines, splineScores,LineConf, state, ipmInfo, cameraInfo);


		CvMat imDisplay = imageResize;
		CvSize inSize = cvSize(imDisplay.width - 1, imDisplay.height - 1);
		//vector<Spline> Splines_ipm = Splines;
		vector<Line> Lines_ipm = Lines;
		mcvLinesImIPM2Im(Lines, ipmInfo, cameraInfo, inSize);
		//mcvSplinesImIPM2Im(Splines, ipmInfo, cameraInfo, inSize);
		CvMat *fipm = cvCloneMat(ipm);
		vector<Line> stateLine_ipm = stateLine;
		mcvLinesImIPM2Im(stateLine, ipmInfo, cameraInfo, inSize);
		//mcvLinesImIPM2Im(stateLine, ipmInfo, cameraInfo, inSize);

		//for (int i = 0; i < Splines.size(); i++)
		//{
		//	mcvDrawSpline(ipm_clone, Splines_ipm[i], CV_RGB(255, 0, 0), 1);
		//	mcvDrawSpline(&imDisplay, Splines[i], CV_RGB(0, 255, 0), 3);
		//}
		mcvDrawLine(&imDisplay, stateLine[0], CV_RGB(255, 0, 0), 3);
		mcvDrawLine(&imDisplay, stateLine[1], CV_RGB(0, 0, 255), 3);
		for (int i = 0; i < Lines.size(); i++)
		{
			mcvDrawLine(ipm_clone, Lines_ipm[i], CV_RGB(255, 255, 0), 1);
			mcvDrawLine(&imDisplay, Lines[i], CV_RGB(0, 255, 255), 3);
		}
		SHOW_IMAGE(ipm_clone, "Detected Lanes_IPM", 1);
		SHOW_IMAGE(&imDisplay, "Detected Lanes", 1);

		frameidx++;

		waitKey(1);
		//release memory


	}



	delete state;
	delete LineConf;
	return 0;
}
