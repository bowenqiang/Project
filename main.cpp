#include "VP.h"

VP::VP(int landWidth, int houghMinLength, int numVps)
{
	m_laneWidth = landWidth;
	m_houghMinLength = houghMinLength;
	m_numVps = numVps;
}

void VP::laneMarkingsDetector(cv::Mat &srcGRAY, cv::Mat &dstGRAY, int tau)
{
	dstGRAY.setTo(0);
	int aux = 0;
	for (int j = 0; j < srcGRAY.rows; ++j)
	{
		unsigned char *ptRowSrc = srcGRAY.ptr<uchar>(j);
		unsigned char *ptRowDst = dstGRAY.ptr<uchar>(j);
		for (int i = tau; i < srcGRAY.cols - tau; ++i)
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

void VP::findVinshingPoints(cv::Mat &srcGRAY)
{
	Mat dstGRAY = Mat::zeros(srcGRAY.size(), srcGRAY.type());
	Mat srcGRAY_cp = srcGRAY.clone();
	for (int i = 0; i < srcGRAY_cp.rows / 3; i++)
	{
		unsigned char *ptRowSrc = srcGRAY_cp.ptr<uchar>(i);
		for (int j = 0; j < srcGRAY_cp.cols; j++)
		{
			ptRowSrc[j] = 0;
		}
	}
	laneMarkingsDetector(srcGRAY_cp, dstGRAY,m_laneWidth);
	Mat dstBGR;
	threshold(dstGRAY, dstBGR, 128, 255, THRESH_BINARY);
	//Canny(dstBGR, dstBGR, 0, 30, 3);
	imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\Binary.png", dstBGR);
	vector<Vec2f> lines_;
	HoughLines(dstBGR, lines_, 1, CV_PI / 180, m_houghMinLength);
	if (lines_.size() > 2)
	{
		std::vector<std::vector<cv::Point> > lineSegments;
		cvtColor(dstBGR, dstBGR, CV_GRAY2RGB);
		for (size_t i = 0; i < lines_.size(); i++)
		{
			float rho = lines_[i][0];
			float theta = lines_[i][1];
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
			Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
			clipLine(dstBGR.size(), pt1, pt2);
			lineSegments.push_back({ pt1, pt2 });
			if (!dstBGR.empty())
				line(dstBGR, pt1, pt2, Scalar(0, 0, 255), 1, 8);
		}
		//imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\HOUGHLINES.png", dstBGR);

		MSAC _msac;
		_msac.init(MODE_NIETO, srcGRAY_cp.size(), 1);
		vector<int>numInliers;
		std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;
		std::vector<cv::Mat> vps;
		_msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, m_numVps);
		_msac.drawCS(dstBGR, lineSegmentsClusters, vps);
		//imwrite("C:\\Users\\bowen\\Documents\\TCL\\LaneDetection\\VanishingPoint.png", dstBGR);
		Point2f vp(vps[0].at<float>(0, 0), vps[0].at<float>(1, 0));
		m_vp = vp;
	}
	else
	{
		Point2f vp(srcGRAY.cols/2, srcGRAY.rows/2);
		m_vp = vp;
	}
}
