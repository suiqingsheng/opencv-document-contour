
#include "../include/DocumentContourProcessorSimple3.h"
#include "../include/Utils.h"

using namespace std;
using namespace cv;

vector<Point> DocumentContourProcessorSimple3::findContour(Mat& src,  bool showResult) {

	Mat gray;
	Utils::toGray(src, gray);
	if (showResult) { Utils::showImage(gray); }

	Mat blurred;
	Utils::gaussianBlur(gray, blurred);
	if (showResult) { Utils::showImage(blurred); }

	Mat edges;
	Utils::cannyEdges(blurred, edges);
	if (showResult) { Utils::showImage(edges); }

	Mat dilated;
	Utils::dilate(edges, dilated);
	if (showResult) { Utils::showImage(dilated); }

	vector<Vec4i> lines = Utils::searchAllHoughLinesP(dilated);
	Utils::drawHoughLines(edges, lines);
	if (showResult) { Utils::showImage(edges); }

	std::vector<std::vector<cv::Point> > contours;
	Utils::findContours(edges, contours);
	if (showResult) { Utils::drawContours(src, contours); }

	std::vector<std::vector<cv::Point> > contoursArcFiltered;
	Utils::filterContoursByArcLength(src.cols, src.rows, contours, contoursArcFiltered);
	if (showResult) { Utils::drawContours(src, contoursArcFiltered); }

	std::vector<std::vector<cv::Point> > contoursAreaFiltered;
	Utils::filterContoursByArea(src.cols, src.rows, contoursArcFiltered, contoursAreaFiltered);
	if (showResult) { Utils::drawContours(src, contoursAreaFiltered); }

	std::vector<std::vector<cv::Point> > contoursApproxPoly(contoursAreaFiltered.size());
	Utils::approxPolyDP(MAX(src.rows, src.cols), contoursAreaFiltered, contoursApproxPoly);
	if (showResult) { Utils::drawContours(src, contoursApproxPoly); }

	if (!contoursApproxPoly.empty()) {
		return contoursApproxPoly[0];
	} else {
		return std::vector<cv::Point>();
	}
};

