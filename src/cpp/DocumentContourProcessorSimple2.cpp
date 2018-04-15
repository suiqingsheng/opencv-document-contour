#include "../include/DocumentContourProcessorSimple2.h"
#include "../include/Utils.h"

using namespace std;
using namespace cv;

vector<Point> DocumentContourProcessorSimple2::findContour(Mat& src, bool showResult) {

	Mat blurred;
	int iterations = 5;
	Utils::medianBlur(src, blurred, iterations);
	if (showResult) { Utils::showImage(blurred); }

	Mat gray;
	Utils::toGray(blurred, gray);
	if (showResult) { Utils::showImage(gray); }

    Mat edges;
    Utils::cannyEdges(gray, edges);
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
