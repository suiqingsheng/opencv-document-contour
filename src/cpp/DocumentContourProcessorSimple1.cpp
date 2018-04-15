#include <iostream>
#include "../include/DocumentContourProcessorSimple1.h"
#include "../include/Utils.h"

using namespace std;
using namespace cv;

vector<Point> DocumentContourProcessorSimple1::findContour(Mat &src, bool showResult) {

    Mat gray;
    Utils::toGray(src, gray);
    if (showResult) { Utils::showImage(gray); }

    Mat blurred;
    int iterations = 3;
    Utils::medianBlur(gray, blurred, iterations);
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

    std::vector<std::vector<cv::Point> > contoursPoly(contoursAreaFiltered.size());
    Utils::approxPolyDP(MAX(src.rows, src.cols), contoursAreaFiltered, contoursPoly);
    if (showResult) { Utils::drawContours(src, contoursPoly); }

    std::vector<std::vector<cv::Point> > contoursSquares;
    Utils::findSquaresInContours(contoursPoly, contoursSquares);
    if (showResult) { Utils::drawContours(src, contoursSquares); }

    if (!contoursSquares.empty()) {
        return contoursSquares[0];
    } else {
        return std::vector<cv::Point>();
    }
}