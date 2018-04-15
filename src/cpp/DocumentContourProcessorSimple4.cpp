#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "../include/DocumentContourProcessorSimple4.h"
#include "../include/Utils.h"

using namespace cv;
using namespace std;

vector<Point> DocumentContourProcessorSimple4::findContour(Mat &image, bool showResult) {

    Mat blur;
    Utils::medianBlur(image, blur, 3);
    if (showResult) { Utils::showImage(blur); }

    Mat gray0;
    Utils::toGray(blur, gray0);
    if (showResult) { Utils::showImage(gray0); }

    Mat gray;
    vector<vector<Point> > squares;

    int thresholdLevel = 2;
    for (int l = 0; l < thresholdLevel; l++) {

        if (l == 0) {
            Utils::cannyEdges(gray0, gray);
            if (showResult) { Utils::showImage(gray); }

            Utils::dilate(gray, gray);
            if (showResult) { Utils::showImage(gray); }

        } else {
            gray = gray0 >= (l + 1) * 255 / thresholdLevel;
            if (showResult) { Utils::showImage(gray); }
        }

        vector<vector<Point> > contours;
        Utils::findContours(gray, contours);
        if (showResult) { Utils::drawContours(image, contours); }

        vector<vector<Point> > approxContours(contours.size());
        Utils::approxPolyDP(image.rows, contours, approxContours);
        if (showResult) { Utils::drawContours(image, approxContours); }

        for (size_t i = 0; i < approxContours.size(); i++) {
            vector<Point> approx = approxContours[i];
            if (approx.size() == 4 && isContourConvex(approx)) {
                double maxCosine = 0;
                for (int j = 2; j < 5; j++) {
                    double cosine = fabs(Utils::angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                    maxCosine = MAX(maxCosine, cosine);
                }
                if (maxCosine < 0.3)
                    squares.push_back(approx);
            }
        }

        if (showResult) { Utils::drawContours(image, squares); }
    }

    double largest_area = -1;
    int largest_contour_index = 0;
    for (int i = 0; i < squares.size(); i++) {
        double a = contourArea(squares[i], false);
        if (a > largest_area) {
            largest_area = a;
            largest_contour_index = i;
        }
    }
    vector<Point> points;
    if (!squares.empty()) {
        points = squares[largest_contour_index];
    }
    return points;
}