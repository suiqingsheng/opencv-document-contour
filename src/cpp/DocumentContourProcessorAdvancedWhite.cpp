#include <iostream>
#include "../include/DocumentContourProcessorAdvancedWhite.h"
#include "../include/Utils.h"

using namespace std;
using namespace cv;

vector<Point> DocumentContourProcessorAdvancedWhite::findContour(Mat &src, bool showResult) {

    Mat gray;
    Utils::toGray(src, gray);
    if (showResult) { Utils::showImage(gray); }

    float value = Utils::getAverageColorValue(gray);
    std::cout << "Average pixel value " << value << "\n";

    std::vector<Vec4i> houghLines;
    if (value > 170) {

        Mat mixedChannel(src.size(), CV_8U);
        int ch[] = {0, 0};
        mixChannels(&src, 1, &mixedChannel, 1, ch, 1);
        if (showResult) { Utils::showImage(mixedChannel); }

        Mat blurred;
        int iterations = 3;
        Utils::medianBlur(mixedChannel, blurred, iterations);
        if (showResult) { Utils::showImage(blurred); }

        Mat edges;
        Utils::cannyEdgesSuperWhite(blurred, edges);
        if (showResult) { Utils::showImage(edges); }

        Mat dilated;
        Utils::dilate(edges, dilated);
        if (showResult) { Utils::showImage(dilated); }

        Mat hough;
        houghLines = Utils::searcAllHoughLinesWhiteImageP(dilated);
        if (showResult) {
            Utils::initNewMat(src.cols, src.rows, hough);
            Utils::drawHoughLines(hough, houghLines);
            Utils::showImage(hough);
        }
    } else if (value > 140) {

        Mat mixedChannel(src.size(), CV_8U);
        int ch[] = {0, 0};
        mixChannels(&src, 1, &mixedChannel, 1, ch, 1);
        if (showResult) { Utils::showImage(mixedChannel); }

        Mat blurred;
        int iterations = 3;
        Utils::medianBlur(mixedChannel, blurred, iterations);
        if (showResult) { Utils::showImage(blurred); }

        Mat edges;
        Utils::cannyEdgesWhite(blurred, edges);
        if (showResult) { Utils::showImage(edges); }

        Mat dilated;
        Utils::dilate(edges, dilated);
        if (showResult) { Utils::showImage(dilated); }

        Mat hough;
        houghLines = Utils::searcAllHoughLinesWhiteImageP(dilated);
        if (showResult) {
            Utils::initNewMat(src.cols, src.rows, hough);
            Utils::drawHoughLines(hough, houghLines);
            Utils::showImage(hough);
        }
    } else {
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

        Mat hough;
        houghLines = Utils::searchAllHoughLinesP(dilated);
        if (showResult) {
            Utils::initNewMat(src.cols, src.rows, hough);
            Utils::drawHoughLines(hough, houghLines);
            Utils::showImage(hough);
        }
    }


    std::vector<Point> intersectionsAll = findValidIntersections(houghLines, src.cols, src.rows);
    if (showResult) {
        Mat srcCopy;
        src.copyTo(srcCopy);
        Utils::showPoints(intersectionsAll, srcCopy);
    }

    vector<Point> hullPoints;
    Utils::calculateConvexHull(intersectionsAll, hullPoints);
    if (showResult) {
        Mat hullMat;
        Utils::initNewMat(src.cols, src.rows, hullMat);
        Utils::showPoints(intersectionsAll, hullMat);
    }

    vector<Point> borders = reduceCountOfPointsWithAverage(hullPoints);
    if (showResult) {
        Mat bordersMat;
        Utils::initNewMat(src.cols, src.rows, bordersMat);
        Utils::showPoints(borders, bordersMat);
    }

    vector<Point> border4Polygon = get4CornersPolygon(borders);
    if (showResult) {
        Mat bordersPolygonMat;
        Utils::initNewMat(src.cols, src.rows, bordersPolygonMat);
        Utils::showPoints(border4Polygon, bordersPolygonMat);
    }

    return border4Polygon;

};

vector<Point> DocumentContourProcessorAdvancedWhite::findValidIntersections(vector<Vec4i>& lines, int cols, int rows){
    std::vector<Point> points;
    for (int i = 0; i < lines.size(); i++){
        Vec4i line = lines[i];
        Point2f o1 (line[0], line[1]);
        Point2f p1 (line[2], line[3]);
        for (int j = i; j < lines.size(); j++){
            Vec4i lineCheck = lines[j];
            Point2f o2 (lineCheck[0], lineCheck[1]);
            Point2f p2 (lineCheck[2], lineCheck[3]);
            Point2f intersection;
            if (Utils::intersections(o1, p1, o2, p2, intersection)){
                double angle = Utils::innerAngle(p1.x, p1.y, p2.x, p2.y, intersection.x, intersection.y);
                if ((angle > 80) && (angle < 105)) {
                    if (intersection.x > 0 && intersection.y > 0 && (intersection.x < cols) && (intersection.y < rows)){
                        points.push_back(intersection);
                    }
                }
            }
        }
    }
    std::cout << "Found " << points.size() << " intersections\n";
    return points;
}

vector<Point> DocumentContourProcessorAdvancedWhite::reduceCountOfPointsWithAverage(vector<Point> points) {

    std::vector<Point> vec;
    std::copy (points.begin(), points.end(), std::back_inserter(vec));
    double distanceMin = 50;
    bool isProcessing = true;
    while(vec.size() > 4 && isProcessing){
        Point point1;
        Point point2;
        bool isBreak = false;
        int insertIndex = -1;
        for (int i = 0; i < vec.size() - 1 ; i++){
            point1 = vec[i];
            isBreak = false;
            for (int j = i + 1; j < vec.size(); j++){
                point2 = vec[j];
                double distance  = sqrt ((point1.x-point2.x)*(point1.x-point2.x) + (point1.y - point2.y)*(point1.y - point2.y));
                if (distance < distanceMin){
                    vec.erase(vec.begin() + i);
                    vec.erase(vec.begin() + j - 1);
                    isBreak = true;
                    break;
                }
            }
            if (isBreak){
                insertIndex = i;
                break;
            }
        }
        if (isBreak) {
            int x = (abs(point1.x + point2.x)) / 2;
            int y = (abs(point1.y + point2.y)) / 2;
            Point pointNew(x, y);
            vec.insert(vec.begin() + insertIndex,pointNew);
        } else {
            isProcessing = false;
        }
    }
    return vec;
}

vector<Point> DocumentContourProcessorAdvancedWhite::get4CornersPolygon(vector<Point>& points) {

    Point point1;
    Point point2;
    Point point3;
    Point point4;
    double maxArea = 0;
    std::vector<Point> result;
    if (points.size() > 4){
        for (int i = 0; i < points.size() - 3; i++){
            for (int j = i + 1; j < points.size() - 2; j++){
                for (int k = j + 1; k < points.size() - 1; k++){
                    for (int l = k + 1; l < points.size(); l++){
                        std::vector<Point> pointsArea;
                        pointsArea.clear();
                        pointsArea.push_back(points[i]);
                        pointsArea.push_back(points[j]);
                        pointsArea.push_back(points[k]);
                        pointsArea.push_back(points[l]);
                        double area = contourArea(pointsArea);
                        if (area > maxArea){
                            point1 = points[i];
                            point2 = points[j];
                            point3 = points[k];
                            point4 = points[l];
                            maxArea = area;
                        }
                    }
                }
            }
        }
        result.push_back(point1);
        result.push_back(point2);
        result.push_back(point3);
        result.push_back(point4);
    } else {
        result = points;
    }
    return result;
}





