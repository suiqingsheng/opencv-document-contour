#ifndef SRC_INCLUDE_DOCUMENTCONTOURPROCESSORADVANCEDWHITE_H_
#define SRC_INCLUDE_DOCUMENTCONTOURPROCESSORADVANCEDWHITE_H_


#include "opencv2/imgproc.hpp"
#include <vector>
#include <cstdio>

using namespace cv;
using namespace std;

class DocumentContourProcessorAdvancedWhite {

public:
    vector<Point> findContour(Mat &image, bool showResult);

private:

    vector<Point> findValidIntersections(vector<Vec4i>& lines, int cols, int rows);

    vector<Point> reduceCountOfPointsWithAverage(vector<Point> points);

    vector<Point> get4CornersPolygon(vector<Point>& points);
};

#endif
