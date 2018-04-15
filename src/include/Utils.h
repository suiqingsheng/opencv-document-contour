#ifndef SRC_INCLUDE_UTILS_H_
#define SRC_INCLUDE_UTILS_H_

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class Utils {

private:

    static RNG rng;

public:

    static void showImage(Mat &mat);

    static void toGray(Mat &src, Mat &target);

    static void calculateSobelGradient(Mat &src, Mat &out);

    static void calculateSccharGradient(Mat &src, Mat &out);

    static void calculateLaplacianGradient(Mat &src, Mat &out);

    static void equalizeHist(Mat &src, Mat &out);

    static void gaussianBlur(Mat &src, Mat &out);

    static void calculateHistogram(Mat &src);

    static void medianBlur(Mat &src, Mat &out, int iterations);

    static void cannyEdges(Mat &src, Mat &out);

    static void cannyEdgesLowThreshold(Mat &src, Mat &out);

    static void cannyEdgesWhite(Mat &src, Mat &out);

    static void dilate(Mat &src, Mat &out);

    static void showPoints(vector<Point> &points, Mat &image);

    static void calculateConvexHull(vector<Point> &points, vector<Point> &hullPoints);

    static void initNewMat(int cols, int rows, Mat& out);

    static void makeBrighter(Mat &matInputOutput, int threshold);

    static void getROI(Mat& input, Mat& output, int x, int y, int width, int height);

    static void findContours(Mat& input, std::vector<std::vector<cv::Point> > &contours);

    static vector<Point> findPreciseContoursThroughGradientPicture(Mat &gradient, vector<Point> points, vector<Point> polygonPoints);

    static void cornerHarrisDemo(Mat& prepared, Mat &out);

    static vector<Point> findPreciseContoursThroughPoints(Mat &gradient, vector<Point> points, vector<Point> polygonPoints);

    static vector<Vec4i> searchHoughLines(Mat &src, int threshold);

    static vector<Vec4i> searchAllHoughLinesP(Mat &src);

    static vector<Vec4i> searchHoughLinesP(Mat &src, int threshold, double minLineLength, double maxGap);

    static void drawHoughLines(Mat &target, std::vector<Vec4i> lines);

    static bool intersections(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);

    static double innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1);

    static void invert(Mat& src, Mat& dst);

    static void thresholdOtsu(Mat& src, Mat& dst);

    static void makeDarker(Mat &matInputOutput, int threshold);

    static void applyBlurred(Mat& blurred, Mat& dst);

    static float getAverageColorValue(Mat &src);

    static vector<Vec4i> searcAllHoughLinesWhiteImageP(Mat &src);

    static void resizeImage(Mat& src, Mat& dst, int maxDimension);

    static void cannyEdgesSuperWhite(Mat &src, Mat &out);

    static void filterContoursByArcLength(int srcWidth,
                                          int srcHeight,
                                          std::vector<std::vector<cv::Point> > &contoursInput,
                                          std::vector<std::vector<cv::Point> > &contoursOutput);

    static void filterContoursByArea(int srcWidth,
                                     int srcHeight,
                                     std::vector<std::vector<cv::Point> > &contoursInput,
                                     std::vector<std::vector<cv::Point> > &contoursOutput);

    static void drawContour(Mat &src, std::vector<cv::Point> &contours);

    static void drawContours(Mat &src, std::vector<std::vector<cv::Point> > &contours);

    static void approxPolyDP(int height, std::vector<std::vector<cv::Point> > &contoursAreaFiltered, std::vector<std::vector<cv::Point> > &contoursOutput);

    static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 );

    static void findSquaresInContours(std::vector<std::vector<cv::Point> > &contours,
                                      std::vector<std::vector<cv::Point> > &squaresOut);
};


#endif /* SRC_INCLUDE_UTILS_H_ */
