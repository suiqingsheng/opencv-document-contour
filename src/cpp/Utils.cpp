#include <iostream>
#include "../include/DocumentContourProcessorAdvanced.h"
#include "../include/Utils.h"

RNG Utils::rng = RNG(12345);

void Utils::showImage(Mat &mat) {
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", mat);
    waitKey(0);
}

void Utils::toGray(Mat &src, Mat &target) {
    cv::cvtColor(src, target, CV_BGR2GRAY);
}

void Utils::calculateSobelGradient(Mat &src, Mat &out) {
    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int kernelSize = 3;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel(src, grad_x, ddepth, 1, 0, kernelSize, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    /// Gradient Y
    Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out);
}

void Utils::calculateSccharGradient(Mat &src, Mat &out) {

    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X, kernel size = 3 by default
    Scharr(src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    /// Gradient X, kernel size = 3 by default
    Scharr(src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out);
}

void Utils::calculateLaplacianGradient(Mat &src, Mat &out) {

    Mat dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Apply Laplace function
    Mat abs_dst;

    Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(dst, out);
}

void Utils::equalizeHist(Mat &src, Mat &out) {
    cv::equalizeHist(src, out);
}

void Utils::gaussianBlur(Mat &src, Mat &out) {
    int kernel = 3;
    GaussianBlur(src, out, Size(kernel, kernel), 0, 0, BORDER_DEFAULT);
}

void Utils::calculateHistogram(Mat &src) {

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Scalar red(255, 0, 0);
    Scalar green(0, 255, 0);
    Scalar blue(0, 0, 255);
    Scalar white(255, 255, 255);

    // Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    int bin_w = cvRound((double) hist_w / histSize);

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    if (bgr_planes.size() > 1) {

        Mat b_hist, g_hist, r_hist;

        /// Compute the histograms:
        cv::calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

        /// Draw for each channel
        for (int i = 1; i < histSize; i++) {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                 red, 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                 green, 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                 blue, 2, 8, 0);
        }
    } else {
        Mat w_hist;

        /// Compute the histograms:
        cv::calcHist(&bgr_planes[0], 1, 0, Mat(), w_hist, 1, &histSize, &histRange, uniform, accumulate);

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(w_hist, w_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

        /// Draw for each channel
        for (int i = 1; i < histSize; i++) {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(w_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(w_hist.at<float>(i))),
                 red, 2, 8, 0);
        }
    }

    Utils::showImage(histImage);
}

void Utils::medianBlur(Mat &src, Mat &out, int iterations) {

    for (int i = 0; i < iterations; i++) {
        cv::medianBlur(src, out, 5);
        cv::medianBlur(out, out, 5);
        cv::medianBlur(out, out, 5);
    }
}

void Utils::cannyEdgesLowThreshold(Mat &src, Mat &out) {
    int thresholdLow = 10;
    int thresholdHigh = 30;
    cv::Canny(src, out, thresholdLow, thresholdHigh);
}

void Utils::cannyEdges(Mat &src, Mat &out) {
    int thresholdLow = 50;
    int thresholdHigh = 160;
    cv::Canny(src, out, thresholdLow, thresholdHigh);
}

void Utils::cannyEdgesWhite(Mat &src, Mat &out) {
    int thresholdLow = 15;
    int thresholdHigh = 45;
    cv::Canny(src, out, thresholdLow, thresholdHigh);
}

void Utils::cannyEdgesSuperWhite(Mat &src, Mat &out) {
    int thresholdLow = 5;
    int thresholdHigh = 15;
    cv::Canny(src, out, thresholdLow, thresholdHigh);
}

void Utils::dilate(Mat &src, Mat &out) {
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(2,2));
    cv::dilate(src, out, kernel);
}

void Utils::showPoints(vector<Point> &points, Mat &image) {
    for (int i = 0; i < points.size(); i++) {
        circle(image,
               points[i],
               1,
               Scalar(Utils::rng.uniform(20, 255), Utils::rng.uniform(40, 255), Utils::rng.uniform(60, 255)),
               2,
               8);
    }
    Utils::showImage(image);
}

void Utils::calculateConvexHull(vector<Point> &points, vector<Point> &hullPoints) {
    bool pointsReturn = false;
    convexHull(Mat(points), hullPoints, pointsReturn);
}

void Utils::initNewMat(int cols, int rows, Mat &out) {
    out = Mat::zeros(Size(cols, rows), CV_8UC1);
}

void Utils::makeBrighter(Mat &matInputOutput, int threshold) {
    // accept only char type matrices
    CV_Assert(matInputOutput.depth() == CV_8UC1);
    float average = getAverageColorValue(matInputOutput);
    const int channels = matInputOutput.channels();
    MatIterator_<uchar> it, end;
    for (it = matInputOutput.begin<uchar>(), end = matInputOutput.end<uchar>(); it != end; it++) {
        if (*it < threshold) {
            *it =  average;
        }
    }
}

float Utils::getAverageColorValue(Mat &src) {

    // accept only char type matrices
    CV_Assert(src.depth() == CV_8UC1);
    const int channels = src.channels();
    MatIterator_<uchar> it, end;
    int i = 0;
    float sum = 0;
    for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++) {
        sum = sum + *it;
        i++;
    }
    return sum/i;
}

void Utils::makeDarker(Mat &matInputOutput, int threshold) {
    // accept only char type matrices
    CV_Assert(matInputOutput.depth() == CV_8UC1);
    const int channels = matInputOutput.channels();
    MatIterator_<uchar> it, end;
    for (it = matInputOutput.begin<uchar>(), end = matInputOutput.end<uchar>(); it != end; it++) {
        if (*it > threshold) {
            *it = threshold / 2;
        }
    }
}

void Utils::getROI(Mat &input, Mat &output, int x, int y, int width, int height) {
    cv::Rect roi(60, 200, 470, 590);
    output = input(roi);
}

void Utils::findContours(Mat &input, std::vector<std::vector<cv::Point> > &contours) {
    cv::findContours(input, contours, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS);
}

vector<Point> Utils::findPreciseContoursThroughGradientPicture(Mat &gradient, vector<Point> points,
                                                               vector<Point> polygonPoints) {
    std::vector<Point> result;
    int threshold = 20;
    for (int i = 0; i < polygonPoints.size(); i++) {
        int maxGradient = 0;
        Point pointToAdd;
        int x1 = polygonPoints[i].x - threshold;
        int x2 = polygonPoints[i].x + threshold;
        int y1 = polygonPoints[i].y - threshold;
        int y2 = polygonPoints[i].y + threshold;
        for (int j = x1; j < x2; j++) {
            for (int k = y1; k < y2; k++) {
                int gradientVal = gradient.at<uchar>(j, k);
                if (gradientVal > maxGradient) {
                    maxGradient = gradientVal;
                    Point pointNew(j, k);
                    pointToAdd = pointNew;
                }
            }
        }
        result.push_back(pointToAdd);
    }
    return result;
}

void Utils::cornerHarrisDemo(Mat &prepared, Mat &out) {

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros(prepared.size(), CV_32FC1);

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 250;

    /// Detecting corners
    cornerHarris(prepared, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    /// Normalizing
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    /// Drawing a circle around corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int) dst_norm.at<float>(j, i) > thresh) {
                circle(out, Point(i, j), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
}

vector<Point> Utils::findPreciseContoursThroughPoints(Mat &gradient, vector<Point> points,
                                                      vector<Point> polygonPoints) {
    std::vector<Point> result;
    int threshold = 20;
    for (int i = 0; i < polygonPoints.size(); i++) {
        int maxGradient = 0;
        Point pointToAdd;
        int x1 = polygonPoints[i].x - threshold;
        int x2 = polygonPoints[i].x + threshold;
        int y1 = polygonPoints[i].y - threshold;
        int y2 = polygonPoints[i].y + threshold;
        for (int j = 0; j < points.size(); j++) {
            Point pointCheck = points[j];
            if (pointCheck.x >= x1 && pointCheck.x <= x2 && pointCheck.y >= y1 && pointCheck.y <= y2) {
                int gradientVal = gradient.at<uchar>(pointCheck.x, pointCheck.y);
                if (gradientVal > maxGradient) {
                    maxGradient = gradientVal;
                    pointToAdd = pointCheck;
                }
            }
        }
        result.push_back(pointToAdd);
    }
    return result;
}


vector<Vec4i> Utils::searchHoughLines(Mat &src, int threshold)
{
    vector<Vec2f> lines;
    lines.clear();
    HoughLines(src, lines, 1, CV_PI / 180, threshold);
    std::cout << "Find " << lines.size() << " lines with HoughLines() (not P!)\n";
    std::vector<Vec4i> linesCartesian;
    for( size_t i = 0; i < lines.size(); i++ )
    {

        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

        Vec4i vec4i(pt1.x, pt1.y, pt2.x, pt2.y);
        linesCartesian.push_back(vec4i);
    }
    return linesCartesian;
}

vector<Vec4i> Utils::searchHoughLinesP(Mat &src, int threshold, double minLineLength, double maxGap)
{
    std::vector<Vec4i> linesCartesian;
    linesCartesian.clear();
    HoughLinesP(src, linesCartesian, 1, CV_PI / 180, threshold, minLineLength, maxGap);
    return linesCartesian;
}

void Utils::drawHoughLines(Mat& target, std::vector<Vec4i> lines){

    auto it = lines.begin();
    for (; it != lines.end(); ++it) {
        cv::Vec4i l = *it;
        cv::line(target, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 2, 8);
    }
}

vector<Vec4i> Utils::searchAllHoughLinesP(Mat &src){
    std::vector<Vec4i> houghLinesPHeight = searchHoughLinesP(src, 50, (float)src.rows * 0.70, (float)src.rows * 0.85); // 100, 600, 700
    std::vector<Vec4i> houghLinesPWidth = searchHoughLinesP(src, 40, (float)src.cols * 0.30  , (float)src.cols * 0.33); // 300, 200, 200
    std::copy (houghLinesPHeight.begin(), houghLinesPHeight.end(), std::back_inserter(houghLinesPWidth));
    return houghLinesPWidth;
}

vector<Vec4i> Utils::searcAllHoughLinesWhiteImageP(Mat &src){
    std::vector<Vec4i> houghLinesPHeight = searchHoughLinesP(src, 100, (float) src.rows * 0.6, (float) src.rows * 0.5); // 500, 200
    std::vector<Vec4i> houghLinesPWidth = searchHoughLinesP(src, 50, (float) src.cols * 0.4, (float ) src.rows*0.4); // 300, 200
    std::copy (houghLinesPHeight.begin(), houghLinesPHeight.end(), std::back_inserter(houghLinesPWidth));
    return houghLinesPWidth;
}

bool Utils::intersections(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {

    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;
    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < /*EPS*/1e-8) {
        return false;
    } else {
        double t1 = (x.x * d2.y - x.y * d2.x) / cross;
        r = o1 + d1 * t1;
        return true;
    }
}

double Utils::innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{
    double angle1 = atan2(py1-cy1, px1-cx1);
    double angle2 = atan2(py2-cy1, px2-cx1);
    double result = (angle2-angle1) * 180 / 3.14;
    if (result<0) {
        result+=360;
    }
    return result;
}

void Utils::invert(Mat& src, Mat& dst){
    bitwise_not(src, dst);
}

void Utils::thresholdOtsu(Mat& src, Mat& dst) {
    cv::threshold(src, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}

void Utils::applyBlurred(Mat& blurred, Mat& dst){
    MatIterator_<uchar> it, end, endRes;
    endRes = blurred.end<uchar>();
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++) {
        if (*it < 150) {
            *it = *endRes;
        }
        endRes++;
    }
}

void Utils::resizeImage(Mat& src, Mat& dst, int maxDimension) {

    //Do first resize very big
    Mat toResize = src;
    int width = src.cols;
    int height = src.rows;
    int max = 0;
    if (height > max || width > maxDimension) {
        do {
            width = width * 0.5;
            height = height * 0.5;
            Size size (width, height);
            resize(toResize,dst,size);
            Utils::showImage(dst);
            toResize = dst;
            max = width;
            if (height > max) {
                max = height;
            }
        } while (max > maxDimension);
    } else {
        dst = src;
    }
}

void Utils::filterContoursByArcLength(int srcWidth, int srcHeight, std::vector<std::vector<cv::Point> > &contoursInput,
                                      std::vector<std::vector<cv::Point> > &contoursOutput) {
    double arcMinLength = 2 * (srcWidth + srcHeight) * 0.4;
    for (int i = 0; i < contoursInput.size(); i++) {
        if (cv::arcLength(contoursInput[i], false) > arcMinLength)
            contoursOutput.push_back(contoursInput[i]);
    }
}

void Utils::filterContoursByArea(int srcWidth, int srcHeight, std::vector<std::vector<cv::Point> > &contoursInput, std::vector<std::vector<cv::Point> > &contoursOutput) {
    double areaMin = (srcWidth + srcHeight) * 0.4;
    for (int i = 0; i < contoursInput.size(); i++) {
        if (cv::contourArea(contoursInput[i]) > areaMin) {
            contoursOutput.push_back(contoursInput[i]);
        }
    }
}


void Utils::approxPolyDP(int height, std::vector<std::vector<cv::Point> > &contoursAreaFiltered, std::vector<std::vector<cv::Point> > &contoursOutput) {
    for (int i = 0; i < contoursAreaFiltered.size(); i++) {
        cv::approxPolyDP(Mat(contoursAreaFiltered[i]), contoursOutput[i], height * 0.1, true);
    }
}

void Utils::findSquaresInContours(std::vector<std::vector<cv::Point> > &contours, std::vector<std::vector<cv::Point> > &squaresOut) {

    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() == 4 ) {
            double maxCosine = 0;
            for (int j = 2; j < 5; j++) {
                double cosine = fabs(Utils::angle(contours[i][j % 4], contours[i][j - 2], contours[i][j - 1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            if (maxCosine < 1) {
                squaresOut.push_back(contours[i]);
            }
        }
    }
}

double Utils::angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 ) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void Utils::drawContour(Mat &src, std::vector<cv::Point> &contour) {
    Mat result(src);
    Utils::showPoints(contour, result);
    std::vector<std::vector<cv::Point> > contours;
    contours.push_back(contour);
    cv::drawContours(result, contours, 0, cv::Scalar(0, 255, 0), 1);
    Utils::showImage(result);
};

void Utils::drawContours(Mat &src, std::vector<std::vector<cv::Point> > &contours) {
    Mat result;
    src.copyTo(result);
    cv::drawContours(result, contours, -1, cv::Scalar(0, 255, 0), 1);
    Utils::showImage(result);
};







