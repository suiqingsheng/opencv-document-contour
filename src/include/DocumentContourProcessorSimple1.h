#ifndef FIND_DOCUMENT_CONTOUR_PROJ_DOCUMENTCONTOURPROCESSORSIMPLE1_H
#define FIND_DOCUMENT_CONTOUR_PROJ_DOCUMENTCONTOURPROCESSORSIMPLE1_H

#include "opencv2/imgproc.hpp"
#include <cstdio>

class DocumentContourProcessorSimple1 {

public:
    std::vector<cv::Point> findContour(cv::Mat &image, bool showResult);
};

#endif
