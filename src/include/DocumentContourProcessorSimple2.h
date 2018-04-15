#ifndef SRC_INCLUDE_DOCUMENTCONTOURPROCESSORSIMPLE2_H_
#define SRC_INCLUDE_DOCUMENTCONTOURPROCESSORSIMPLE2_H_

#include "opencv2/imgproc.hpp"
#include <cstdio>

class DocumentContourProcessorSimple2 {

public:
    std::vector<cv::Point> findContour(cv::Mat &image, bool showResult);
};

#endif
