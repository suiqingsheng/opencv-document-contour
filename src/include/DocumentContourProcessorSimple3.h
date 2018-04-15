#ifndef SRC_INCLUDE_DOCUMENTCONTOURPROCESSORSIMPLE3_H_
#define SRC_INCLUDE_DOCUMENTCONTOURPROCESSORSIMPLE3_H_

#include "opencv2/imgproc.hpp"
#include <cstdio>


class DocumentContourProcessorSimple3 {

public:
    std::vector<cv::Point> findContour(cv::Mat &image,  bool showResult);

};

#endif
