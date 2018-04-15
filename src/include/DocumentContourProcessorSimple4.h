
#ifndef FIND_DOCUMENT_CONTOUR_PROJECT_DOCUMENTCONTOURPROCESSORSIMPLE4_H
#define FIND_DOCUMENT_CONTOUR_PROJECT_DOCUMENTCONTOURPROCESSORSIMPLE4_H

#include "opencv2/imgproc.hpp"
#include <vector>

class DocumentContourProcessorSimple4 {

public:

    std::vector<cv::Point> findContour(cv::Mat &image, bool showResult);

};


#endif //FIND_DOCUMENT_CONTOUR_PROJECT_DOCUMENTCONTOURPROCESSORSIMPLE4_H
