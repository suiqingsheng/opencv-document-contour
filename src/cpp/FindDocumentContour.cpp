#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "../include/Utils.h"
#include "../include/DocumentContourProcessorAdvanced.h"
#include "../include/DocumentContourProcessorSimple1.h"
#include "../include/DocumentContourProcessorAdvancedWhite.h"
#include "../include/DocumentContourProcessorSimple2.h"
#include "../include/DocumentContourProcessorSimple3.h"
#include "../include/DocumentContourProcessorSimple4.h"
#include <iostream>

using namespace cv;
using namespace std;

static const int TARGET_MAX_DIMENSION = 200;
static const bool SHOW_RESULTS = true;

int main(int argc, const char** argv) {

	if (argc != 2) {
		std::cout << "Incorrect arguments count";
		return -1;
	}
	std::string fileName(argv[1]);
	Mat image = imread(fileName, CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cout << "Cannot read image file: " << fileName;
		return -1;
	}

    Mat resized;
    Utils::resizeImage(image, resized, TARGET_MAX_DIMENSION);
    vector<cv::Point> contour;

    // TODO: Uncomment to use DocumentContourProcessorSimple1
    DocumentContourProcessorSimple1 processorSimple1;
	contour = processorSimple1.findContour(resized, SHOW_RESULTS);

    // TODO: Uncomment to use DocumentContourProcessorSimple2
	DocumentContourProcessorSimple2 processorSimple2;
    // contour = processorSimple2.findContour(resized, SHOW_RESULTS);

    // TODO: Uncomment to use DocumentContourProcessorSimple3
	DocumentContourProcessorSimple3 processorSimple3;
	// contour = processorSimple3.findContour(resized, SHOW_RESULTS);

	// TODO: Uncomment to use DocumentContourProcessorSimple3
	DocumentContourProcessorSimple4 processorSimple4;
	// contour = processorSimple4.findContour(resized, SHOW_RESULTS);

    // TODO: Uncomment to use DocumentContourProcessorAdvanced
	DocumentContourProcessorAdvanced processorAdvanced;
	// contour = processorAdvanced.findContour(resized, SHOW_RESULTS);

    // TODO: Uncomment to use DocumentContourProcessorAdvancedWhite
	DocumentContourProcessorAdvancedWhite processorWhite;
	// contour = processorWhite.findContour(resized, SHOW_RESULTS);

    Utils::drawContour(resized, contour);
}

