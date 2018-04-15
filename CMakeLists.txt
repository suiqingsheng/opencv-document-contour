cmake_minimum_required(VERSION 3.5)
PROJECT( find_document_contour_project )
FIND_PACKAGE( OpenCV REQUIRED )

set(SOURCES src/cpp/FindDocumentContour.cpp
        src/include/DocumentContourProcessorAdvanced.h
        src/cpp/DocumentContourProcessorAdvanced.cpp
        src/include/Utils.h
        src/cpp/Utils.cpp
        src/include/DocumentContourProcessorAdvancedWhite.h
        src/cpp/DocumentContourProcessorAdvancedWhite.cpp
        src/include/DocumentContourProcessorSimple1.h
        src/cpp/DocumentContourProcessorSimple1.cpp
        src/include/DocumentContourProcessorSimple2.h
        src/cpp/DocumentContourProcessorSimple2.cpp
        src/include/DocumentContourProcessorSimple3.h
        src/cpp/DocumentContourProcessorSimple3.cpp
        src/include/DocumentContourProcessorSimple4.h
        src/cpp/DocumentContourProcessorSimple4.cpp )

ADD_EXECUTABLE( FindDocumentContour ${SOURCES})
TARGET_LINK_LIBRARIES( FindDocumentContour ${OpenCV_LIBS})