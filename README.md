# opencv-document-contour
Experiments with OpenCV image processing functions to recognize document contour.

## How it works?

All algorithms are based on the "classic" approach to find document contours:

1. Resize image - significantly speeds up image processing.
2. Convert image to gray - speeds up processing, better for some image processing functions. 
3. Make image blurred - reduces impact of small contrasting regions on the whole recognition process.
4. Apply "Canny Edge" - finds any edges between different components on image.
5. Dilate - increases size of every edge which allows to remove holes and cuts between edges and strengthens the value of the most important edges.
6. Find lines with Hough transformation and draw - finds all lines which can connect broken edges.
7. Find contours - finds all closed contours on drawed lines and edges and represents contours with points.
8. Apply any kind of filters (convexHull, calculate arc and area, approxPolyDP) - filters and finds better contours. 

Algorithms have a bit differences in sequence of steps, input params for image processing functions.
[DocumentContourProcessorAdvanced](src/cpp/DocumentContourProcessorAdvanced.cpp) contains a bit different aproach for filtering contours and points (whith help of Sobel function and some kinds of custom points sorting and filtering), and [DocumentContourProcessorAdvancedWhite](src/cpp/DocumentContourProcessorAdvancedWhite.cpp) tries to recognize white document contour on white surface.

## License
This project is licensed under the MIT - see the [LICENSE.md](LICENSE.md) file for details.
