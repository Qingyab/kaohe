#include "rgb2nv12.h"
#include <opencv2/opencv.hpp>

void rgb2nv12(const cv::Mat& src, cv::Mat& dst) {
    int width = src.cols;
    int height = src.rows;
    int ySize = width * height;
    int uvSize = (width / 2) * (height / 2);

    dst.create(height * 3 / 2, width, CV_8UC1);
    uchar* yPlane = dst.data;
    uchar* uvPlane = yPlane + ySize;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b rgb = src.at<cv::Vec3b>(y, x);
            int r = rgb[2], g = rgb[1], b = rgb[0];

            // Y calculation
            yPlane[y * width + x] = static_cast<uchar>((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;

            if (y % 2 == 0 && x % 2 == 0) {
                // U and V calculation
                int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                uvPlane[(y / 2) * (width / 2) + (x / 2) * 2] = static_cast<uchar>(u);
                uvPlane[(y / 2) * (width / 2) + (x / 2) * 2 + 1] = static_cast<uchar>(v);
            }
        }
    }
}
