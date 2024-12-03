#include "nv122rgb.h"
#include <opencv2/opencv.hpp>

void nv122rgb(const cv::Mat& src, cv::Mat& dst) {
    int width = src.cols;
    int height = src.rows * 2 / 3; // NV12的高度是实际图像高度的3/2

    // 创建RGB图像
    dst.create(height, width, CV_8UC3);

    const uchar* yPlane = src.data;
    const uchar* uvPlane = yPlane + (width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int yIndex = y * width + x;
            int uIndex = (y / 2) * (width / 2) + (x / 2) * 2;
            int vIndex = uIndex + 1;

            int Y = yPlane[yIndex];
            int U = uvPlane[uIndex] - 128;
            int V = uvPlane[vIndex] - 128;

            // 转换公式（这里使用的是一个简化的版本，实际应用中可能需要更精确的转换）
            int R = Y + (1.402 * V);
            int G = Y - (0.344 * U) - (0.714 * V);
            int B = Y + (1.772 * U);

            // 确保颜色值在0-255范围内
            R = std::max(0, std::min(255, R));
            G = std::max(0, std::min(255, G));
            B = std::max(0, std::min(255, B));

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(B), static_cast<uchar>(G), static_cast<uchar>(R));
        }
    }
}
