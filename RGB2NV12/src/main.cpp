#include <opencv2/opencv.hpp>
#include <iostream>
#include "rgb2nv12.h"
#include "nv122rgb.h"
#include <chrono>
#include <fstream>

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("test1.jpg", IMREAD_COLOR);
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // 记录开始时间
    auto start = chrono::steady_clock::now();

    // RGB to NV12
    Mat nv12;
    rgb2nv12(image, nv12);

    string nv12Path = "output/test1_nv12.yuv";
    string jpgPath = "output/test1_reconstructed.jpg";

    // 保存NV12文件
    vector<uchar> nv12_buffer(nv12.total() * nv12.elemSize());
    memcpy(nv12_buffer.data(), nv12.data, nv12.total() * nv12.elemSize());
    ofstream outFile(nv12Path, ios::out | ios::binary);
    outFile.write((char*)nv12_buffer.data(), nv12_buffer.size());
    outFile.close();

    // NV12 to RGB and save as JPG
    Mat rgb;
    nv122rgb(nv12, rgb);
    imwrite(jpgPath, rgb);

    // 记录结束时间并计算时间差
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    double time_taken = chrono::duration <double, milli> (diff).count();

    cout << "Time taken: " << time_taken << " ms" << endl;
    cout << "Frame rate: " << 1000.0 / time_taken << " fps" << endl;

    return 0;
}
