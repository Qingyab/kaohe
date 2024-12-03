#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    const std::string videoPath = "C:/Users/22132/Desktop/vedio/20241026_225305.mp4"; 

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::QRCodeDetector qrDecoder;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Point> points;
        std::string decodedText = qrDecoder.detectAndDecode(frame, points);

        if (!decodedText.empty()) {
            std::cout << "Decoded QR Code: " << decodedText << std::endl;

            // Draw the QR code boundaries on the frame
            if (!points.empty()) {
                for (size_t i = 0; i < points.size(); i++) {
                    cv::line(frame, points[i], points[(i + 1) % points.size()], cv::Scalar(255, 0, 0), 2);
                }
            }
        }

        cv::imshow("QR Code Detection", frame);
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
