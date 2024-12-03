#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

int main() {
    // 视频文件路径
    std::string videoPath = "C:/Users/22132/Desktop/vedio/20241026_225305.mp4";

    // 加载视频
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    // 加载 ONNX 模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx");

    // 检查是否成功加载模型
    if (net.empty()) {
        std::cerr << "Error: Could not load the ONNX model." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 预处理步骤
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

        // 设置输入
        net.setInput(blob);

        // 获取输出
        cv::Mat output = net.forward();

        // 处理输出（假设二分类）
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc);
        int label = maxLoc.x;

        // 显示标签（简单示例）
        cv::putText(frame, (label == 0 ? "q1" : "q2"), cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // 显示视频帧
        cv::imshow("Video", frame);

        // 按下 ESC 键退出
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
