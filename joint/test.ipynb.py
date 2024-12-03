import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义函数用于绘制特征点
def plot_keypoints(img, keypoints):
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# 定义函数用于图像拼接
def stitch_images(image1, image2):
    # 转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用SIFT算法提取特征点和描述符
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 绘制特征点
    print("Step 1: 特征点提取")
    plot_keypoints(image1, keypoints1)
    plot_keypoints(image2, keypoints2)

    # 使用FLANN进行特征点匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比例测试来选择好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 计算单应性矩阵
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        return None

    # 使用透视变换进行图像拼接
    result = cv2.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1], max(image1.shape[0], image2.shape[0])))
    result[0:image2.shape[0], 0:image2.shape[1]] = image2

    return result


# 主函数
def main():
    # 开始计时
    start_time = time.time()

    # 读取图片
    img1 = cv2.imread('3.jpg')
    img2 = cv2.imread('4.jpg')
    img3 = cv2.imread('8.jpg')
    img4 = cv2.imread('9.jpg')

    # 拼接图片
    stitched1 = stitch_images(img1, img2)
    stitched2 = stitch_images(img3, img4)

    # 展示拼接结果
    if stitched1 is not None:
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(stitched1, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("拼接结果：3.jpg 和 4.jpg")
        plt.show()

    if stitched2 is not None:
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(stitched2, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("拼接结果：8.jpg 和 9.jpg")
        plt.show()

    # 计算帧率
    end_time = time.time()
    processing_time = end_time - start_time
    fps = 1 / processing_time
    print(f"处理时间: {processing_time:.4f} 秒")
    print(f"帧率: {fps:.4f} fps")


if __name__ == "__main__":
    main()
