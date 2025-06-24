import cv2
import numpy as np

# 线性绝对值映射函数：y = saturate(|alpha * x + beta|)
def ScaleAbs(img, alpha, beta):
    dst = np.abs(np.float64(alpha) * img + np.float64(beta))
    dst[dst > 255] = 255
    return np.round(dst).astype(np.uint8)

# 拉普拉斯锐化卷积
def laplace(img):
    r, c = img.shape
    new_image = np.zeros((r, c), dtype=np.float32)
    L_kernel = np.array([[0, -1.01, 0],
                         [-1.01, 4.01, -1.01],
                         [0, -1.01, 0]], dtype=np.float32)

    # 卷积（注意边界到 r-2, c-2）
    for i in range(r - 2):
        for j in range(c - 2):
            new_image[i + 1, j + 1] = np.sum(img[i:i+3, j:j+3] * L_kernel)

    # 取绝对值并裁到 [0,255]，再转 uint8
    new_image = np.abs(new_image)
    new_image[new_image > 255] = 255
    return new_image.astype(np.uint8)

if __name__ == '__main__':
    # 1. 读取并灰度化
    img = cv2.imread('image.jpg')
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 拉普拉斯锐化
    Lap = laplace(grayImage)

    # 3. ScaleAbs 增强对比度
    # alpha >1 会放大细节（高对比度），beta 可用于整体亮度偏移
    alpha = 1.5
    beta  = 0.5
    Lap_enhanced = ScaleAbs(Lap, alpha, beta)

    # 保存处理后的图像
    cv2.imwrite('Convolution_Experiment.jpg', Lap_enhanced)
