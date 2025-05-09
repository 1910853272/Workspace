import numpy as np
import cv2

# 读取图像
img = cv2.imread('image.jpg')
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ScaleAbs(img, alpha, beta):
    dst = np.abs(np.float64(alpha) * img + np.float64(beta))
    dst[dst > 255] = 255
    return np.round(dst).astype(np.uint8)


# 检查图像是否成功读取
if img is None:
    print("图像未能加载，请检查路径")
else:
    # 自定义锐化卷积核
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])

    # 使用卷积核进行锐化处理
    sharpened_image = cv2.filter2D(grayImage, -1, sharpen_kernel)

    # 增强对比度：增加亮度和对比度
    enhanced_image = ScaleAbs(sharpened_image, alpha=1.5, beta=0)  # alpha 控制对比度, beta 控制亮度

    # 保存结果图像
    cv2.imwrite('result.jpg', enhanced_image)

    # 如果需要显示图像
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Sharpened Image", sharpened_image)
    # cv2.imshow("Enhanced Image", enhanced_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
