import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_float
from scipy import ndimage
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'


class FairComparisonImplementation:
    """
    公平对比：统一两种方法的数值尺度
    """

    def __init__(self, threshold=0.05, amplification_factor=5.0):
        """
        需要先去阅读NE文章，思想很简单，总共3*3像素，通过做差，发现有一定的数值差别了，就给中心像素加个负反馈，带着这个观点去看代码就很容易了

        参数:
            threshold (float): 梯度阈值，用于判断是否增强中心像素的响应。
            amplification_factor (float): 增强因子，当梯度超过阈值时，中心像素的权重。
        """
        self.threshold = threshold
        self.amplification_factor = amplification_factor

        # 论文的响应性矩阵（基础核）
        # 这是一个3x3的核，对角线元素为-0.25，中心为1，其他为0。
        self.base_responsivity_matrix = np.array([
            [-0.25, 0, -0.25],
            [0, 1, 0],
            [-0.25, 0, -0.25]
        ], dtype=np.float32)

        # 调整后的传统卷积核：与论文方法相似的数值尺度
        # 原始拉普拉斯核通常是: [[0,-1,0],[-1,4,-1],[0,-1,0]]
        # 这里将其系数调整，使其中心值和周围值与论文方法的基础核具有相似的数值范围，以实现公平对比。
        self.traditional_kernel = np.array([
            [0, -0.25, 0],
            [-0.25, 1, -0.25],
            [0, -0.25, 0]
        ], dtype=np.float32)

        # 也提供标准拉普拉斯作为参考，不进行数值调整。
        self.standard_laplacian = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def calculate_gradient(self, patch):
        """
        计算给定图像块的对角线梯度。
        梯度用于衡量局部区域的边缘或细节强度。

        参数:
            patch (numpy.ndarray): 一个3x3的图像块。

        返回:
            float: 计算出的对角线梯度值。
        """
        p1 = patch[0, 0]  # 左上角像素 (DP1)
        p3 = patch[0, 2]  # 右上角像素 (DP3)
        p6 = patch[2, 0]  # 左下角像素 (DP6)
        p8 = patch[2, 2]  # 右下角像素 (DP8)

        # 对角线梯度计算方法：|p8 - p1| + |p3 - p6|  论文里一样的思路，做差
        gradient = abs(p8 - p1) + abs(p3 - p6)
        return gradient

    def create_dynamic_responsivity_matrix(self, gradient):
        """
        根据计算出的梯度值创建动态响应性矩阵。
        如果梯度大于阈值，则增强中心像素的权重。

        参数:
            gradient (float): 当前图像块的梯度值。

        返回:
            tuple:
                - numpy.ndarray: 动态调整后的响应性矩阵。
                - bool: 一个标志，指示矩阵是否被增强 (True) 或未增强 (False)。
        """
        # 复制基础响应性矩阵，避免修改原始矩阵
        responsivity_matrix = self.base_responsivity_matrix.copy()
        is_enhanced = False  # 增强标志初始化

        # 如果梯度超过阈值，说明当前区域是边缘或细节，需要增强
        if gradient > self.threshold:
            responsivity_matrix[1, 1] = self.amplification_factor  # 增强中心像素权重
            is_enhanced = True
        else:
            # 否则，使用基础的中心权重1.0
            responsivity_matrix[1, 1] = 1.0
            is_enhanced = False

        return responsivity_matrix, is_enhanced

    def paper_dynamic_convolution(self, image):
        """
        实现论文中描述的动态卷积方法。
        该方法根据局部梯度动态调整卷积核（响应性矩阵）。

        参数:
            image (numpy.ndarray): 输入图像，可以是灰度或彩色图像。

        返回:
            tuple:
                - numpy.ndarray: 动态卷积后的结果图像。
                - numpy.ndarray: 一个布尔映射，指示图像中哪些区域的响应性矩阵被增强了。
        """
        # 如果是彩色图像 (3通道)，转换为灰度图像
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将图像转换为浮点类型，以进行精确计算
        image = img_as_float(image)
        height, width = image.shape
        # 初始化结果图像和增强映射，尺寸与输入图像相同
        result = np.zeros_like(image, dtype=np.float32)
        enhancement_map = np.zeros_like(image, dtype=bool)

        # 遍历图像的每个像素（除了边界，因为需要3x3的图像块）
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # 提取当前像素周围的3x3图像块
                patch = image[i - 1:i + 2, j - 1:j + 2]
                # 计算图像块的梯度
                gradient = self.calculate_gradient(patch)
                # 根据梯度创建动态响应性矩阵
                responsivity_matrix, is_enhanced = self.create_dynamic_responsivity_matrix(gradient)

                # 将图像块与动态响应性矩阵进行元素乘法
                device_responses = patch * responsivity_matrix
                # 对乘法结果求和，得到当前像素的总响应
                total_response = np.sum(device_responses)

                # 将总响应值存储到结果图像中，确保非负
                result[i, j] = max(0, total_response)
                # 记录当前像素是否被增强
                enhancement_map[i, j] = is_enhanced

        return result, enhancement_map

    def traditional_convolution_fair(self, image):
        """
        执行调整数值尺度后的传统卷积。
        此方法使用与论文方法相似数值范围的拉普拉斯核，以实现公平对比。

        参数:
            image (numpy.ndarray): 输入图像。

        返回:
            numpy.ndarray: 卷积后的结果图像（取绝对值）。
        """
        # 如果是彩色图像，转换为灰度图像
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将图像转换为浮点类型
        image = img_as_float(image)
        # 使用 SciPy 的 ndimage.convolve 函数进行卷积，mode='constant' 表示边界填充0
        result = ndimage.convolve(image, self.traditional_kernel, mode='constant')
        # 对结果取绝对值，因为拉普拉斯算子通常产生正负值，绝对值表示边缘强度
        result = np.abs(result)

        return result

    def traditional_convolution_standard(self, image):
        """
        执行标准拉普拉斯卷积，作为参考基准。
        此方法使用未调整的标准拉普拉斯核。

        参数:
            image (numpy.ndarray): 输入图像。

        返回:
            numpy.ndarray: 卷积后的结果图像（取绝对值）。
        """
        # 如果是彩色图像，转换为灰度图像
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将图像转换为浮点类型
        image = img_as_float(image)
        # 使用标准拉普拉斯核进行卷积
        result = ndimage.convolve(image, self.standard_laplacian, mode='constant')
        # 对结果取绝对值
        result = np.abs(result)

        return result


def analyze_kernel_differences():
    """
    分析并可视化不同卷积核的差异，包括它们的响应值和权重和。
    这有助于理解不同核的数值尺度。
    """
    print("=== 卷积核数值尺度分析 ===")

    # 创建一个测试用的3x3图像块
    test_patch = np.array([
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7]
    ])

    print("测试patch:")
    print(test_patch)
    print()

    # 定义不同类型的卷积核及其名称
    kernels = {
        '论文响应性矩阵': np.array([[-0.25, 0, -0.25], [0, 1, 0], [-0.25, 0, -0.25]]),
        '论文增强矩阵': np.array([[-0.25, 0, -0.25], [0, 5, 0], [-0.25, 0, -0.25]]),  # 中心元素被增强
        '调整后传统核': np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]),
        '标准拉普拉斯': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    }

    print("不同卷积核的响应对比:")
    print("卷积核类型          响应值    核的权重和")
    print("-" * 50)

    # 遍历每个卷积核，计算其在测试patch上的响应和核本身的权重和
    for name, kernel in kernels.items():
        response = np.sum(test_patch * kernel)  # 卷积响应 = 图像块元素乘核元素后求和
        kernel_sum = np.sum(kernel)  # 核的权重和
        print(f"{name:20} {response:8.3f}    {kernel_sum:8.3f}")  # 格式化输出

    print()

    # 可视化不同核的图像表示
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 创建2x2的子图布局
    fig.suptitle('Kernel Comparison', fontsize=16)  # 设置主标题

    kernel_list = list(kernels.items())
    for i, (name, kernel) in enumerate(kernel_list):
        row, col = i // 2, i % 2  # 计算当前子图的行和列
        # 显示卷积核的图像，使用红蓝渐变色图，并设定颜色范围
        im = axes[row, col].imshow(kernel, cmap='RdBu', vmin=-1, vmax=5)
        # 设置子图标题，包括核的名称和其权重和
        axes[row, col].set_title(f'{name}\nSum: {np.sum(kernel):.2f}')
        axes[row, col].axis('off')  # 关闭坐标轴

        # 在每个核的图像上显示具体的数值
        for r in range(3):
            for c in range(3):
                axes[row, col].text(c, r, f'{kernel[r, c]:.2f}',
                                    ha='center', va='center', fontsize=10)

        plt.colorbar(im, ax=axes[row, col], shrink=0.6)  # 为每个子图添加颜色条

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()  # 显示图表


def comprehensive_fair_comparison():
    """
    对不同图像在不同亮度下，进行论文方法、调整后传统方法和标准拉普拉斯方法的全面公平对比。
    """
    # 准备测试图像
    images = {
        'Cameraman': data.camera(),  # 内置的Cameraman灰度图像
        'Astronaut': cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY),  # 将彩色宇航员图像转换为灰度
        'Coins': data.coins()  # 内置的Coins灰度图像
    }

    # 定义不同的亮度级别和对应的名称  图像变暗 可以自行修改
    brightness_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    brightness_names = ['Original', 'Bright (×0.8)', 'Dim (×0.6)', 'Dark (×0.4)', 'Very Dark (×0.2)',
                        'Extremely Dark (×0.1)']

    # 创建图像检测器实例
    detector = FairComparisonImplementation(threshold=0.05, amplification_factor=5.0)

    # 遍历每张图像进行测试
    for img_name, original_img in images.items():
        print(f"\n=== 公平对比测试: {img_name} ===")

        # 为当前图像创建一个大的图，包含所有亮度级别的结果
        fig, axes = plt.subplots(len(brightness_levels), 5, figsize=(20, 24))
        fig.suptitle(f'Fair Comparison: {img_name}', fontsize=16)

        # 遍历每个亮度级别
        for i, (brightness, level_name) in enumerate(zip(brightness_levels, brightness_names)):
            # 调整图像亮度，并转换为浮点类型
            dimmed_img = img_as_float(original_img) * brightness

            # 分别使用三种方法处理图像
            paper_result, enhancement_map = detector.paper_dynamic_convolution(dimmed_img)
            traditional_fair = detector.traditional_convolution_fair(dimmed_img)
            traditional_standard = detector.traditional_convolution_standard(dimmed_img)

            # 统计并打印一些关键信息
            enhancement_percent = np.sum(enhancement_map) / enhancement_map.size * 100  # 论文方法增强区域的百分比
            paper_max = np.max(paper_result)  # 论文方法结果的最大响应值
            fair_max = np.max(traditional_fair)  # 公平传统方法结果的最大响应值
            standard_max = np.max(traditional_standard)  # 标准拉普拉斯方法结果的最大响应值

            # 显示结果图像
            axes[i, 0].imshow(dimmed_img, cmap='gray', vmin=0, vmax=1)  # 显示当前亮度下的原始图像
            axes[i, 0].set_title(f'{level_name}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(traditional_fair, cmap='gray')  # 显示公平传统方法结果
            axes[i, 1].set_title(f'Traditional Fair\nMax: {fair_max:.3f}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(paper_result, cmap='gray')  # 显示论文动态方法结果
            axes[i, 2].set_title(f'Paper Dynamic\nMax: {paper_max:.3f}')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(traditional_standard, cmap='gray')  # 显示标准拉普拉斯方法结果
            axes[i, 3].set_title(f'Standard Laplacian\nMax: {standard_max:.3f}')
            axes[i, 3].axis('off')

            # 性能对比条形图
            methods = ['Fair Trad', 'Paper', 'Standard']
            max_responses = [fair_max, paper_max, standard_max]

            axes[i, 4].bar(methods, max_responses, alpha=0.7, color=['blue', 'red', 'green'])  # 绘制条形图
            axes[i, 4].set_title(f'Response Comparison\nPaper/Fair: {paper_max / fair_max:.2f}')  # 设置标题，包含论文/公平传统的比率
            axes[i, 4].set_ylabel('Max Response')  # Y轴标签

            # 打印当前亮度级别下的详细数值对比
            print(f"  {level_name}: Fair={fair_max:.3f}, Paper={paper_max:.3f}, "
                  f"Standard={standard_max:.3f}, Ratio={paper_max / fair_max:.2f}")

        plt.tight_layout()  # 自动调整布局
        plt.show()  # 显示图表


def demonstrate_enhancement_effect():
    """
    详细演示论文方法的动态增强效果。
    选取图像的局部区域，分析不同方法的响应和增强区域。这部分不重要，就是一开始做测试用的 直接看后面核心部分
    """
    print("\n=== 增强效果细节演示 ===")

    # 从cameraman图像中截取一个局部区域，并将其亮度减小进行测试
    cameraman = img_as_float(data.camera())
    region = cameraman[100:130, 150:180] * 0.2  # 截取并变暗的测试区域

    # 创建图像检测器实例
    detector = FairComparisonImplementation(threshold=0.05, amplification_factor=5.0)

    # 对局部区域应用不同方法
    paper_result, enhancement_map = detector.paper_dynamic_convolution(region)
    traditional_fair = detector.traditional_convolution_fair(region)
    traditional_standard = detector.traditional_convolution_standard(region)

    # 打印输入图像和输出结果的统计信息
    print(f"输入图像统计:")
    print(f"  平均亮度: {np.mean(region):.3f}")
    print(f"  亮度范围: {np.min(region):.3f} - {np.max(region):.3f}")
    print()

    print(f"输出结果统计:")
    print(f"  论文方法 - 最大: {np.max(paper_result):.3f}, 平均: {np.mean(paper_result):.3f}")
    print(f"  公平传统 - 最大: {np.max(traditional_fair):.3f}, 平均: {np.mean(traditional_fair):.3f}")
    print(f"  标准传统 - 最大: {np.max(traditional_standard):.3f}, 平均: {np.mean(traditional_standard):.3f}")
    print()

    # 计算增强区域的百分比
    enhancement_percent = np.sum(enhancement_map) / enhancement_map.size * 100
    print(f"增强区域: {enhancement_percent:.1f}%")

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 创建2x3的子图布局
    fig.suptitle('Enhancement Effect Demonstration', fontsize=16)  # 设置主标题

    # 第一行：显示输入图像和两种主要方法的结果
    axes[0, 0].imshow(region, cmap='gray')
    axes[0, 0].set_title('Input Region')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(traditional_fair, cmap='gray')
    axes[0, 1].set_title('Traditional Fair')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(paper_result, cmap='gray')
    axes[0, 2].set_title('Paper Dynamic')
    axes[0, 2].axis('off')

    # 第二行：分析增强映射、响应分布和最大响应对比
    axes[1, 0].imshow(enhancement_map.astype(float), cmap='gray')  # 显示增强映射 (True为1，False为0)
    axes[1, 0].set_title(f'Enhancement Map\n({enhancement_percent:.1f}%)')
    axes[1, 0].axis('off')

    # 绘制响应分布直方图
    axes[1, 1].hist(traditional_fair.flatten(), bins=50, alpha=0.7, label='Traditional Fair')
    axes[1, 1].hist(paper_result.flatten(), bins=50, alpha=0.7, label='Paper Dynamic')
    axes[1, 1].set_title('Response Distribution')
    axes[1, 1].set_xlabel('Response Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()  # 显示图例

    # 响应对比条形图
    methods = ['Traditional Fair', 'Paper Dynamic', 'Standard Laplacian']
    max_responses = [np.max(traditional_fair), np.max(paper_result), np.max(traditional_standard)]

    axes[1, 2].bar(methods, max_responses, alpha=0.7, color=['blue', 'red', 'green'])
    axes[1, 2].set_title('Max Response Comparison')
    axes[1, 2].set_ylabel('Max Response')
    axes[1, 2].tick_params(axis='x', rotation=45)  # 旋转X轴标签，防止重叠

    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图表


def main():
    """
    主函数，程序的入口点。
    负责组织和调用各项分析和对比功能。
    """

    print("调整方案：")
    print("  论文基础核: [[-0.25, 0, -0.25], [0, 1, 0], [-0.25, 0, -0.25]]")
    print("  调整传统核: [[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]")
    print("  标准拉普拉斯: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]] (参考)")
    print()

    # 1. 分析不同卷积核的差异，包括它们的权重和在测试图像块上的响应
    analyze_kernel_differences()

    # 2. 对不同图像和亮度级别下的三种方法进行全面的公平对比
    comprehensive_fair_comparison()

    # 3. 详细演示论文方法的动态增强效果及其对结果的影响
    demonstrate_enhancement_effect()


# 当脚本直接运行时，执行主函数
if __name__ == "__main__":
    main()