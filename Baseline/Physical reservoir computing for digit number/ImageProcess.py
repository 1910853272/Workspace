from PIL import Image
import os
import numpy as np
import pandas as pd

# 定义输入和输出文件夹路径
input_folder = '/Users/lzs/Desktop/workspace/Baseline/Physical Reservoir Computing for Handwriting Digit Number/minist'  # 替换为28×28像素图片的文件夹路径
output_folder = '/Users/lzs/Desktop/workspace/Baseline/Physical Reservoir Computing for Handwriting Digit Number/number'  # 替换为保存20×20像素图片的文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义映射表
data_mapping = {
    '00000': 5.0, '00001': 45.16, '00010': 58.28, '00011': 85.67,
    '00100': 68.25, '00101': 102.73, '00110': 113.61, '00111': 161.61,
    '01000': 85.89, '01001': 104.65, '01010': 97.1, '01011': 140.56,
    '01100': 85.78, '01101': 127.6, '01110': 141.83, '01111': 175.4,
    '10000': 118.98, '10001': 144.5, '10010': 151.72, '10011': 171.72,
    '10100': 108.29, '10101': 148.19, '10110': 157.45, '10111': 181.47,
    '11000': 110.98, '11001': 151.18, '11010': 171.55, '11011': 203.96,
    '11100': 141.47, '11101': 194.09, '11110': 204.9, '11111': 251.4,
}

# 批量处理函数
def process_image_to_binary_matrix(image_path):
    # 打开图片
    img = Image.open(image_path)

    # 将图片转换为灰度图像（如果不是灰度图像）
    if img.mode != 'L':
        img = img.convert('L')

    # 调整图像大小为 20×20 像素（确保尺寸正确）
    img = img.resize((20, 20), Image.Resampling.LANCZOS)

    # 转换为 numpy 数组
    img_array = np.array(img)

    # 二值化处理：将像素值大于128的设置为1，其他设置为0
    binary_matrix = (img_array > 128).astype(int)

    # 返回二值化的图像矩阵（每行为一维数组）
    return binary_matrix

# 用于保存所有图片的二值化字符串数据
all_data = {}

# 遍历输入文件夹中的所有图片
for file_name in os.listdir(input_folder):
    # 构建完整路径
    input_path = os.path.join(input_folder, file_name)
    # 检查是否为图片文件
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 去除文件名中的 .png 后缀
        file_name_without_extension = os.path.splitext(file_name)[0]
        # 打开图片并处理
        binary_matrix = process_image_to_binary_matrix(input_path)
        # 将二值化矩阵转化为字符串形式
        binary_string = ''.join([''.join(map(str, row)) for row in binary_matrix])
        # 将字符串按5个字符分割并进行映射
        values = []
        for i in range(0, len(binary_string), 5):
            substring = binary_string[i:i+5]
            # 如果 substring 长度不为 5, 补充零
            if len(substring) < 5:
                substring = substring.ljust(5, '0')
            # 获取映射值
            value = data_mapping.get(substring, None)
            if value is not None:
                values.append(value)
        # 保存数据到字典，使用去掉扩展名的文件名作为键
        all_data[file_name_without_extension] = values
        # 保存处理后的图像到输出文件夹
        output_path_resized = os.path.join(output_folder, file_name_without_extension + '.png')
        img_resized = Image.open(input_path).resize((20, 20), Image.Resampling.LANCZOS)
        img_resized.save(output_path_resized)

# 将所有数据保存到 DataFrame
df = pd.DataFrame(all_data)
print(df)
# 将结果保存为 Excel 文件
output_excel_path = 'data.xlsx'
df.to_excel(output_excel_path, index=False)

print("二值化处理完成，数据已保存为 data.xlsx。")
