import numpy as np
import matplotlib.pyplot as plt

# 1. 指定 1001.txt 的完整路径
file_path = '1001.txt'  # 请根据你的目录结构调整

# 2. 读取数据（假设以制表符分隔，每行至少有 3 列）
data = np.loadtxt(file_path, delimiter='\t')

# 3. 提取“时间”（第 2 列）和“强度”（第 3 列）
time = data[:, 1]       # 单位：ms
intensity = data[:, 2]  # 测量值

# 4. 绘制时间—强度关系曲线
plt.figure()
plt.plot(time, intensity)
plt.xlabel('Time (ms)')
plt.ylabel('Intensity')
plt.title('1001.txt — Time vs. Intensity')
plt.grid(True)
plt.show()
