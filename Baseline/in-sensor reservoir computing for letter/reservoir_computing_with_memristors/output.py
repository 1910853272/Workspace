import numpy as np
import matplotlib.pyplot as plt

# 模拟忆阻器的电导行为
def output_row_memristor(initial_state, input_signal):
    """
    模拟memristor的行为，生成电导值的序列。
    """
    a = [initial_state]
    for i in range(len(input_signal)):
        if input_signal[i] > 0:
            a.append(np.clip(a[i], 0.1, 1) * np.exp(1))
        else:
            a.append(np.clip(a[i], 1, 10) * (3 - np.exp(1)))
    return np.array(a).flatten()

# 创建输入数据
letters = ['l0_ya.npy', 'l1_yu.npy', 'l2_oi.npy', 'l3_yoi.npy', 'l4_yai.npy',
           'l5_p.npy', 'l6_m.npy', 'l7_t.npy', 'l8_r.npy', 'l9_b.npy']
d = {}  # 用来存储字母与电导值的映射
for lett in letters:
    # 加载文件并生成电导值
    data = np.load(lett)
    initial_state = np.random.random(1)
    output = [output_row_memristor(initial_state, row) for row in data]
    d[lett] = np.concatenate(output)

# 可视化每个字母的电导值变化
plt.figure(figsize=(15, 20))

for idx, lett in enumerate(letters):
    plt.subplot(5, 2, idx + 1)
    conductance_values = d[lett]
    plt.plot(conductance_values)
    plt.title(f'Conductance for {lett}')
    plt.xlabel('Time Step')
    plt.ylabel('Conductance')

plt.tight_layout()
plt.show()
