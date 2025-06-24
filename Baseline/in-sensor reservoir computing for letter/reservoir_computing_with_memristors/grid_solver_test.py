import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

# 定义网格图的尺寸 Nx和Ny分别表示网格的列数和行数
Nx = 300
Ny = 40
G = nx.grid_2d_graph(Ny, Nx)  # 创建一个二维网格图

# 绘图设置
fig, ax = plt.subplots(1, 1)
pos = {(x, y): (y, -x) for x, y in G.nodes()}
ax.set_title('Voltage and Current Distribution')
ax.axis('on')

# 定义左边界电压和右边界电压
V_left = 10.0  # 左边界的电压（10V）
V_right = 0.0  # 右边界的电压（接地）

# 构建电阻矩阵
R_internal = 2  # 内部电阻（欧姆）
R_boundary = 0.001  # 边界电阻（欧姆）

# 初始化节点电压
voltages = {node: V_left if node[1] == 0 else (V_right if node[1] == Nx - 1 else 0) for node in G.nodes()}

# 迭代计算节点电压（使用拉普拉斯迭代法）
def update_voltages(G, voltages, R_internal, R_boundary, max_iter=1000, tol=1e-5):
    for _ in range(max_iter):
        new_voltages = voltages.copy()
        max_diff = 0
        for node in G.nodes():
            if node[1] == 0:  # 左边界
                new_voltages[node] = V_left
            elif node[1] == Nx - 1:  # 右边界
                new_voltages[node] = V_right
            else:
                # 邻居节点电压的平均值
                neighbor_voltages = [voltages[neighbor] for neighbor in G.neighbors(node)]
                new_voltages[node] = sum(neighbor_voltages) / len(neighbor_voltages)

            max_diff = max(max_diff, abs(new_voltages[node] - voltages[node]))

        voltages = new_voltages
        if max_diff < tol:
            break
    return voltages

# 更新节点电压
voltages = update_voltages(G, voltages, R_internal, R_boundary)

# 将节点电压映射为颜色
node_colors = [plt.cm.jet(voltages[node] / V_left) for node in G.nodes()]

# 计算电流
currents = []
for edge in G.edges():
    voltage_diff = voltages[edge[0]] - voltages[edge[1]]
    current = voltage_diff / R_internal  # Ohm 定律计算电流
    currents.append(plt.cm.jet(current / 2.5))

# 绘制带颜色的电网图，节点和边的颜色表示电压和电流
nx.draw(
    G, pos, ax=ax,
    node_color=node_colors,       # 节点颜色根据电压
    edge_color=currents,          # 边的颜色根据电流
    node_size=30,
    with_labels=False,
    width=1,
)
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), ax=ax, orientation='vertical', label='Voltage (V)')
plt.show()
