# -*- coding: utf-8 -*-

import networkx as nx  # 导入NetworkX用于处理图结构
from matplotlib import pyplot as plt  # 导入Matplotlib用于绘图

# 导入PySpice库以便进行电路仿真
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# 创建一个电路实例，用于使用Millman定理分析电路
circuit = Circuit("Millman's theorem")

# 定义网格图的尺寸 Nx和Ny分别表示网格的列数和行数
Nx = 300
Ny = 40
G = nx.grid_2d_graph(Ny, Nx)  # 创建一个二维网格图

# 绘图设置
fig, ax = plt.subplots(1, 1)
# 设置节点的位置，使其按网格排列
pos = {(x, y): (y, -x) for x, y in G.nodes()}
# 设置图的标题和坐标轴显示
ax.set_title('ee')
ax.axis('on')

# 检查轴对象的其他图像组件
ax.lines
ax.patches
ax.artists

# 定义一个获取字典键的辅助函数
def get_key(val, dictu):
    """
    从字典中查找与给定值对应的键。
    """
    for key, value in dictu.items():
         if val == value:
             return key

# 创建一个字典，用于为每个网格节点编号
dict_nodes = {}
for n, node in enumerate(G.nodes()):
    print(n, node)
    dict_nodes[n+1] = node  # 从1开始编号

# 检查get_key函数的正确性
assert 2 == get_key((0,1), dict_nodes)

# 打印节点总数
print('节点总数为:', len(dict_nodes))

# 定义虚拟的输入节点编号
NV = Nx * Ny + 100

# 在电路中添加一个输入电压源，连接到NV节点和地
circuit.V('input', NV, circuit.gnd, 10 @u_V)

# 将电阻连接到网格中的节点
i = 0
# 遍历所有节点，给边界的节点添加电阻
for x, y in G.nodes():
    if x == 0:  # 若节点在左边界
        circuit.R('el' + str(i), NV,
                  get_key((x, y), dict_nodes), 0.001 @u_kOhm)
        i += 1  # 增加电阻编号

# 遍历图中的边，为每一条边添加一个电阻连接两个相邻节点
for edge in G.edges():
    origin, end = edge
    print(origin, get_key(origin, dict_nodes), '-->',
          end, get_key(end, dict_nodes))
    # 使用2 kOhm电阻连接两个相邻节点
    circuit.R('el' + str(i), get_key(origin, dict_nodes),
              get_key(end, dict_nodes), 2@u_kOhm)
    i += 1  # 增加电阻编号

# 为右边界节点连接到地，添加电阻
for x, y in G.nodes():
    if x == Ny - 1:  # 若节点在右边界
        circuit.R('el' + str(i), get_key((x, y), dict_nodes),
                  circuit.gnd, 0.001@u_kOhm)
        i += 1

# 创建电路模拟器
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# 执行操作点分析以计算节点电压和电流
analysis = simulator.operating_point()

# 输出每个节点的电压值
for node in analysis.nodes.values():
    print('节点 {}: {:4.1f} V'.format(str(node), float(node)))

# 输出每个分支的电流值
for node in analysis.branches.values():
    print('节点 {}: {:5.2f} A'.format(str(node), float(node)))

#%%
# 为显示电压和电流设置颜色映射
voltages = [i for i in range(0, Nx * Ny)]
nodes = [i for i in range(0, Nx * Ny)]

# 将节点电压转换为颜色
for node in analysis.nodes.values():
    nnode = int(str(node))
    if nnode in list(range(1, Nx * Ny + 1)):
        print(nnode, float(node))
        voltages[nnode-1] = plt.cm.jet(float(node) / 10.)
        nodes[nnode-1] = float(node)

# 计算电流
currents = []
for edge in G.edges():
    origin, end = edge
    idx_init = get_key(origin, dict_nodes)
    idx_end = get_key(end, dict_nodes)
    # 使用两个相邻节点的电压差作为电流近似值
    current = nodes[idx_init-1] - nodes[idx_end-1]
    print(origin, idx_init, '-->', end, idx_end, ':', current)
    currents.append(plt.cm.jet(float(current) / 2.5))

# 绘制带颜色的电网图，节点和边的颜色表示电压和电流
nx.draw(
    G, pos, ax=ax,
    node_color=voltages,       # 节点颜色根据电压
    edge_color=currents,       # 边的颜色根据电流
    node_size=400,
    with_labels=False,
    width=6,
)
