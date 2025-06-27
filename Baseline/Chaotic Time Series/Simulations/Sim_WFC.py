import numpy as np
import matplotlib.pyplot as plt
import DMnode as DM  # 导入自定义的动态忆阻器模型模块

##################################################
# GLOBAL VARIABLES
##################################################
# 测试控制参数
SEED = 20  # 设置随机种子，确保实验可重复
STEP = 400  # 设置模拟的时间步数为400
WUP = 50  # 设置预热步数，前50步的数据将不被用于训练

# 超参数优化
ML = 5  # 多步长度，影响数据的时间步数
N = 24  # 动态忆阻器的节点数
T = 0.25  # 动态忆阻器的温度参数
S = 2  # 动态忆阻器的一个控制参数
alpha = 0.2  # 动态忆阻器的另一个控制参数

##################################################
# NETWORK MODEL
##################################################
# Mask设置：创建一个随机生成的Mask矩阵，取值为-1和1
Mask = 2*np.random.randint(2, size=(ML, N)) - 1  # 生成一个(ML, N)大小的随机矩阵，元素为-1或1
# 初始化动态忆阻器节点
u = DM.DM_node(T, S, alpha)

# 动态忆阻器分类器函数
def DMClassifier(Input):
    L = len(Input[0, :])  # 获取输入数据的时间步长（列数）
    Input_ex = np.zeros((N, L*ML))  # 初始化扩展后的输入数据矩阵，维度为(N, L*ML)

    # 遍历每个动态忆阻器节点，进行输入扩展
    for j in range(N):
        # 使用Mask矩阵和输入数据进行矩阵乘法，得到扩展输入
        Input_ex[j, :] = np.dot(Input[[0], :].T, Mask[:, [j]].T).reshape((1, -1))

    memout = np.zeros((N, L*ML))  # 初始化存储忆阻器输出的矩阵
    Vm = 0  # 初始化电压为0

    # 计算动态忆阻器的输出
    for i in range(L*ML):
        # 将每个输入传给动态忆阻器，并更新电压状态
        memout[:, i], Vm = u.test(Input_ex[:, i], Vm)

    # 初始化神经网络的输出矩阵
    neuout = np.zeros((N*ML, L))
    for i in range(L):
        # 将多个时间步的输出整合为最终的神经网络输出
        neuout[:, i] = memout[:, i*ML:(i+1)*ML].reshape((-1, 1))[:, 0]

    return neuout  # 返回神经网络的最终输出

##################################################
# DATASET
##################################################
# 数据生成函数，用于生成输入和目标数据
def DataGen(step):
    sample = 8  # 每次生成8个样本
    np.random.seed(SEED)  # 设置随机种子以保证可重复性
    p1 = (np.random.rand(sample)-0.5)*2  # 生成一组随机数据，范围在[-1, 1]之间
    p2 = (np.random.rand(sample)-0.5)*2  # 生成另一组随机数据，范围在[-1, 1]之间
    np.random.seed()  # 重置随机种子
    Input = np.zeros((1, step))  # 初始化输入数据矩阵，维度为(1, step)
    Target = np.zeros((1, step))  # 初始化目标标签矩阵，维度为(1, step)

    # 按照给定的步数生成数据
    for i in range(int(step/sample)):
        q = np.random.randint(2)  # 随机选择0或1
        if q == 1:
            Input[0, sample*i:sample*(i+1)] = p1  # 填充输入数据
            Target[0, sample*i:sample*(i+1)] = 1  # 填充目标标签
        else:
            Input[0, sample*i:sample*(i+1)] = p2  # 填充输入数据
            Target[0, sample*i:sample*(i+1)] = 0  # 填充目标标签
    Input = np.vstack([Input, Input, Input])  # 将输入数据复制三次，以满足神经网络输入要求
    return Input, Target  # 返回生成的输入数据和目标标签

##################################################
# SYSTEM RUN
##################################################
# 训练过程
def Train(Input, Target):
    # 通过动态忆阻器分类器计算训练数据的状态
    States = DMClassifier(Input)
    # 在状态矩阵顶部添加一行全1的行
    States = np.vstack([np.ones((1, STEP)), States])
    # 使用最小二乘法计算输出权重Wout
    Wout = Target[:, WUP:].dot(States[:, WUP:].T).dot(np.linalg.pinv(np.dot(States[:, WUP:], States[:, WUP:].T)))
    # 使用权重计算最终输出
    Output = np.dot(Wout, States)
    # 计算训练误差（标准化均方根误差）
    NRMSE = np.mean(np.sqrt(np.mean((Output[:, WUP:]-Target[:, WUP:])**2, axis=1)/np.var(Target[:, WUP:], axis=1)))
    print('Train_error: ' + str(NRMSE))  # 输出训练误差
    return Wout, NRMSE  # 返回训练得到的权重和误差

# 测试过程
def Test(Wout, Input, Target):
    # 通过动态忆阻器分类器计算测试数据的状态
    States = DMClassifier(Input)
    # 在状态矩阵顶部添加一行全1的行
    States = np.vstack([np.ones((1, STEP)), States])
    # 使用训练得到的权重Wout计算测试输出
    Output = np.dot(Wout, States)
    # 计算测试误差
    NRMSE = np.mean(np.sqrt(np.mean((Output[:, WUP:]-Target[:, WUP:])**2, axis=1)/np.var(Target[:, WUP:], axis=1)))
    print('Test_error: ' + str(NRMSE))  # 输出测试误差
    return Output, States  # 返回输出和状态

##################################################
# MAIN
##################################################
if __name__ == '__main__':
    # 训练过程
    Input, Target_train = DataGen(STEP)  # 生成训练数据
    Wout, _ = Train(Input, Target_train)  # 训练模型并得到权重

    # 测试过程
    Input, Target_test = DataGen(STEP)  # 生成测试数据
    Output, States = Test(Wout, Input, Target_test)  # 测试模型并获得输出

    # 绘图
    plt.figure()  # 创建一个新的图形窗口
    plt.subplot(2, 1, 1)  # 创建第一个子图
    plt.plot(Input.T)  # 绘制输入数据
    plt.ylabel('Input')  # 设置y轴标签
    plt.xlim(0, STEP)  # 设置x轴范围
    plt.subplot(2, 1, 2)  # 创建第二个子图
    plt.plot(Target_test.T)  # 绘制目标数据
    plt.plot(Output.T)  # 绘制模型输出数据
    plt.axis([0, STEP, -0.2, 1.2])  # 设置坐标轴范围
    plt.xlabel('Time Step')  # 设置x轴标签
    plt.ylabel('Output')  # 设置y轴标签

    plt.figure()  # 创建另一个图形窗口
    plt.plot(States.T)  # 绘制状态数据
    plt.xlim(0, 200)  # 设置x轴范围

    plt.show()  # 显示所有绘制的图形
