import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import DMnode as DM  # 导入动态忆阻器模型的模块
from sklearn.model_selection import train_test_split  # 用于数据集的训练集和测试集分割

##################################################
# GLOBAL VARIABLES
##################################################

# 任务设置
N_CLASSES = 1  # 设置目标类别数为1
STEP = 1000  # 数据处理的步数，表示最多处理1000步数据

# 超参数优化
ML = 5  # 设置多步长度，影响神经网络的时间步长
N = 24  # 设置动态忆阻器的节点数
T = 0.08  # 设置动态忆阻器的温度参数
S = 2  # 设置动态忆阻器的某个控制参数
alpha = 0.035  # 设置动态忆阻器的另一个控制参数

##################################################
# NETWORK MODEL
##################################################
# 从ECGpara.mat文件中加载Mask矩阵，用于输入的扩展
Mask = io.loadmat('ECGpara.mat')['Mask']
# 创建一个动态忆阻器节点u，传入的参数T、S、alpha
u = DM.DM_node(T, S, alpha)

# 动态忆阻器分类器函数
def DMClassifier(Input):
    L = len(Input[0, :])  # 获取输入数据的时间步长（列数）
    Input_ex = np.zeros((N, L*ML))  # 初始化扩展后的输入数据矩阵，维度是(N, L*ML)

    # 遍历动态忆阻器节点，对每个节点进行输入扩展
    for j in range(N):
        # 使用Mask矩阵与输入数据进行矩阵乘法，得到扩展输入
        Input_ex[j, :] = np.dot(Input[[0], :].T, Mask[:, [j]].T).reshape((1, -1))

    memout = np.zeros((N, L*ML))  # 初始化存储动态忆阻器输出的矩阵
    Vm = 0  # 初始电压设置为0

    # 计算动态忆阻器的输出
    for i in range(L*ML):
        # 每次输入给动态忆阻器，并更新其电压状态
        memout[:, i], Vm = u.test(Input_ex[:, i], Vm)

    # 初始化神经网络的输出矩阵
    neuout = np.zeros((N*ML, L))
    for i in range(L):
        # 将多个时间步的输出结果整合为最终的神经网络输出
        neuout[:, i] = memout[:, i*ML:(i+1)*ML].reshape((-1, 1))[:, 0]

    return neuout  # 返回最终的神经网络输出

##################################################
# DATA PROCESSING
##################################################
# 数据处理函数，将输入和目标标签转换为适合神经网络输入的格式
def DataProcess(x, y):
    # 对输入x进行形状调整
    Input = x.reshape((1, -1, 1))[0, :, :].T
    # 对标签y进行形状调整，符合目标输出的维度
    Target = y.reshape((1, -1, N_CLASSES))[0, :, :].T
    return Input, Target  # 返回处理后的输入和目标数据

# 数据生成函数，加载数据并进行预处理
def DataGen():
    # 从'ECGdataset.mat'文件中加载数据，并截取前STEP步数据
    data = io.loadmat('ECGdataset.mat')['dataset'][:STEP, :, :]

    # 对输入数据进行归一化处理，确保每个样本的值在[-1, 1]之间
    inputs = data[:, :, 0]/np.max(np.abs(data[:, :, 0]), axis=1).reshape((-1, 1))
    labels = data[:, :, 1:]  # 获取标签数据

    # 输出输入数据和标签数据的形状
    print("Data shape: ", inputs.shape)
    print("Labels shape:", labels.shape)

    # 将数据分为训练集和测试集，使用train_test_split函数，30%的数据为测试集
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size=0.3)
    # 输出训练集和测试集的大小
    print("X train size: ", len(X_train))
    print("X test size: ", len(X_test))
    print("Y train size: ", len(Y_train))
    print("Y test size: ", len(Y_test))
    return X_train, X_test, Y_train, Y_test  # 返回训练集和测试集

##################################################
# SYSTEM RUN
##################################################
# 训练过程
def Train(Input, Target):
    # 通过动态忆阻器分类器计算训练数据的状态
    States = DMClassifier(Input)
    # 在状态矩阵顶部添加一行全1的行
    States = np.vstack([np.ones((1, len(Input[0, :]))), States])
    # 使用最小二乘法计算输出权重Wout
    Wout = Target.dot(States.T).dot(np.linalg.pinv(np.dot(States, States.T)))
    # 计算最终输出
    Output = np.dot(Wout, States)
    # 计算训练误差，使用标准化均方根误差（NRMSE）
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Train_error: ' + str(NRMSE))  # 输出训练误差
    return Wout, NRMSE  # 返回权重和训练误差

# 测试过程
def Test(Wout, Input, Target):
    # 通过动态忆阻器分类器计算测试数据的状态
    States = DMClassifier(Input)
    # 在状态矩阵顶部添加一行全1的行
    States = np.vstack([np.ones((1, len(Input[0, :]))), States])
    # 使用训练得到的权重Wout计算测试输出
    Output = np.dot(Wout, States)
    # 计算测试误差
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))  # 输出测试误差
    return Output, States, NRMSE  # 返回输出、状态和测试误差

##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # 加载数据并划分为训练集和测试集
    X_train, X_test, Y_train, Y_test = DataGen()

    # 训练过程
    Input, Target = DataProcess(X_train, Y_train)  # 处理训练数据
    Wout, _ = Train(Input, Target)  # 训练模型并得到输出权重

    # 测试过程
    Input, Target = DataProcess(X_test, Y_test)  # 处理测试数据
    Output, States, _ = Test(Wout, Input, Target)  # 测试模型并获得输出

    # 准确率计算
    LEN = 50  # 每个样本的长度
    ACC = np.zeros((60, 5))  # 初始化准确率矩阵
    TH_list = np.zeros(2)  # 阈值列表
    TH_box = np.arange(0.21, 0.8, 0.01)  # 阈值的范围
    THS_box = np.arange(1, 6)  # 阈值数量范围
    j = 0
    for TH in TH_box:  # 遍历所有阈值
        k = 0
        for THS in THS_box:  # 遍历所有阈值数量
            # 计算输出的阈值结果
            Fout = np.heaviside(Output[0, :].reshape(-1, LEN)-TH, 1)
            Fout = np.heaviside(np.sum(Fout, axis=1)-THS, 1)
            # 获取目标的最大值
            Ftar = np.max(Target[0, :].reshape(-1, LEN), axis=1)
            # 比较输出与目标之间的差异
            Fbox = Fout-Ftar
            # 计算准确率
            ACC[j, k] = len(Fbox[Fbox == 0])/len(Fbox)
            k = k+1
        j = j+1
    print(np.max(ACC))  # 输出最大准确率

    # 绘制输入和输出数据的对比图
    plt.figure()  # 创建图形
    plt.subplot(2, 1, 1)  # 绘制第一个子图
    plt.plot(Input.T)  # 绘制输入数据
    plt.axis([0, 5000, -1, 1])  # 设置坐标轴范围
    plt.ylabel('Input')  # 设置y轴标签
    plt.subplot(2, 1, 2)  # 绘制第二个子图
    plt.plot(Target[0, :])  # 绘制目标数据
    plt.plot(Output[0, :])  # 绘制输出数据
    plt.axis([0, 5000, -0.2, 1.2])  # 设置坐标轴范围
    plt.ylabel('Output')  # 设置y轴标签
    plt.show()  # 显示图形
