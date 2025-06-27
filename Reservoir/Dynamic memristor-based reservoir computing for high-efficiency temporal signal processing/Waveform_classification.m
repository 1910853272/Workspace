%% Waveform classification
clc;
clear;

% ----------------------模型参数----------------------
para.r = 0.99;     % 电导更新的衰减因子（记忆衰减因子），控制忆阻器的更新速率
para.G0 = 0.5;     % 初始电导值（G0），用于初始化忆阻器
para.Kp = 9.13;    % 电导正向常数，用于计算电流
para.Kn = 0.32;    % 电导反向常数，用于计算电流
para.alpha = 0.23; % 更新因子alpha，用于控制电导更新的速度

% ----------------------DM_RC 参数----------------------
ML = 4;    % 每个掩膜行的列数，影响输入特征的扩展
N = 10;    % 掩膜的数量
Vmax = 3;  % 电压最大值
Vmin = -2; % 电压最小值

% ----------------------数据集----------------------
sample = 8;  % 每个样本的长度
step = 2000; % 数据的总长度（步数）
Data = zeros(1, 2 * step);  % 初始化数据数组
p1 = sin(pi * 2 * (0:sample-1) / sample);  % 正弦波信号（p1）
p2(1:sample/2) = 1;  % 方波信号的前一半为1
p2(sample/2+1:sample) = -1;  % 方波信号的后一半为-1
Label = zeros(1, 2 * step);  % 初始化标签数组

% 生成数据
for i = 1:2 * step / sample
    q = unidrnd(2);  % 随机选择1或2
    if q == 1
        Data(sample * (i - 1) + 1:sample * i) = p1;  % 如果q=1，使用正弦波信号
        Label(sample * (i - 1) + 1:sample * i) = 0;  % 标签为0
    else
        Data(sample * (i - 1) + 1:sample * i) = p2;  % 如果q=2，使用方波信号
        Label(sample * (i - 1) + 1:sample * i) = 1;  % 标签为1
    end
end

% ----------------------训练----------------------
% 初始化输入流
Input = Data(1:step);  % 训练集的输入数据

% 生成目标标签
Target = Label(1:step);

% 掩膜处理
Mask = 2 * unidrnd(2, N, ML) - 3;  % 随机生成掩膜，取值为-1或+1
Input_ex = [];  % 初始化扩展后的输入数据

% 扩展输入数据
for j = 1:N
    for i = 1:step
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i) * Mask(j, :);  % 根据掩膜扩展输入
    end
end

% 对扩展后的输入数据进行归一化处理
UL = max(max(Input_ex));  % 最大值
DL = min(min(Input_ex));  % 最小值
Input_ex = (Input_ex - DL) / (UL - DL) * (Vmax - Vmin) + Vmin;  % 归一化到[Vmin, Vmax]范围

% 忆阻器输出
memout = [];  % 初始化忆阻器输出
G = para.G0;  % 初始化电导G为初始值

% 计算忆阻器输出
for i = 1:length(Input_ex(1, :))
    [memout(:, i), G] = DynamicMemristor(Input_ex(:, i), G, para);  % 更新忆阻器输出
    sprintf('%s', ['train:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])  % 输出当前进度
end

% 状态收集
states = [];  % 初始化状态数组
for i = 1:step
    a = memout(:, ML * (i - 1) + 1:ML * i);  % 获取每步的忆阻器输出
    states(:, i) = a(:);  % 将每步的输出按列收集
end
X = [ones(1, step); states];  % 将状态数据合并成一个矩阵

% 线性回归（利用伪逆矩阵进行回归）
Wout = Target * pinv(X);  % 计算输出权重

% ----------------------测试----------------------
% 初始化输入流
Input = Data(step + 1:end);  % 测试集的输入数据

% 生成目标标签
Target = Label(step + 1:end);

% 掩膜处理
Input_ex = [];  % 初始化扩展后的输入数据
for j = 1:N
    for i = 1:step
        Input_ex(j, (i - 1) * ML + 1:ML * i) = Input(i) * Mask(j, :);  % 根据掩膜扩展输入
    end
end

% 对扩展后的输入数据进行归一化处理
UL = max(max(Input_ex));  % 最大值
DL = min(min(Input_ex));  % 最小值
Input_ex = (Input_ex - DL) / (UL - DL) * (Vmax - Vmin) + Vmin;  % 归一化到[Vmin, Vmax]范围

% 忆阻器输出
memout = [];  % 初始化忆阻器输出
states = [];  % 初始化状态数组
G = para.G0;  % 初始化电导G为初始值

% 计算忆阻器输出
for i = 1:length(Input_ex(1, :))
    [memout(:, i), G] = DynamicMemristor(Input_ex(:, i), G, para);  % 更新忆阻器输出
    sprintf('%s', ['test:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])  % 输出当前进度
end

% 状态收集
for i = 1:step
    a = memout(:, ML * (i - 1) + 1:ML * i);  % 获取每步的忆阻器输出
    states(:, i) = a(:);  % 将每步的输出按列收集
end
X = [ones(1, step); states];  % 将状态数据合并成一个矩阵

% 系统输出
Out = Wout * X;  % 计算输出
NRMSE = sqrt(mean((Out(10:end) - Target(10:end)).^2) / var(Target(10:end)));  % 计算标准化均方根误差（NRMSE）
sprintf('%s', ['NRMSE:', num2str(NRMSE)])  % 输出NRMSE值

% ----------------------绘图----------------------
figure;

% 绘制输入信号
subplot(2, 1, 1);
plot(Input, 'b', 'linewidth', 1);  % 使用蓝色线绘制输入信号
hold on;
plot(Input, '.r');  % 使用红色点绘制输入信号
axis([0, 400, -1.2, 1.2])  % 设置坐标轴范围
ylabel('Input')  % 设置Y轴标签
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置坐标轴字体和大小

% 绘制目标和输出信号
subplot(2, 1, 2);
plot(Target, 'k', 'linewidth', 2);  % 使用黑色线绘制目标信号
hold on;
plot(Out, 'r', 'linewidth', 1);  % 使用红色线绘制输出信号
axis([0, 400, -0.2, 1.2])  % 设置坐标轴范围
str1 = '\color{black}Target';  % 设置目标信号的图例文本
str2 = '\color{red}Output';  % 设置输出信号的图例文本
lg = legend(str1, str2);  % 创建图例
set(lg, 'Orientation', 'horizon');  % 设置图例方向为水平
ylabel('Prediction')  % 设置Y轴标签
xlabel('Time (\tau)')  % 设置X轴标签
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置坐标轴字体和大小
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);  % 设置图形的位置和大小
