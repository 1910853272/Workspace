<<<<<<< HEAD
clc;
close all;
clear all;

% 设置相关参数
Vmax = 2;  % 最大电压
Vmin = 0;  % 最小电压
wave_length = 8;  % 每个波形的长度

% 生成不同类型的波形
% Sine波形：正弦波
w1 = sin(pi * 2 * (0:wave_length-1) / wave_length);

% Square波形：方波
w2(1:wave_length / 2) = 1;  % 方波前半部分为1
w2(wave_length / 2 + 1:wave_length) = -1;  % 方波后半部分为-1

type = 2;  % 波形类型（1为正弦波，2为方波）
wave_num = 300;  % 波形的数量

% 生成waveform和wave_label
for i = 1:wave_num
    w = randi(type);  % 随机生成1或2，决定使用哪种波形
    if w == 1
        % 使用正弦波
        waveform(wave_length*(i-1) + 1:wave_length*i) = w1;
        wave_label(wave_length*(i-1) + 1:wave_length*i) = 0;  % 0代表正弦波
    else
        % 使用方波
        waveform(wave_length*(i-1) + 1:wave_length*i) = w2;
        wave_label(wave_length*(i-1) + 1:wave_length*i) = 1;  % 1代表方波
    end
end

% 数据集划分比例和参数设置
ratio = 0.5;  % 训练集占总数据集的比例
n = 25;  % mask的数量
m = 50;  % mask的长度

% 划分训练集和测试集
% 训练集
input_train = waveform(1:round(ratio * wave_num) * wave_length);
target_train = wave_label(1:round(ratio * wave_num) * wave_length);
% 测试集
input_test = waveform(round(ratio * wave_num) * wave_length + 1:wave_num * wave_length);
target_test = wave_label(round(ratio * wave_num) * wave_length + 1:wave_num * wave_length);

% 获取训练集和测试集大小
ntrain = length(input_train);  % 训练集的大小
ntest = length(input_test);  % 测试集的大小

% 生成mask
mask = 2 * randi(2, n, m) - 3;  % 随机生成mask，取值为-1或1

% --------------------- 训练过程 ---------------------
% 处理训练集的mask
train_mask = [];
for j = 1:n
    for i = 1:ntrain
        train_mask(j, (i-1)*m + 1:m*i) = input_train(1, i) * mask(j, :);  % 生成带mask的输入数据
    end
end

train_max = max(max(train_mask));  % 训练集mask的最大值
train_min = min(min(train_mask));  % 训练集mask的最小值

% 电压输入
train_voltage = (train_mask - train_min) / (train_max - train_min) * (Vmax - Vmin) + Vmin;

% 设备的输出（线性和模拟）
current_output_linear = device_linear(train_voltage);  % 线性设备响应
current_output_sim = device_sim(train_voltage);  % 模拟设备响应

% 线性回归
a_linear = [];
a_sim = [];
states_linear = [];
states_sim = [];

for i = 1:ntrain
    a_linear = current_output_linear(:, m*(i-1) + 1:m*i);
    a_sim = current_output_sim(:, m*(i-1) + 1:m*i);
    states_linear(:, i) = a_linear(:);  % 线性设备的状态向量
    states_sim(:, i) = a_sim(:);  % 模拟设备的状态向量
end

input_linear = [ones(1, ntrain); states_linear];  % 线性回归的输入数据（加上偏置项1）
input_sim = [ones(1, ntrain); states_sim];  % 模拟回归的输入数据（加上偏置项1）

% 使用伪逆计算权重
weight_linear = target_train * pinv(input_linear);  % 线性回归权重
weight_sim = target_train * pinv(input_sim);  % 模拟回归权重

% --------------------- 测试过程 ---------------------
% 处理测试集的mask
test_mask = [];
for j = 1:n
    for i = 1:ntest
        test_mask(j, (i-1)*m + 1:m*i) = input_test(1, i) * mask(j, :);  % 生成带mask的输入数据
    end
end

test_max = max(max(test_mask));  % 测试集mask的最大值
test_min = min(min(test_mask));  % 测试集mask的最小值

% 电压输入
test_voltage = (test_mask - test_min) / (test_max - test_min) * (Vmax - Vmin) + Vmin;

% 设备的输出（线性和模拟）
current_output_linear = device_linear(test_voltage);  % 线性设备响应
current_output_sim = device_sim(test_voltage);  % 模拟设备响应

% 预测（混沌预测）
a_linear = [];
a_sim = [];
states_linear = [];
states_sim = [];

for i = 1:ntest
    a_linear = current_output_linear(:, m*(i-1) + 1:m*i);
    a_sim = current_output_sim(:, m*(i-1) + 1:m*i);
    states_linear(:, i) = a_linear(:);  % 线性设备的状态向量
    states_sim(:, i) = a_sim(:);  % 模拟设备的状态向量
end

input_linear = [ones(1, ntest); states_linear];  % 线性回归的输入数据（加上偏置项1）
input_sim = [ones(1, ntest); states_sim];  % 模拟回归的输入数据（加上偏置项1）

% 计算预测结果
output_linear = weight_linear * input_linear;  % 线性回归预测输出
output_sim = weight_sim * input_sim;  % 模拟回归预测输出

% 计算归一化均方根误差（NRMSE）
NRMSE_linear = sqrt(mean((output_linear(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));
NRMSE_sim = sqrt(mean((output_sim(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));

% 打印NRMSE
sprintf('%s', ['NRMSE_linear:', num2str(NRMSE_linear)])
sprintf('%s', ['NRMSE_sim:', num2str(NRMSE_sim)])

% ---------------------- 绘图 ----------------------
% 绘制输入、目标输出和预测输出
figure;

subplot(3, 1, 1);
plot(input_test, 'b', 'linewidth', 1);  % 绘制输入信号
hold on;
plot(input_test, '.r');  % 绘制红点表示输入数据
axis([0, wave_length*50, -1.2, 1.2]);
ylabel('Input');
set(gca, 'FontName', 'Arial', 'FontSize', 20);

subplot(3, 1, 2);
plot(target_test, 'k', 'linewidth', 2);  % 绘制目标输出
hold on;
plot(output_sim, 'r', 'linewidth', 1);  % 绘制模拟回归输出
axis([0, 400, -0.2, 1.2]);
str1 = '\color{black}Target';
str2 = '\color{red}Output sim';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction');
xlabel('Time (\tau)');
set(gca, 'FontName', 'Arial', 'FontSize', 20);

subplot(3, 1, 3);
plot(target_test, 'k', 'linewidth', 2);  % 绘制目标输出
hold on;
plot(output_linear, 'r', 'linewidth', 1);  % 绘制线性回归输出
axis([0, 400, -0.2, 1.2]);
str1 = '\color{black}Target';
str2 = '\color{red}Output linear';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction');
xlabel('Time (\tau)');
set(gca, 'FontName', 'Arial', 'FontSize', 20);

% 调整图形大小
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.6]);
=======
clc;
close all;
clear all;

% 设置相关参数
Vmax = 2;  % 最大电压
Vmin = 0;  % 最小电压
wave_length = 8;  % 每个波形的长度

% 生成不同类型的波形
% Sine波形：正弦波
w1 = sin(pi * 2 * (0:wave_length-1) / wave_length);

% Square波形：方波
w2(1:wave_length / 2) = 1;  % 方波前半部分为1
w2(wave_length / 2 + 1:wave_length) = -1;  % 方波后半部分为-1

type = 2;  % 波形类型（1为正弦波，2为方波）
wave_num = 300;  % 波形的数量

% 生成waveform和wave_label
for i = 1:wave_num
    w = randi(type);  % 随机生成1或2，决定使用哪种波形
    if w == 1
        % 使用正弦波
        waveform(wave_length*(i-1) + 1:wave_length*i) = w1;
        wave_label(wave_length*(i-1) + 1:wave_length*i) = 0;  % 0代表正弦波
    else
        % 使用方波
        waveform(wave_length*(i-1) + 1:wave_length*i) = w2;
        wave_label(wave_length*(i-1) + 1:wave_length*i) = 1;  % 1代表方波
    end
end

% 数据集划分比例和参数设置
ratio = 0.5;  % 训练集占总数据集的比例
n = 25;  % mask的数量
m = 50;  % mask的长度

% 划分训练集和测试集
% 训练集
input_train = waveform(1:round(ratio * wave_num) * wave_length);
target_train = wave_label(1:round(ratio * wave_num) * wave_length);
% 测试集
input_test = waveform(round(ratio * wave_num) * wave_length + 1:wave_num * wave_length);
target_test = wave_label(round(ratio * wave_num) * wave_length + 1:wave_num * wave_length);

% 获取训练集和测试集大小
ntrain = length(input_train);  % 训练集的大小
ntest = length(input_test);  % 测试集的大小

% 生成mask
mask = 2 * randi(2, n, m) - 3;  % 随机生成mask，取值为-1或1

% --------------------- 训练过程 ---------------------
% 处理训练集的mask
train_mask = [];
for j = 1:n
    for i = 1:ntrain
        train_mask(j, (i-1)*m + 1:m*i) = input_train(1, i) * mask(j, :);  % 生成带mask的输入数据
    end
end

train_max = max(max(train_mask));  % 训练集mask的最大值
train_min = min(min(train_mask));  % 训练集mask的最小值

% 电压输入
train_voltage = (train_mask - train_min) / (train_max - train_min) * (Vmax - Vmin) + Vmin;

% 设备的输出（线性和模拟）
current_output_linear = device_linear(train_voltage);  % 线性设备响应
current_output_sim = device_sim(train_voltage);  % 模拟设备响应

% 线性回归
a_linear = [];
a_sim = [];
states_linear = [];
states_sim = [];

for i = 1:ntrain
    a_linear = current_output_linear(:, m*(i-1) + 1:m*i);
    a_sim = current_output_sim(:, m*(i-1) + 1:m*i);
    states_linear(:, i) = a_linear(:);  % 线性设备的状态向量
    states_sim(:, i) = a_sim(:);  % 模拟设备的状态向量
end

input_linear = [ones(1, ntrain); states_linear];  % 线性回归的输入数据（加上偏置项1）
input_sim = [ones(1, ntrain); states_sim];  % 模拟回归的输入数据（加上偏置项1）

% 使用伪逆计算权重
weight_linear = target_train * pinv(input_linear);  % 线性回归权重
weight_sim = target_train * pinv(input_sim);  % 模拟回归权重

% --------------------- 测试过程 ---------------------
% 处理测试集的mask
test_mask = [];
for j = 1:n
    for i = 1:ntest
        test_mask(j, (i-1)*m + 1:m*i) = input_test(1, i) * mask(j, :);  % 生成带mask的输入数据
    end
end

test_max = max(max(test_mask));  % 测试集mask的最大值
test_min = min(min(test_mask));  % 测试集mask的最小值

% 电压输入
test_voltage = (test_mask - test_min) / (test_max - test_min) * (Vmax - Vmin) + Vmin;

% 设备的输出（线性和模拟）
current_output_linear = device_linear(test_voltage);  % 线性设备响应
current_output_sim = device_sim(test_voltage);  % 模拟设备响应

% 预测（混沌预测）
a_linear = [];
a_sim = [];
states_linear = [];
states_sim = [];

for i = 1:ntest
    a_linear = current_output_linear(:, m*(i-1) + 1:m*i);
    a_sim = current_output_sim(:, m*(i-1) + 1:m*i);
    states_linear(:, i) = a_linear(:);  % 线性设备的状态向量
    states_sim(:, i) = a_sim(:);  % 模拟设备的状态向量
end

input_linear = [ones(1, ntest); states_linear];  % 线性回归的输入数据（加上偏置项1）
input_sim = [ones(1, ntest); states_sim];  % 模拟回归的输入数据（加上偏置项1）

% 计算预测结果
output_linear = weight_linear * input_linear;  % 线性回归预测输出
output_sim = weight_sim * input_sim;  % 模拟回归预测输出

% 计算归一化均方根误差（NRMSE）
NRMSE_linear = sqrt(mean((output_linear(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));
NRMSE_sim = sqrt(mean((output_sim(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));

% 打印NRMSE
sprintf('%s', ['NRMSE_linear:', num2str(NRMSE_linear)])
sprintf('%s', ['NRMSE_sim:', num2str(NRMSE_sim)])

% ---------------------- 绘图 ----------------------
% 绘制输入、目标输出和预测输出
figure;

subplot(3, 1, 1);
plot(input_test, 'b', 'linewidth', 1);  % 绘制输入信号
hold on;
plot(input_test, '.r');  % 绘制红点表示输入数据
axis([0, wave_length*50, -1.2, 1.2]);
ylabel('Input');
set(gca, 'FontName', 'Arial', 'FontSize', 20);

subplot(3, 1, 2);
plot(target_test, 'k', 'linewidth', 2);  % 绘制目标输出
hold on;
plot(output_sim, 'r', 'linewidth', 1);  % 绘制模拟回归输出
axis([0, 400, -0.2, 1.2]);
str1 = '\color{black}Target';
str2 = '\color{red}Output sim';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction');
xlabel('Time (\tau)');
set(gca, 'FontName', 'Arial', 'FontSize', 20);

subplot(3, 1, 3);
plot(target_test, 'k', 'linewidth', 2);  % 绘制目标输出
hold on;
plot(output_linear, 'r', 'linewidth', 1);  % 绘制线性回归输出
axis([0, 400, -0.2, 1.2]);
str1 = '\color{black}Target';
str2 = '\color{red}Output linear';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction');
xlabel('Time (\tau)');
set(gca, 'FontName', 'Arial', 'FontSize', 20);

% 调整图形大小
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.6]);
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
