clc;
close all;
clear all;

% 设置相关参数
Vmax = 2;  % 最大电压
Vmin = 0;  % 最小电压
datasize = 2000;  % Henon数据集的大小
% 生成Henon映射的数据
[x, y] = Henon(datasize + 1);  % 生成Henon映射序列，输出x和y序列

ratio = 0.5;  % 训练集占总数据集的比例
n = 25;  % mask的数量
m = 50;  % mask的长度

% 划分训练集和测试集
% 训练集
input_train = x(1:round(ratio * datasize));  % 训练集输入
target_train = x(2:round(ratio * datasize) + 1);  % 训练集目标输出
% 测试集
input_test = x(round(ratio * datasize) + 1:datasize);  % 测试集输入
target_test = x(round(ratio * datasize) + 2:datasize + 1);  % 测试集目标输出

ntrain = length(input_train);  % 训练集的大小
ntest = length(input_test);  % 测试集的大小

% 生成随机mask
mask = 2 * randi(2, n, m) - 3;  % 生成n个随机mask，每个mask的长度为m，取值为-1或1

% --------------------- 训练过程 ---------------------
% 训练集的mask处理
train_mask = [];
for j = 1:n
    for i = 1:ntrain
        train_mask(j, (i - 1) * m + 1:m * i) = input_train(1, i) * mask(j, :);  % 对每个输入信号应用mask
    end
end

train_max = max(max(train_mask));  % 训练集mask的最大值
train_min = min(min(train_mask));  % 训练集mask的最小值

% 电压输入归一化
train_voltage = (train_mask - train_min) / (train_max - train_min) * (Vmax - Vmin) + Vmin;

% 计算设备的输出（线性与模拟设备响应）
current_output_linear = device_linear(train_voltage);  % 线性设备的输出
current_output_sim = device_sim(train_voltage);  % 模拟设备的输出

% 线性回归
a_linear = [];
a_sim = [];
states_linear = [];
states_sim = [];

for i = 1:ntrain
    a_linear = current_output_linear(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
    a_sim = current_output_sim(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
    states_linear(:, i) = a_linear(:);  % 存储线性设备的状态向量
    states_sim(:, i) = a_sim(:);  % 存储模拟设备的状态向量
end

% 线性回归输入数据（添加偏置项1）
input_linear = [ones(1, ntrain); states_linear];
input_sim = [ones(1, ntrain); states_sim];

% 使用伪逆法计算回归权重
weight_linear = target_train * pinv(input_linear);  % 线性回归权重
weight_sim = target_train * pinv(input_sim);  % 模拟回归权重

% --------------------- 测试过程 ---------------------
% 测试集的mask处理
test_mask = [];
for j = 1:n
    for i = 1:ntest
        test_mask(j, (i - 1) * m + 1:m * i) = input_test(1, i) * mask(j, :);  % 对每个输入信号应用mask
    end
end

test_max = max(max(test_mask));  % 测试集mask的最大值
test_min = min(min(test_mask));  % 测试集mask的最小值

% 电压输入归一化
test_voltage = (test_mask - test_min) / (test_max - test_min) * (Vmax - Vmin) + Vmin;

% 计算设备的输出（线性与模拟设备响应）
current_output_linear = device_linear(test_voltage);  % 线性设备的输出
current_output_sim = device_sim(test_voltage);  % 模拟设备的输出

% 预测（混沌预测）
a_linear = [];
a_sim = [];
states_linear = [];
states_sim = [];

for i = 1:ntest
    a_linear = current_output_linear(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
    a_sim = current_output_sim(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
    states_linear(:, i) = a_linear(:);  % 存储线性设备的状态向量
    states_sim(:, i) = a_sim(:);  % 存储模拟设备的状态向量
end

% 线性回归输入数据（添加偏置项1）
input_linear = [ones(1, ntest); states_linear];
input_sim = [ones(1, ntest); states_sim];

% 计算预测输出
output_linear = weight_linear * input_linear;  % 线性回归预测结果
output_sim = weight_sim * input_sim;  % 模拟回归预测结果

% 计算归一化均方根误差（NRMSE）
NRMSE_linear = sqrt(mean((output_linear(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));
NRMSE_sim = sqrt(mean((output_sim(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));

% 打印NRMSE值
sprintf('%s', ['NRMSE_linear:', num2str(NRMSE_linear)])
sprintf('%s', ['NRMSE_sim:', num2str(NRMSE_sim)])

% ---------------------- 绘图 ----------------------
% 绘制时间序列图
figure(1);
subplot(2, 1, 1);
plot(target_test(1:200), 'k', 'linewidth', 2);  % 绘制目标测试数据
hold on;
plot(output_linear(1:200), 'r', 'linewidth', 1);  % 绘制线性回归预测结果
axis([0, 200, -2, 2]);  % 设置坐标轴范围
str1 = '\color{black}Target';  % 标签：目标数据
str2 = '\color{red}Output_linear';  % 标签：线性回归输出
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon', 'box', 'off');  % 设置图例
ylabel('Prediction');  % 设置y轴标签
xlabel('Time (\tau)');  % 设置x轴标签
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置字体和字号
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);  % 设置图形窗口大小

subplot(2, 1, 2);
plot(target_test(1:200), 'k', 'linewidth', 2);  % 绘制目标测试数据
hold on;
plot(output_sim(1:200), 'r', 'linewidth', 1);  % 绘制模拟回归预测结果
axis([0, 200, -2, 2]);  % 设置坐标轴范围
str1 = '\color{black}Target';  % 标签：目标数据
str2 = '\color{red}Output sim';  % 标签：模拟回归输出
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon', 'box', 'off');  % 设置图例
ylabel('Prediction');  % 设置y轴标签
xlabel('Time (\tau)');  % 设置x轴标签
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置字体和字号
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.6]);  % 设置图形窗口大小

% 绘制2D映射图
figure(2);
subplot(1, 2, 1);
plot(target_test(2:end), 0.3 * target_test(1:end-1), '.k', 'markersize', 12);  % 绘制目标测试数据的2D图
hold on;
plot(output_linear(2:end), 0.3 * output_linear(1:end-1), '.r', 'markersize', 12);  % 绘制线性回归预测结果的2D图
str1 = '\color{black}Target';  % 标签：目标数据
str2 = '\color{red}Output linear';  % 标签：线性回归输出
lg = legend(str1, str2);
set(lg, 'box', 'off');
ylabel('{\ity} (n)');  % 设置y轴标签
xlabel('{\itx} (n)');  % 设置x轴标签
axis([-2, 2, -0.4, 0.4]);  % 设置坐标轴范围
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置字体和字号
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.3, 0.45]);  % 设置图形窗口大小

subplot(1, 2, 2);
plot(target_test(2:end), 0.3 * target_test(1:end-1), '.k', 'markersize', 12);  % 绘制目标测试数据的2D图
hold on;
plot(output_sim(2:end), 0.3 * output_sim(1:end-1), '.r', 'markersize', 12);  % 绘制模拟回归预测结果的2D图
str1 = '\color{black}Target';  % 标签：目标数据
str2 = '\color{red}Output sim';  % 标签：模拟回归输出
lg = legend(str1, str2);
set(lg, 'box', 'off');
ylabel('{\ity} (n)');  % 设置y轴标签
xlabel('{\itx} (n)');  % 设置x轴标签
axis([-2, 2, -0.4, 0.4]);  % 设置坐标轴范围
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置字体和字号
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.45]);  % 设置图形窗口大小
