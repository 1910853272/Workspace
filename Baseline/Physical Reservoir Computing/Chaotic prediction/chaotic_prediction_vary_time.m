<<<<<<< HEAD
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

% 迭代过程中，对不同的`modu`和`relax`进行测试
for modu = 0:50  % 遍历不同的modu值
    for relax = 0:200  % 遍历不同的relax值
        % 获取设备在该`modu`和`relax`情况下的输出
        current_output = device_sim_vary_time(train_voltage, relax, modu);

        % 线性回归
        a = [];
        states = [];
        for i = 1:ntrain
            a = current_output(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
            states(:, i) = a(:);  % 存储状态向量
        end

        % 线性回归的输入数据（添加偏置项1）
        input = [ones(1, ntrain); states];

        % 使用伪逆法计算回归权重
        weight = target_train * pinv(input);  % 计算回归权重

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

        % 获取设备在该`modu`和`relax`情况下的输出
        current_output = device_sim_vary_time(test_voltage, relax, modu);

        % 预测过程
        a = [];
        states = [];
        for i = 1:ntest
            a = current_output(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
            states(:, i) = a(:);  % 存储状态向量
        end

        % 线性回归的输入数据（添加偏置项1）
        input = [ones(1, ntest); states];

        % 使用回归权重计算预测输出
        output = weight * input;

        % 计算归一化均方根误差（NRMSE）
        NRMSE(modu + 1, relax + 1) = sqrt(mean((output(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));
    end
end
=======
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

% 迭代过程中，对不同的`modu`和`relax`进行测试
for modu = 0:50  % 遍历不同的modu值
    for relax = 0:200  % 遍历不同的relax值
        % 获取设备在该`modu`和`relax`情况下的输出
        current_output = device_sim_vary_time(train_voltage, relax, modu);

        % 线性回归
        a = [];
        states = [];
        for i = 1:ntrain
            a = current_output(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
            states(:, i) = a(:);  % 存储状态向量
        end

        % 线性回归的输入数据（添加偏置项1）
        input = [ones(1, ntrain); states];

        % 使用伪逆法计算回归权重
        weight = target_train * pinv(input);  % 计算回归权重

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

        % 获取设备在该`modu`和`relax`情况下的输出
        current_output = device_sim_vary_time(test_voltage, relax, modu);

        % 预测过程
        a = [];
        states = [];
        for i = 1:ntest
            a = current_output(:, m * (i - 1) + 1:m * i);  % 从设备输出中提取当前状态
            states(:, i) = a(:);  % 存储状态向量
        end

        % 线性回归的输入数据（添加偏置项1）
        input = [ones(1, ntest); states];

        % 使用回归权重计算预测输出
        output = weight * input;

        % 计算归一化均方根误差（NRMSE）
        NRMSE(modu + 1, relax + 1) = sqrt(mean((output(10:end) - target_test(10:end)).^2) / var(target_test(10:end)));
    end
end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
