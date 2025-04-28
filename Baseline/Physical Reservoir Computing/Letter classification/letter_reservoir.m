<<<<<<< HEAD
clc;
clear;
close all;

% 从Excel读取输入的字母数据，并进行转置以便于后续处理
letter = xlsread('letter.xlsx')';
target = diag(ones(20, 1));  % 定义目标向量（单位矩阵）

larger = 10;  % 数据集扩展因子
[row, col] = size(letter);  % 获取输入数据的行列数

% 通过添加椒盐噪声来创建更大的数据集
t = zeros(row, col, larger - 1);
for i = 1:larger - 1
    t = imnoise(letter(1:row, :), 'salt & pepper', 0.05);  % 添加椒盐噪声
    letter(row * i + 1:row * (i + 1), :) = t();  % 将噪声数据添加到字母数据中
end

% 扩展目标矩阵以匹配增大的数据集
for i = 1:larger - 1
    target(:, col * i + 1:col * (i + 1)) = target(:, 1:col);
end

% 定义每个输入的状态向量（根据问题定义）
state = [0.0, 8.314, 7.603, 16.859, 6.917, 16.816, 14.823, 24.789,...
    7.115, 16.177, 14.101, 25.118, 15.333, 23.401, 22.758, 32.286];
inputs = zeros(5, 20);  % 初始化输入矩阵

% 将字母数据映射为状态向量中的值，构建新的输入矩阵
for i = 1:20 * larger
    for j = 1:5
       order = letter(i, 4 * j - 3) * 8 + letter(i, 4 * j - 2) * 4 + ...
               letter(i, 4 * j - 1) * 2 + letter(i, 4 * j) + 1;
       inputs(j, i) = state(order);  % 将状态值赋给输入矩阵
    end
end

% 对输入数据进行归一化处理
inputs_max = max(max(inputs));
inputs_min = min(min(inputs));
inputs = (inputs - inputs_min) / (inputs_max - inputs_min);

% 根据扩展因子增大数据集的行列数
row = row * larger;
col = col * larger;

% 对输入数据和目标数据进行列随机打乱
colrank = randperm(col);
new_inputs = inputs(:, colrank);
new_target = target(:, colrank);

% 将数据集划分为训练集和测试集（70%为训练集，30%为测试集）
train_input = new_inputs(:, 1:round(col * 0.7));
train_target = new_target(:, 1:round(col * 0.7));
test_input = new_inputs(:, round(col * 0.7) + 1:end);
test_target = new_target(:, round(col * 0.7) + 1:end);

% 获取训练集和测试集的大小
[mtrain, ntrain] = size(train_target);
[mtest, ntest] = size(test_target);

% 定义神经网络架构（5个输入神经元，20个隐藏神经元）
arch = [5, 20];
nlayer = length(arch);  % 神经网络的层数

% 定义超参数
mini_batch_size = 100;  % 小批量大小
max_epochs = 2000;  % 最大训练周期
max_accu_epoch = max_epochs;  % 初始最大准确率训练周期
zeta = 5;  % 学习率
threshold = 0;  % 停止条件阈值

% 初始化每一层的权重和偏置
weight = cell(1, nlayer);
bias = cell(1, nlayer);
nabla_weight = cell(1, nlayer);
nabla_bias = cell(1, nlayer);

% 初始化每一层的激活值和预激活值
a = cell(1, nlayer);
z = cell(1, nlayer);

% 初始化测试集上的激活值和预激活值
at = cell(1, nlayer);
zt = cell(1, nlayer);

% 随机初始化每一层的权重和偏置
for in = 2:nlayer
    weight{in} = rand(arch(1, in), arch(1, in - 1)) * 2 - 1;  % 权重随机初始化
    bias{in} = rand(arch(1, in), 1);  % 偏置随机初始化
    nabla_weight{in} = rand(arch(1, in), arch(1, in - 1));  % 初始化权重梯度
    nabla_bias{in} = rand(arch(1, in), 1);  % 初始化偏置梯度
end

% 初始化激活值和预激活值矩阵
for in = 1:nlayer
    a{in} = zeros(arch(in), mini_batch_size);
    z{in} = zeros(arch(in), mini_batch_size);
    at{in} = zeros(arch(in), ntest);
    zt{in} = zeros(arch(in), ntest);
end

time = 0:max_epochs;
accuracy = zeros(max_epochs + 1, 1);  % 准确率数组

s = 0;

% 在测试集上进行前向传播
at{1} = test_input;
for in = 2:nlayer
    wt = weight{in};
    bt = bias{in};
    ixt = at{in - 1};
    izt = wt * ixt;
    for im = 1:ntest
        izt(:, im) = izt(:, im) + bt;  % 添加偏置
    end
    if in == nlayer
        zt{in} = izt;
        exat = exp(izt);  % 使用Softmax激活函数
        stant = sum(exat);
        for im = 1:ntest
            at{in}(:, im) = exat(:, im) / stant(1, im);
        end
    else
        zt{in} = izt;
        at{in} = relu(izt);  % 隐藏层使用ReLU激活函数
    end
end

% 计算测试集上的准确率
for m = 1:ntest
    [dx, wz] = max(at{nlayer}(:, m));
    [dx2, wz2] = max(test_target(:, m));
    if wz == wz2
        s = s + 1;
    end
end
accuracy(1, 1) = s / ntest;  % 输出初始的准确率

% 训练过程
s = 0;
for ip = 1:max_epochs
    pos = randi(ntrain + 1 - mini_batch_size);  % 随机选择一个小批量
    input = train_input(:, pos:pos + mini_batch_size - 1);
    output = train_target(:, pos:pos + mini_batch_size - 1);

    a{1} = input;  % 设置输入层的激活值
    if (accuracy(ip, 1) ~= 1)
        % 前向传播
        for in = 2:nlayer
            w = weight{in};
            b = bias{in};
            ix = a{in - 1};
            iz = w * ix;  % 计算预激活值

            for im = 1:mini_batch_size
                iz(:, im) = iz(:, im) + b;  % 添加偏置
            end
            z{in} = iz;
            if in == nlayer
                exa = exp(iz);
                stan = sum(exa);
                for m = 1:mini_batch_size
                    a{in}(:, m) = exa(:, m) / stan(1, m);  % Softmax激活
                end
            else
                a{in} = relu(iz);  % 隐藏层使用ReLU激活
            end
        end

        % 反向传播
        delta = a{nlayer} - output;  % 输出层的误差
        nabla_bias{nlayer} = mean(delta, 2);  % 计算偏置的梯度
        nabla_weight{nlayer} = (delta * (a{nlayer - 1})') / mini_batch_size;  % 计算权重的梯度

        % 对于隐藏层进行反向传播
        if nlayer >= 3
            for in = nlayer - 1:-1:2
                delta = weight{in + 1}' * delta .* relu_prime(z{in});  % 计算隐藏层的误差
                nabla_bias{in} = mean(delta, 2);  % 计算偏置梯度
                nabla_weight{in} = (delta * a{in - 1}') / mini_batch_size;  % 计算权重梯度
            end
        end

        % 更新权重和偏置
        for in = 2:nlayer
            weight{in} = weight{in} - zeta * nabla_weight{in};
            bias{in} = bias{in} - zeta * nabla_bias{in};
        end
    end

    % 测试过程
    at{1} = test_input;
    for in = 2:nlayer
        wt = weight{in};
        bt = bias{in};
        ixt = at{in - 1};
        izt = wt * ixt;

        for im = 1:ntest
            izt(:, im) = izt(:, im) + bt;  % 添加偏置
        end
        if in == nlayer
            zt{in} = izt;
            exat = exp(izt);  % Softmax激活
            stant = sum(exat);
            for im = 1:ntest
                at{in}(:, im) = exat(:, im) / stant(1, im);
            end
        else
            zt{in} = izt;
            at{in} = relu(izt);  % ReLU激活
        end
    end

    % 计算测试集上的准确率
    for m = 1:ntest
        [dx, wz] = max(at{nlayer}(:, m));
        [dx2, wz2] = max(test_target(:, m));
        if wz == wz2
            s = s + 1;
        end
    end
    accuracy(ip + 1, 1) = s / ntest;  % 更新准确率
    if (accuracy(ip + 1, 1) == 1) && (ip < max_accu_epoch)
        max_accu_epoch = ip;  % 如果达到最大准确率，更新停止训练的周期
    end
    fprintf('%d  %.2f\n', ip, accuracy(ip + 1, 1) * 100);  % 打印每轮的准确率

    s = 0;
end

% 输出训练过程中的最大准确率对应的周期和最终准确率
disp(max_accu_epoch);
disp(accuracy(201, 1));

% 绘制准确率曲线
figure, plot(time, accuracy, 'k', 'linewidth', 3);
axis([0 max_epochs, 0, 1]);
xlabel('epoch');
ylabel('accuracy');
=======
clc;
clear;
close all;

% 从Excel读取输入的字母数据，并进行转置以便于后续处理
letter = xlsread('letter.xlsx')';
target = diag(ones(20, 1));  % 定义目标向量（单位矩阵）

larger = 10;  % 数据集扩展因子
[row, col] = size(letter);  % 获取输入数据的行列数

% 通过添加椒盐噪声来创建更大的数据集
t = zeros(row, col, larger - 1);
for i = 1:larger - 1
    t = imnoise(letter(1:row, :), 'salt & pepper', 0.05);  % 添加椒盐噪声
    letter(row * i + 1:row * (i + 1), :) = t();  % 将噪声数据添加到字母数据中
end

% 扩展目标矩阵以匹配增大的数据集
for i = 1:larger - 1
    target(:, col * i + 1:col * (i + 1)) = target(:, 1:col);
end

% 定义每个输入的状态向量（根据问题定义）
state = [0.0, 8.314, 7.603, 16.859, 6.917, 16.816, 14.823, 24.789,...
    7.115, 16.177, 14.101, 25.118, 15.333, 23.401, 22.758, 32.286];
inputs = zeros(5, 20);  % 初始化输入矩阵

% 将字母数据映射为状态向量中的值，构建新的输入矩阵
for i = 1:20 * larger
    for j = 1:5
       order = letter(i, 4 * j - 3) * 8 + letter(i, 4 * j - 2) * 4 + ...
               letter(i, 4 * j - 1) * 2 + letter(i, 4 * j) + 1;
       inputs(j, i) = state(order);  % 将状态值赋给输入矩阵
    end
end

% 对输入数据进行归一化处理
inputs_max = max(max(inputs));
inputs_min = min(min(inputs));
inputs = (inputs - inputs_min) / (inputs_max - inputs_min);

% 根据扩展因子增大数据集的行列数
row = row * larger;
col = col * larger;

% 对输入数据和目标数据进行列随机打乱
colrank = randperm(col);
new_inputs = inputs(:, colrank);
new_target = target(:, colrank);

% 将数据集划分为训练集和测试集（70%为训练集，30%为测试集）
train_input = new_inputs(:, 1:round(col * 0.7));
train_target = new_target(:, 1:round(col * 0.7));
test_input = new_inputs(:, round(col * 0.7) + 1:end);
test_target = new_target(:, round(col * 0.7) + 1:end);

% 获取训练集和测试集的大小
[mtrain, ntrain] = size(train_target);
[mtest, ntest] = size(test_target);

% 定义神经网络架构（5个输入神经元，20个隐藏神经元）
arch = [5, 20];
nlayer = length(arch);  % 神经网络的层数

% 定义超参数
mini_batch_size = 100;  % 小批量大小
max_epochs = 2000;  % 最大训练周期
max_accu_epoch = max_epochs;  % 初始最大准确率训练周期
zeta = 5;  % 学习率
threshold = 0;  % 停止条件阈值

% 初始化每一层的权重和偏置
weight = cell(1, nlayer);
bias = cell(1, nlayer);
nabla_weight = cell(1, nlayer);
nabla_bias = cell(1, nlayer);

% 初始化每一层的激活值和预激活值
a = cell(1, nlayer);
z = cell(1, nlayer);

% 初始化测试集上的激活值和预激活值
at = cell(1, nlayer);
zt = cell(1, nlayer);

% 随机初始化每一层的权重和偏置
for in = 2:nlayer
    weight{in} = rand(arch(1, in), arch(1, in - 1)) * 2 - 1;  % 权重随机初始化
    bias{in} = rand(arch(1, in), 1);  % 偏置随机初始化
    nabla_weight{in} = rand(arch(1, in), arch(1, in - 1));  % 初始化权重梯度
    nabla_bias{in} = rand(arch(1, in), 1);  % 初始化偏置梯度
end

% 初始化激活值和预激活值矩阵
for in = 1:nlayer
    a{in} = zeros(arch(in), mini_batch_size);
    z{in} = zeros(arch(in), mini_batch_size);
    at{in} = zeros(arch(in), ntest);
    zt{in} = zeros(arch(in), ntest);
end

time = 0:max_epochs;
accuracy = zeros(max_epochs + 1, 1);  % 准确率数组

s = 0;

% 在测试集上进行前向传播
at{1} = test_input;
for in = 2:nlayer
    wt = weight{in};
    bt = bias{in};
    ixt = at{in - 1};
    izt = wt * ixt;
    for im = 1:ntest
        izt(:, im) = izt(:, im) + bt;  % 添加偏置
    end
    if in == nlayer
        zt{in} = izt;
        exat = exp(izt);  % 使用Softmax激活函数
        stant = sum(exat);
        for im = 1:ntest
            at{in}(:, im) = exat(:, im) / stant(1, im);
        end
    else
        zt{in} = izt;
        at{in} = relu(izt);  % 隐藏层使用ReLU激活函数
    end
end

% 计算测试集上的准确率
for m = 1:ntest
    [dx, wz] = max(at{nlayer}(:, m));
    [dx2, wz2] = max(test_target(:, m));
    if wz == wz2
        s = s + 1;
    end
end
accuracy(1, 1) = s / ntest;  % 输出初始的准确率

% 训练过程
s = 0;
for ip = 1:max_epochs
    pos = randi(ntrain + 1 - mini_batch_size);  % 随机选择一个小批量
    input = train_input(:, pos:pos + mini_batch_size - 1);
    output = train_target(:, pos:pos + mini_batch_size - 1);

    a{1} = input;  % 设置输入层的激活值
    if (accuracy(ip, 1) ~= 1)
        % 前向传播
        for in = 2:nlayer
            w = weight{in};
            b = bias{in};
            ix = a{in - 1};
            iz = w * ix;  % 计算预激活值

            for im = 1:mini_batch_size
                iz(:, im) = iz(:, im) + b;  % 添加偏置
            end
            z{in} = iz;
            if in == nlayer
                exa = exp(iz);
                stan = sum(exa);
                for m = 1:mini_batch_size
                    a{in}(:, m) = exa(:, m) / stan(1, m);  % Softmax激活
                end
            else
                a{in} = relu(iz);  % 隐藏层使用ReLU激活
            end
        end

        % 反向传播
        delta = a{nlayer} - output;  % 输出层的误差
        nabla_bias{nlayer} = mean(delta, 2);  % 计算偏置的梯度
        nabla_weight{nlayer} = (delta * (a{nlayer - 1})') / mini_batch_size;  % 计算权重的梯度

        % 对于隐藏层进行反向传播
        if nlayer >= 3
            for in = nlayer - 1:-1:2
                delta = weight{in + 1}' * delta .* relu_prime(z{in});  % 计算隐藏层的误差
                nabla_bias{in} = mean(delta, 2);  % 计算偏置梯度
                nabla_weight{in} = (delta * a{in - 1}') / mini_batch_size;  % 计算权重梯度
            end
        end

        % 更新权重和偏置
        for in = 2:nlayer
            weight{in} = weight{in} - zeta * nabla_weight{in};
            bias{in} = bias{in} - zeta * nabla_bias{in};
        end
    end

    % 测试过程
    at{1} = test_input;
    for in = 2:nlayer
        wt = weight{in};
        bt = bias{in};
        ixt = at{in - 1};
        izt = wt * ixt;

        for im = 1:ntest
            izt(:, im) = izt(:, im) + bt;  % 添加偏置
        end
        if in == nlayer
            zt{in} = izt;
            exat = exp(izt);  % Softmax激活
            stant = sum(exat);
            for im = 1:ntest
                at{in}(:, im) = exat(:, im) / stant(1, im);
            end
        else
            zt{in} = izt;
            at{in} = relu(izt);  % ReLU激活
        end
    end

    % 计算测试集上的准确率
    for m = 1:ntest
        [dx, wz] = max(at{nlayer}(:, m));
        [dx2, wz2] = max(test_target(:, m));
        if wz == wz2
            s = s + 1;
        end
    end
    accuracy(ip + 1, 1) = s / ntest;  % 更新准确率
    if (accuracy(ip + 1, 1) == 1) && (ip < max_accu_epoch)
        max_accu_epoch = ip;  % 如果达到最大准确率，更新停止训练的周期
    end
    fprintf('%d  %.2f\n', ip, accuracy(ip + 1, 1) * 100);  % 打印每轮的准确率

    s = 0;
end

% 输出训练过程中的最大准确率对应的周期和最终准确率
disp(max_accu_epoch);
disp(accuracy(201, 1));

% 绘制准确率曲线
figure, plot(time, accuracy, 'k', 'linewidth', 3);
axis([0 max_epochs, 0, 1]);
xlabel('epoch');
ylabel('accuracy');
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
