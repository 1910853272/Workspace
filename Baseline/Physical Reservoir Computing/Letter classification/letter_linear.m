<<<<<<< HEAD
clc
clear
close all;

% 从 Excel 文件中读取字母数据，并进行转置
letter = xlsread('letter.xlsx')';  % 字母数据是 5x20 的矩阵，每列代表一个字母
target = diag(ones(20,1));  % 创建一个 20x20 的单位矩阵作为目标输出，表示20个类别

% 以下代码段用于扩展数据集（注释掉了）
% larger=10;
% [row, col] = size(letter);  % 获取字母数据的行列数
% % 添加椒盐噪声
% t = zeros(row, col, larger-1);
% for i = 1:larger-1
%     t = imnoise(letter(1:row,:), 'salt & pepper', 0.05);  % 为数据添加椒盐噪声
%     letter(row*i + 1:row*(i+1), :) = t();  % 将噪声数据合并
% end

% % 扩展目标矩阵，以匹配噪声数据的数量
% for i = 1:larger-1
%     target(:, col*i + 1:col*(i+1)) = target(:, 1:col);
% end

% 将字母数据每列按每个字母的4个像素点进行汇总，得到每个字母的状态值
inputs = zeros(5, 20);  % 初始化输入数据矩阵
for i = 1:20
    for j = 1:5
       order = letter(i, 4*j - 3) + letter(i, 4*j - 2) + letter(i, 4*j - 1) + letter(i, 4*j);  % 计算每个字母的状态值
       inputs(j, i) = order;  % 存储计算的状态值
    end
end

% 数据归一化处理，将数据归一化到 [0, 1] 的范围
inputs_max = max(max(inputs));  % 获取输入数据的最大值
inputs_min = min(min(inputs));  % 获取输入数据的最小值
inputs = (inputs - inputs_min) / (inputs_max - inputs_min);  % 归一化

% 将训练集和测试集设置为相同的数据集（可以根据需要调整）
train_input = inputs;  % 训练输入数据
train_target = target;  % 训练目标数据
test_input = inputs;  % 测试输入数据
test_target = target;  % 测试目标数据
[mtrain, ntrain] = size(train_target);  % 获取训练集的大小
[mtest, ntest] = size(test_target);  % 获取测试集的大小

% 设置神经网络的结构（5个输入神经元，20个隐藏神经元）
arch = [5, 20];
nlayer = length(arch);  % 网络层数

% 设置训练的超参数
mini_batch_size = 20;  % 每个小批量的样本数
max_epochs = 200;  % 最大训练周期数
max_accu_epoch = max_epochs;  % 记录达到最高准确率时的训练周期
zeta = 8.1;  % 学习率
threshold = 0;  % 停止条件的阈值

% 初始化神经网络的权重和偏置
weight = cell(1, nlayer);
bias = cell(1, nlayer);
nabla_weight = cell(1, nlayer);  % 权重梯度
nabla_bias = cell(1, nlayer);  % 偏置梯度

% 初始化每一层的激活值和预激活值
a = cell(1, nlayer);
z = cell(1, nlayer);
at = cell(1, nlayer);  % 测试集上的激活值
zt = cell(1, nlayer);  % 测试集上的预激活值

% 随机初始化权重和偏置
for in = 2:nlayer
    weight{in} = rand(arch(1, in), arch(1, in - 1)) * 2 - 1;  % 权重随机初始化
    bias{in} = rand(arch(1, in), 1);  % 偏置随机初始化
    nabla_weight{in} = rand(arch(1, in), arch(1, in - 1));  % 初始化权重梯度
    nabla_bias{in} = rand(arch(1, in), 1);  % 初始化偏置梯度
end

% 初始化每一层的激活值和预激活值矩阵
for in = 1:nlayer
    a{in} = zeros(arch(in), mini_batch_size);  % 激活值矩阵
    z{in} = zeros(arch(in), mini_batch_size);  % 预激活值矩阵
    at{in} = zeros(arch(in), ntest);  % 测试集激活值矩阵
    zt{in} = zeros(arch(in), ntest);  % 测试集预激活值矩阵
end

% 进行前向传播，计算测试集上的准确率
at{1} = test_input;  % 将测试集的输入数据赋给激活值
for in = 2:nlayer
    wt = weight{in};
    bt = bias{in};
    ixt = at{in - 1};
    izt = wt * ixt;  % 计算预激活值

    for im = 1:ntest
        izt(:, im) = izt(:, im) + bt;  % 添加偏置
    end

    if in == nlayer  % 输出层使用 Softmax 激活
        zt{in} = izt;
        exat = exp(izt);  % 计算 Softmax 输出
        stant = sum(exat);  % 计算每个样本的总和
        for im = 1:ntest
            at{in}(:, im) = exat(:, im) / stant(1, im);  % 归一化为概率
        end
    else
        zt{in} = izt;
        at{in} = relu(izt);  % 隐藏层使用 ReLU 激活函数
    end
end

% 计算测试集上的准确率
for m = 1:ntest
    [dx, wz] = max(at{nlayer}(:, m));  % 获取预测的标签
    [dx2, wz2] = max(test_target(:, m));  % 获取真实的标签
    if wz == wz2
        s = s + 1;  % 如果预测正确，计数加1
    end
end
accuracy(1, 1) = s / ntest;  % 计算初始准确率

% 开始训练
s = 0;
for ip = 1:max_epochs
    input = train_input;  % 训练集输入
    output = train_target;  % 训练集目标

    a{1} = input;  % 设置输入层的激活值
    if accuracy(ip, 1) ~= 1  % 如果准确率没有达到100%，继续训练
        for in = 2:nlayer
            w = weight{in};
            b = bias{in};
            ix = a{in - 1};
            iz = w * ix;  % 计算预激活值
            for im = 1:mini_batch_size
                iz(:, im) = iz(:, im) + b;  % 添加偏置
            end
            z{in} = iz;
            if in == nlayer  % 输出层使用 Softmax 激活
                exa = exp(iz);
                stan = sum(exa);  % 计算每个样本的总和
                for m = 1:mini_batch_size
                    a{in}(:, m) = exa(:, m) / stan(1, m);  % 归一化为概率
                end
            else
                a{in} = relu(iz);  % 隐藏层使用 ReLU 激活
            end
        end

        % 计算输出层的误差并进行反向传播
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
            weight{in} = weight{in} - zeta * nabla_weight{in};  % 更新权重
            bias{in} = bias{in} - zeta * nabla_bias{in};  % 更新偏置
        end
    end

    % 计算测试集上的准确率
    at{1} = test_input;  % 测试集输入
    for in = 2:nlayer
        wt = weight{in};
        bt = bias{in};
        ixt = at{in - 1};
        izt = wt * ixt;  % 计算预激活值

        for im = 1:ntest
            izt(:, im) = izt(:, im) + bt;  % 添加偏置
        end
        if in == nlayer  % 输出层使用 Softmax 激活
            zt{in} = izt;
            exat = exp(izt);
            stant = sum(exat);
            for im = 1:ntest
                at{in}(:, im) = exat(:, im) / stant(1, im);  % 归一化为概率
            end
        else
            zt{in} = izt;
            at{in} = relu(izt);  % 隐藏层使用 ReLU 激活
        end
    end

    % 计算准确率
    for m = 1:ntest
        [dx, wz] = max(at{nlayer}(:, m));  % 获取预测的标签
        [dx2, wz2] = max(test_target(:, m));  % 获取真实标签
        if wz == wz2
            s = s + 1;  % 如果预测正确，计数加1
        end
    end
    accuracy(ip + 1, 1) = s / ntest;  % 计算准确率
    if (accuracy(ip + 1, 1) == 1) && (ip < max_accu_epoch)
        max_accu_epoch = ip;  % 如果准确率为100%，记录当前的训练周期
    end
    fprintf('%d  %.2f\n', ip, accuracy(ip + 1, 1) * 100);  % 输出当前训练周期的准确率
    s = 0;  % 重置正确分类数量
end

% 输出训练结果
disp(max_accu_epoch);  % 输出达到最高准确率时的训练周期
disp(accuracy(201, 1));  % 输出第 200 轮的准确率

% 绘制训练过程中的准确率曲线
figure, plot(time, accuracy, 'k', 'linewidth', 3);
axis([0 max_epochs, 0, 1]);  % 设置图形的坐标范围
xlabel('epoch');  % x轴标签
ylabel('accuracy');  % y轴标签
=======
clc
clear
close all;

% 从 Excel 文件中读取字母数据，并进行转置
letter = xlsread('letter.xlsx')';  % 字母数据是 5x20 的矩阵，每列代表一个字母
target = diag(ones(20,1));  % 创建一个 20x20 的单位矩阵作为目标输出，表示20个类别

% 以下代码段用于扩展数据集（注释掉了）
% larger=10;
% [row, col] = size(letter);  % 获取字母数据的行列数
% % 添加椒盐噪声
% t = zeros(row, col, larger-1);
% for i = 1:larger-1
%     t = imnoise(letter(1:row,:), 'salt & pepper', 0.05);  % 为数据添加椒盐噪声
%     letter(row*i + 1:row*(i+1), :) = t();  % 将噪声数据合并
% end

% % 扩展目标矩阵，以匹配噪声数据的数量
% for i = 1:larger-1
%     target(:, col*i + 1:col*(i+1)) = target(:, 1:col);
% end

% 将字母数据每列按每个字母的4个像素点进行汇总，得到每个字母的状态值
inputs = zeros(5, 20);  % 初始化输入数据矩阵
for i = 1:20
    for j = 1:5
       order = letter(i, 4*j - 3) + letter(i, 4*j - 2) + letter(i, 4*j - 1) + letter(i, 4*j);  % 计算每个字母的状态值
       inputs(j, i) = order;  % 存储计算的状态值
    end
end

% 数据归一化处理，将数据归一化到 [0, 1] 的范围
inputs_max = max(max(inputs));  % 获取输入数据的最大值
inputs_min = min(min(inputs));  % 获取输入数据的最小值
inputs = (inputs - inputs_min) / (inputs_max - inputs_min);  % 归一化

% 将训练集和测试集设置为相同的数据集（可以根据需要调整）
train_input = inputs;  % 训练输入数据
train_target = target;  % 训练目标数据
test_input = inputs;  % 测试输入数据
test_target = target;  % 测试目标数据
[mtrain, ntrain] = size(train_target);  % 获取训练集的大小
[mtest, ntest] = size(test_target);  % 获取测试集的大小

% 设置神经网络的结构（5个输入神经元，20个隐藏神经元）
arch = [5, 20];
nlayer = length(arch);  % 网络层数

% 设置训练的超参数
mini_batch_size = 20;  % 每个小批量的样本数
max_epochs = 200;  % 最大训练周期数
max_accu_epoch = max_epochs;  % 记录达到最高准确率时的训练周期
zeta = 8.1;  % 学习率
threshold = 0;  % 停止条件的阈值

% 初始化神经网络的权重和偏置
weight = cell(1, nlayer);
bias = cell(1, nlayer);
nabla_weight = cell(1, nlayer);  % 权重梯度
nabla_bias = cell(1, nlayer);  % 偏置梯度

% 初始化每一层的激活值和预激活值
a = cell(1, nlayer);
z = cell(1, nlayer);
at = cell(1, nlayer);  % 测试集上的激活值
zt = cell(1, nlayer);  % 测试集上的预激活值

% 随机初始化权重和偏置
for in = 2:nlayer
    weight{in} = rand(arch(1, in), arch(1, in - 1)) * 2 - 1;  % 权重随机初始化
    bias{in} = rand(arch(1, in), 1);  % 偏置随机初始化
    nabla_weight{in} = rand(arch(1, in), arch(1, in - 1));  % 初始化权重梯度
    nabla_bias{in} = rand(arch(1, in), 1);  % 初始化偏置梯度
end

% 初始化每一层的激活值和预激活值矩阵
for in = 1:nlayer
    a{in} = zeros(arch(in), mini_batch_size);  % 激活值矩阵
    z{in} = zeros(arch(in), mini_batch_size);  % 预激活值矩阵
    at{in} = zeros(arch(in), ntest);  % 测试集激活值矩阵
    zt{in} = zeros(arch(in), ntest);  % 测试集预激活值矩阵
end

% 进行前向传播，计算测试集上的准确率
at{1} = test_input;  % 将测试集的输入数据赋给激活值
for in = 2:nlayer
    wt = weight{in};
    bt = bias{in};
    ixt = at{in - 1};
    izt = wt * ixt;  % 计算预激活值

    for im = 1:ntest
        izt(:, im) = izt(:, im) + bt;  % 添加偏置
    end

    if in == nlayer  % 输出层使用 Softmax 激活
        zt{in} = izt;
        exat = exp(izt);  % 计算 Softmax 输出
        stant = sum(exat);  % 计算每个样本的总和
        for im = 1:ntest
            at{in}(:, im) = exat(:, im) / stant(1, im);  % 归一化为概率
        end
    else
        zt{in} = izt;
        at{in} = relu(izt);  % 隐藏层使用 ReLU 激活函数
    end
end

% 计算测试集上的准确率
for m = 1:ntest
    [dx, wz] = max(at{nlayer}(:, m));  % 获取预测的标签
    [dx2, wz2] = max(test_target(:, m));  % 获取真实的标签
    if wz == wz2
        s = s + 1;  % 如果预测正确，计数加1
    end
end
accuracy(1, 1) = s / ntest;  % 计算初始准确率

% 开始训练
s = 0;
for ip = 1:max_epochs
    input = train_input;  % 训练集输入
    output = train_target;  % 训练集目标

    a{1} = input;  % 设置输入层的激活值
    if accuracy(ip, 1) ~= 1  % 如果准确率没有达到100%，继续训练
        for in = 2:nlayer
            w = weight{in};
            b = bias{in};
            ix = a{in - 1};
            iz = w * ix;  % 计算预激活值
            for im = 1:mini_batch_size
                iz(:, im) = iz(:, im) + b;  % 添加偏置
            end
            z{in} = iz;
            if in == nlayer  % 输出层使用 Softmax 激活
                exa = exp(iz);
                stan = sum(exa);  % 计算每个样本的总和
                for m = 1:mini_batch_size
                    a{in}(:, m) = exa(:, m) / stan(1, m);  % 归一化为概率
                end
            else
                a{in} = relu(iz);  % 隐藏层使用 ReLU 激活
            end
        end

        % 计算输出层的误差并进行反向传播
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
            weight{in} = weight{in} - zeta * nabla_weight{in};  % 更新权重
            bias{in} = bias{in} - zeta * nabla_bias{in};  % 更新偏置
        end
    end

    % 计算测试集上的准确率
    at{1} = test_input;  % 测试集输入
    for in = 2:nlayer
        wt = weight{in};
        bt = bias{in};
        ixt = at{in - 1};
        izt = wt * ixt;  % 计算预激活值

        for im = 1:ntest
            izt(:, im) = izt(:, im) + bt;  % 添加偏置
        end
        if in == nlayer  % 输出层使用 Softmax 激活
            zt{in} = izt;
            exat = exp(izt);
            stant = sum(exat);
            for im = 1:ntest
                at{in}(:, im) = exat(:, im) / stant(1, im);  % 归一化为概率
            end
        else
            zt{in} = izt;
            at{in} = relu(izt);  % 隐藏层使用 ReLU 激活
        end
    end

    % 计算准确率
    for m = 1:ntest
        [dx, wz] = max(at{nlayer}(:, m));  % 获取预测的标签
        [dx2, wz2] = max(test_target(:, m));  % 获取真实标签
        if wz == wz2
            s = s + 1;  % 如果预测正确，计数加1
        end
    end
    accuracy(ip + 1, 1) = s / ntest;  % 计算准确率
    if (accuracy(ip + 1, 1) == 1) && (ip < max_accu_epoch)
        max_accu_epoch = ip;  % 如果准确率为100%，记录当前的训练周期
    end
    fprintf('%d  %.2f\n', ip, accuracy(ip + 1, 1) * 100);  % 输出当前训练周期的准确率
    s = 0;  % 重置正确分类数量
end

% 输出训练结果
disp(max_accu_epoch);  % 输出达到最高准确率时的训练周期
disp(accuracy(201, 1));  % 输出第 200 轮的准确率

% 绘制训练过程中的准确率曲线
figure, plot(time, accuracy, 'k', 'linewidth', 3);
axis([0 max_epochs, 0, 1]);  % 设置图形的坐标范围
xlabel('epoch');  % x轴标签
ylabel('accuracy');  % y轴标签
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
