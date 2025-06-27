%% Spoken-digit recognition
clc;
clear;
addpath('Auditory Toolbox\');  % 添加音频工具箱路径

% ----------------------Model Parameters----------------------
para.r = 0.99;  % 动态忆阻器的递归系数
para.G0 = 0.5;  % 初始电导值
para.Kp = 9.13; % 电导模型的参数
para.Kn = 0.32; % 电导模型的参数
para.alpha = 0.23;  % 电导更新参数

% ----------------------DM_RC Parameters----------------------
ML = 10;  % 每个数据块的维度
N = 40;   % 掩膜的数量
Vmax = 3; % 输入电压的最大值
Vmin = 0; % 输入电压的最小值
Mask = 2 * randi([0, 1], 64, ML, N) - 1;  % 随机生成掩膜，元素值为-1或+1

% ----------------------DATASET----------------------
% 加载数据集文件路径，生成音频文件的路径列表
for i = 1:5
    for j = 1:10
        for k = 1:10
            filename(k + (i - 1) * 10, j) = {['Voice Data\train\f', num2str(i), '\', '0', num2str(j - 1), 'f', num2str(i), 'set', num2str(k - 1), '.wav']};
        end
    end
end

% ----------------------CROSS-VALIDATION----------------------
WRR = 0;  % 初始化准确率
TF = zeros(10, 10);  % 混淆矩阵初始化

for u = 1:10  % 十次交叉验证
    % 数据集随机打乱
    S = [];
    for i = 1:10
        r = randperm(size(filename, 1));  % 随机排列数据
        res = filename(:, i);
        res = res(r, :);
        S = [S, res];  % 打乱后的数据存储
    end

    % TRAIN
    words = 450;  % 训练词的数量
    VL = zeros(words);  % 存储每个词的长度
    Target = [];  % 存储标签
    X = [];  % 存储训练数据
    q = 0; p = 1;  % 初始化词索引
    for j = 1:words
        q = q + 1;
        if q > 10
            q = 1;
            p = p + 1;
        end

        % 数据预处理
        a = audioread(S{p, q});  % 读取音频文件
        a = resample(a, 8000, 12500);  % 重采样，将音频从12500Hz转换为8000Hz
        f = LyonPassiveEar(a, 8000, 250);  % 使用Lyon耳模型提取特征
        
        % 创建目标标签矩阵
        L = zeros(10, length(f(1, :)));
        L(q, :) = ones(1, length(f(1, :)));  % 设置当前类的标签为1
        VL(j) = length(f(1, :));  % 当前词的长度
        Target(:, sum(VL(1:j)) - VL(j) + 1 : sum(VL(1:j))) = L;  % 更新目标矩阵

        % mask处理
        Input = [];  % 输入矩阵初始化
        for k = 1:N  % 对每个掩膜进行处理
            for i = 1:VL(j)  % 对每个特征进行处理
                Input(k, ML * (i - 1) + 1 : ML * i) = abs(f(:, i))' * Mask(:, :, k);  % 掩膜加权
            end
        end
        UL = max(max(Input));  % 归一化处理
        DL = min(min(Input));
        Input = (Input - DL) / (UL - DL) * (Vmax - Vmin) + Vmin;  % 将输入数据归一化到[Vmin, Vmax]

        % 动态忆阻器输出
        memout = [];
        G = para.G0;  % 初始化电导
        for i = 1:length(Input(1, :))  % 逐步计算输出
            [memout(:, i), G] = DynamicMemristor(Input(:, i), G, para);
        end

        % 状态收集
        for i = 1:VL(j)
            a = memout(:, ML * (i - 1) + 1 : ML * i);  % 将输出展开
            X(:, sum(VL(1:j)) - VL(j) + i) = a(:);  % 将状态存入训练数据矩阵
        end
        
        % 输出训练进度
        sprintf('%s', ['loop:', num2str(u), ', train:', num2str(j), ',', num2str(u - 1), ' acc:', num2str(WRR)]);
    end

    % 线性回归训练
    Wout = Target * X' * pinv(X * X');  % 使用伪逆求解线性回归的输出权重

    % TEST
    clc;
    VL = zeros(words);  % 测试集词数
    Target = []; X = [];  % 初始化
    words = 50; q = 0; p = 46;  % 测试集配置
    for j = 1:words
        q = q + 1;
        if q > 10
            q = 1;
            p = p + 1;
        end

        % 数据预处理
        a = audioread(S{p, q});  % 读取音频文件
        a = resample(a, 8000, 12500);  % 重采样
        f = LyonPassiveEar(a, 8000, 250);  % 提取特征
        
        % 创建目标标签矩阵
        L = zeros(10, length(f(1, :)));
        L(q, :) = ones(1, length(f(1, :)));
        VL(j) = length(f(1, :));
        Target(:, sum(VL(1:j)) - VL(j) + 1 : sum(VL(1:j))) = L;  % 更新目标矩阵

        % mask处理
        Input = [];  % 输入矩阵初始化
        for k = 1:N
            for i = 1:VL(j)
                Input(k, ML * (i - 1) + 1 : ML * i) = abs(f(:, i))' * Mask(:, :, k);
            end
        end
        UL = max(max(Input));  % 归一化处理
        DL = min(min(Input));
        Input = (Input - DL) / (UL - DL) * (Vmax - Vmin) + Vmin;

        % 动态忆阻器输出
        memout = [];
        G = para.G0;
        for i = 1:length(Input(1, :))
            [memout(:, i), G] = DynamicMemristor(Input(:, i), G, para);
        end

        % 状态收集
        for i = 1:VL(j)
            a = memout(:, ML * (i - 1) + 1 : ML * i);
            X(:, sum(VL(1:j)) - VL(j) + i) = a(:);
        end

        % 输出测试进度
        sprintf('%s', ['loop:', num2str(u), ', test:', num2str(j)]);
    end

    % 系统输出
    Y = Wout * X;  % 计算系统的输出

    % 准确率计算
    Mout = [];
    rl = zeros(10, 10);
    real = zeros(10, words);
    for i = 1:words
        Mout(:, i) = mean(Y(:, sum(VL(1:i)) - VL(i) + 1 : sum(VL(1:i))), 2);
        [~, id] = max(Mout(:, i));  % 获取最大输出的索引作为预测类别
        real(id, i) = 1;
        if mod(i, 10) == 0
            rl = rl + real(:, (i / 10 - 1) * 10 + 1 : i);
        end
    end
    WRR = 100 * sum(sum(rl .* eye(10, 10))) / words;  % 计算识别准确率
    TF = TF + rl;  % 累加混淆矩阵
end
WRR = 100 * sum(sum(TF .* eye(10, 10))) / (u * words);  % 计算最终的准确率

% ----------------------PLOT----------------------
% 绘制混淆矩阵
figure(1);
x = [0 9]; y = [0 9];
imagesc(x, y, TF);  % 绘制混淆矩阵
ylabel('Predicted output digit');
xlabel('Correct output digit');
title(['Acc: ', num2str(WRR), '%']);
colorbar;
colormap(flipud(hot)); 
set(gca, 'FontName', 'Arial', 'FontSize', 15);

% 绘制输入信号和忆阻器输出信号
figure(2);
subplot(2, 1, 1);
plot(Input(1, :));  % 绘制输入信号
ylabel('Input (V)');
axis([0, inf, -inf, inf]);
set(gca, 'FontName', 'Arial', 'FontSize', 15);
subplot(2, 1, 2);
plot(memout(1, :), 'r');  % 绘制忆阻器输出信号
xlabel('Time step');
ylabel('Output (μA)');
axis([0, inf, -inf, inf]);
set(gca, 'FontName', 'Arial', 'FontSize', 15);
