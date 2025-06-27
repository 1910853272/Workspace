%% Henon Map prediction
clc;  % 清除命令窗口内容
clear;  % 清除工作区所有变量

% ----------------------Model Parameters----------------------
para.r = 0.99;  % 动态忆阻器的递归系数
para.G0 = 0.5;  % 初始电导值
para.Kp = 9.13;  % 电导的正向增益
para.Kn = 0.32;  % 电导的反向增益
para.alpha = 0.23;  % 更新函数的控制参数

% ----------------------DM_RC Parameters----------------------
ML = 4;  % 输入数据的特征维度
N = 25;  % 掩膜的数量
Vmax = 2.5;  % 输入信号的最大电压值
Vmin = 0;  % 输入信号的最小电压值

% ----------------------DATASET----------------------
step = 1000;  % 数据序列的步数
dataset = HenonMap(2*step+1);  % 生成Henon映射的数据集

% ----------------------TRAIN----------------------
% 初始化输入流
Input = dataset(1:step+1);  % 选择前 step+1 个数据作为输入

% 生成目标值（实际数据）
Target = Input(2:end);  % 目标为输入数据的第 2 到 end 个元素

% ----------------------MASK PROCESS----------------------
% 创建掩膜矩阵，用于模拟物理水库计算
Mask = 2*unidrnd(2, N, ML)-3;  % 生成值为 -1 或 1 的掩膜矩阵
Input_ex = [];  % 初始化扩展后的输入矩阵

% 对每个输入数据应用掩膜
for j = 1:N
    for i = 1:step
        % 使用掩膜生成扩展后的输入矩阵
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i)*Mask(j, :);  % 扩展输入矩阵
    end
end

% 归一化处理
UL = max(max(Input_ex));  % 获取输入数据的最大值
DL = min(min(Input_ex));  % 获取输入数据的最小值
Input_ex = (Input_ex - DL) / (UL - DL) * (Vmax - Vmin) + Vmin;  % 归一化至 [Vmin, Vmax]

% ----------------------MEMRISTOR OUTPUT----------------------
memout = [];  % 初始化忆阻器输出
states = [];  % 初始化状态矩阵
G = para.G0;  % 初始电导

% 模拟动态忆阻器的输出
for i = 1:length(Input_ex(1, :))  % 对每个输入信号进行模拟
    [memout(:,i), G] = DynamicMemristor(Input_ex(:,i), G, para);  % 动态忆阻器输出
    sprintf('%s', ['train:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])
end

% ----------------------LINEAR REGRESSION----------------------
% 使用线性回归计算输出权重
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);  % 按照特征维度切片
    states(:, i) = a(:);  % 展开每个切片，得到一维数组
end

X = [ones(1, step); states];  % 构建输入矩阵，第一行是全 1，用于偏置项
Wout = Target * pinv(X);  % 计算线性回归权重（伪逆法）

% ----------------------TEST----------------------
% 初始化测试集输入流
Input = dataset(step+1:2*step+1);  % 从第 step+1 到 2*step+1 个数据作为输入

% 生成测试集目标值
Target = Input(2:end);  % 测试集的目标为输入数据的第 2 到 end 个元素

% ----------------------MASK PROCESS----------------------
% 重新应用掩膜处理
Input_ex = [];
for j = 1:N
    for i = 1:step
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i)*Mask(j, :);  % 扩展输入矩阵
    end
end
UL = max(max(Input_ex));
DL = min(min(Input_ex));
Input_ex = (Input_ex - DL) / (UL - DL) * (Vmax - Vmin) + Vmin;  % 归一化处理

% ----------------------MEMRISTOR OUTPUT----------------------
% 对测试数据进行忆阻器模拟
memout = [];
states = [];
G = para.G0;
for i = 1:length(Input_ex(1, :))  % 对每个输入信号进行模拟
    [memout(:, i), G] = DynamicMemristor(Input_ex(:,i), G, para);  % 计算忆阻器输出
    sprintf('%s', ['test:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])
end

% ----------------------SYSTEM OUTPUT----------------------
% 计算系统的输出
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);  % 切片提取特征
    states(:,i) = a(:);  % 展开每个特征
end

X = [ones(1, step); states];  % 构建输入矩阵，第一行是全 1，用于偏置项
Out = Wout * X;  % 使用线性回归模型计算输出

% ----------------------NRMSE CALCULATION----------------------
% 计算归一化均方根误差（NRMSE）
NRMSE = sqrt(mean((Out(10:end)-Target(10:end)).^2)./var(Target(10:end)));  % 计算从第 10 到最后的数据的 NRMSE
sprintf('%s',['NRMSE:', num2str(NRMSE)])

% ----------------------PLOT----------------------
% 绘制时间序列图
figure(1);
plot(Target(1:200), 'k', 'linewidth', 2);  % 绘制目标值
hold on;
plot(Out(1:200), 'r', 'linewidth',1);  % 绘制输出值
axis([0, 200, -2, 2])  % 设置坐标轴范围
str1 = '\color{black}Target';
str2 = '\color{red}Output';
lg = legend(str1, str2);  % 设置图例
set(lg, 'Orientation', 'horizon', 'box', 'off');  % 设置图例的样式
ylabel('Prediction')  % 设置 y 轴标签
xlabel('Time (\tau)')  % 设置 x 轴标签
set(gca,'FontName', 'Arial', 'FontSize', 20);  % 设置坐标轴字体
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);  % 设置图形窗口位置和大小

% 绘制 2D 映射图
figure(2);
plot(Target(2:end), 0.3*Target(1:end-1), '.k', 'markersize', 12);  % 绘制目标值的 2D 映射
hold on;
plot(Out(2:end), 0.3*Out(1:end-1), '.r', 'markersize', 12);  % 绘制输出值的 2D 映射
str1 = '\color{black}Target';
str2 = '\color{red}Output';
lg = legend(str1,str2);  % 设置图例
set(lg, 'box', 'off');  % 去掉图例框
ylabel('{\ity} (n)');  % 设置 y 轴标签
xlabel('{\itx} (n)');  % 设置 x 轴标签
axis([-2, 2, -0.4, 0.4]);  % 设置坐标轴范围
set(gca, 'FontName', 'Arial', 'FontSize', 20);  % 设置坐标轴字体
set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.3,0.45]);  % 设置图形窗口位置和大小
