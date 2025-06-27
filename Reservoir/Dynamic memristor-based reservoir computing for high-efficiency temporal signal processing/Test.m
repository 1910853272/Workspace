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
Input = dataset(1, 1:step+1);  % 选择前 step+1 个数据作为输入（取第一行作为输入）

% 生成目标值（实际数据）
Target = Input(2:end);  % 目标为输入数据的第 2 到 end 个元素

% ----------------------可视化 Henon Map 生成的 dataset 和 Input, Target----------------------
% 创建一个图形窗口
figure;

% ----------------------绘制 Henon Map 轨迹----------------------
subplot(1, 2, 1);
plot(dataset(1,:), dataset(2,:), 'k-', 'LineWidth', 1.5);  % 绘制x和y的轨迹
xlabel('x'); 
ylabel('y');
title('Henon Map Trajectory');
grid on;

% ----------------------绘制 Input 和 Target 数据----------------------
subplot(1, 2, 2);
plot(1:step, Input, 'b', 'LineWidth', 1.5);  % 绘制 Input 数据
hold on;
plot(2:step+1, Target, 'r', 'LineWidth', 1.5);  % 绘制 Target 数据
xlabel('Index');
ylabel('Value');
title('Input and Target');
legend({'Input', 'Target'});
grid on;

% 调整布局
sgtitle('Henon Map Visualization');  % 设置整体标题
