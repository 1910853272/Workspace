%% I-V
clc;
clear;

% ----------------------模型参数----------------------
para.r = 0.99;     % 电导更新的因子（记忆衰减因子），用于控制忆阻器的更新速率
para.G0 = 0.5;     % 初始电导值（G0），用于忆阻器的初始化
para.Kp = 9.13;    % 电导正向常数，用于计算电流
para.Kn = 0.32;    % 电导反向常数，用于计算电流
para.alpha = 0.23; % 更新因子alpha，用于调整电导更新过程中的影响

% ----------------------电压序列----------------------
Vmin = -3;         % 最小电压值
Vmax = 3;          % 最大电压值
inv = 0.099;       % 步长（每次增量）
% 生成电压序列V，从Vmin到Vmax，然后从Vmax到Vmin，再从Vmin到0
V = -[0:inv:Vmax-inv, Vmax:-inv:Vmin+inv, Vmin:inv:0];

% ----------------------开始仿真----------------------
step = length(V); % 电压序列的总步数
I = zeros(1, step); % 初始化电流数组，大小与电压序列相同
G = para.G0; % 初始化电导值，使用给定的初始值

% 进行仿真，遍历电压序列计算每个电压值下的电流和电导
for i = 1:step
    % 调用动态忆阻器模型函数，计算每个时刻的电流I(i)和更新后的电导G
    [I(i), G] = DynamicMemristor(V(i), G, para);
end

% ----------------------实验数据----------------------
load('exdata.mat'); % 加载实验数据文件，文件中包含实验电压（Vex）和电流（Iex）

% ----------------------绘图----------------------
figure;
% 使用半对数坐标绘制仿真数据，纵轴是电流的绝对值，加一个很小的常数避免0值
semilogy(V, abs(I)+10^-5, 'b');
hold on; % 保持当前图形，允许叠加下一条曲线
% 绘制实验数据，使用红色（'r'）表示
semilogy(Vex, Iex, 'r');

% 设置图例
str1 = '\color{blue}Simulation'; % 蓝色表示仿真数据
str2 = '\color{red}Experiment'; % 红色表示实验数据
lg = legend(str1, str2); % 添加图例
set(lg, 'box', 'off'); % 去除图例的边框

% 设置坐标轴标签
xlabel('Voltage (V)'); % X轴为电压（V）
ylabel('Current (μA)'); % Y轴为电流（μA）

% 设置坐标轴范围，使用对数坐标系
axis([-3, 3, -inf, inf]); % X轴范围为[-3, 3]，Y轴范围自动适应电流值的对数范围

% 设置坐标轴字体和大小
set(gca, 'FontName', 'Arial', 'FontSize', 20);

% 设置图形的尺寸和位置，类似于MATLAB中的'normalized'位置
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.3, 0.45]); % 设置图形的显示位置和大小

