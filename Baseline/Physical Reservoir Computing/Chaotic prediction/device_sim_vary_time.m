<<<<<<< HEAD
function current_output = device_sim_vary_time(voltage_list, relax, modu)
% device_sim_vary_time函数：根据输入电压列表、电流放松时间和调制时间参数，模拟设备的电流输出。
% 输入：
%   voltage_list：输入电压矩阵，大小为m x n，其中m为样本数，n为每个样本的时间步数。
%   relax：放松时间常数，用于电流放松过程的模拟。
%   modu：调制时间常数，用于模拟调制过程。
% 输出：
%   current_output：设备在输入电压作用下的电流输出，大小为m x n-1。

I0 = 230;  % 初始电流值
[m, n] = size(voltage_list);  % 获取电压矩阵的大小，m为样本数，n为时间步数
I = ones(m, 1) * I0;  % 初始化电流矩阵，每个样本的初始电流为I0

% 对每个样本（行）进行迭代
for i = 1:m
    % 对每个时间步（列）进行迭代
    for j = 1:n
        % 根据电压计算电流变化（applyV函数）
        I(i, j + 1) = I(i, j) + applyV(I(i, j), voltage_list(i, j));
        % 调制过程（modulation函数）
        I(i, j + 1) = modulation(I(i, j + 1), voltage_list(i, j), modu);
        % 运动过程（moveV函数）
        I(i, j + 1) = I(i, j + 1) + moveV(I(i, j + 1), voltage_list(i, j));
        % 放松过程（relaxation函数）
        I(i, j + 1) = relaxation(I(i, j + 1), relax);
    end
end

% 返回去除初始电流值后的电流输出
current_output = I(:, 2:end);
end

% -------------- 以下是计算电流的辅助函数 --------------

% applyV函数：根据输入电压V计算电流变化
function I = applyV(I0, V)
if V < 0
    % 电压小于0时，应用apply_a_1和apply_b_1模型
    I = apply_a_1(V) + apply_b_1(V) * I0;
else
    % 电压大于等于0时，应用apply_a_2和apply_b_2模型
    I = apply_a_2(V) + apply_b_2(V) * I0;
end
end

% apply_a_1函数：电压V小于0时的电流变化模型
function y = apply_a_1(x)
y0 = 3.45112;  % 参数值
A1 = -y0;
t1 = 1.34941;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% apply_b_1函数：电压V小于0时的电流变化模型
function y = apply_b_1(x)
y0 = -0.01151;  % 参数值
A1 = -y0;
t1 = 1.27601;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% apply_a_2函数：电压V大于等于0时的电流变化模型
function y = apply_a_2(x)
y = -4.82412 * x;  % 线性关系
end

% apply_b_2函数：电压V大于等于0时的电流变化模型
function y = apply_b_2(x)
y = 0.01625 * x;  % 线性关系
end

% modulation函数：电流调制过程，根据电压和时间常数调节电流
function I = modulation(I0, V, t)
if V < 0
    bias = 230.4;  % 偏置电流
    % 电压小于0时应用modu_quasi、modu_amplitude和modu_tau模型
    I = modu_quasi(V) + modu_amplitude(V) * exp(-(I0 - bias) / modu_tau(V));
else
    % 电压大于等于0时，电流随时间线性增长
    I = I0 + modu_b(V) * t;
end
end

% modu_quasi函数：电压V小于0时的调制过程（快速变化）
function y = modu_quasi(x)
y0 = 0.10667;
A1 = -y0;
t1 = 0.74558;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% modu_amplitude函数：电压V小于0时的调制幅度
function y = modu_amplitude(x)
y0 = -2.18377;
A1 = -y0;
t1 = 1.38766;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% modu_tau函数：电压V小于0时的调制时间常数
function y = modu_tau(x)
y0 = 10.96015;
A1 = 2.53295e-4;
t1 = 0.3041;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% modu_b函数：电压V大于等于0时的调制系数
function y = modu_b(x)
y0 = 8.57408E-4;
A1 = -8.51028E-4;
t1 = 0.47445;
y = y0 + A1 * exp(x / t1);  % 指数增长模型
end

% moveV函数：电流由于电压引起的运动效应
function I = moveV(I0, V)
if V < 0
    % 电压小于0时，应用move_a_1和move_b_1模型
    I = move_a_1(V) + move_b_1(V) * I0;
else
    % 电压大于等于0时，应用move_a_2和move_b_2模型
    I = move_a_2(V) + move_b_2(V) * I0;
end
end

% move_a_1函数：电压V小于0时的运动过程（快速变化）
function y = move_a_1(x)
y0 = 3.45112;
A1 = -y0;
t1 = 1.34941;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% move_b_1函数：电压V小于0时的运动过程（快速变化）
function y = move_b_1(x)
y0 = -0.01151;
A1 = -y0;
t1 = 1.27601;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% move_a_2函数：电压V大于等于0时的运动过程（线性增长）
function y = move_a_2(x)
y = 4.03075 * x;  % 线性关系
end

% move_b_2函数：电压V大于等于0时的运动过程（线性变化）
function y = move_b_2(x)
y = -0.0139 * x;  % 线性关系
end

% relaxation函数：电流放松过程
function I = relaxation(I1, t)
I = I1 + relax_amplitude(I1) * (1 - exp(-t / relax_tau(I1)));  % 放松模型
end

% relax_amplitude函数：放松过程的幅度，使用Boltzmann拟合
function y = relax_amplitude(x)
A1 = 9.40829;
A2 = 0;
x0 = 205.72315;
dx = 7.73827;
y = A2 + (A1 - A2) / (1 + exp((x - x0) / dx));  % Boltzmann拟合
end

% relax_tau函数：放松时间常数
function y = relax_tau(x)
a = 178.62049;
b = -0.57756;
y = a + b * x;  % 线性模型
end
=======
function current_output = device_sim_vary_time(voltage_list, relax, modu)
% device_sim_vary_time函数：根据输入电压列表、电流放松时间和调制时间参数，模拟设备的电流输出。
% 输入：
%   voltage_list：输入电压矩阵，大小为m x n，其中m为样本数，n为每个样本的时间步数。
%   relax：放松时间常数，用于电流放松过程的模拟。
%   modu：调制时间常数，用于模拟调制过程。
% 输出：
%   current_output：设备在输入电压作用下的电流输出，大小为m x n-1。

I0 = 230;  % 初始电流值
[m, n] = size(voltage_list);  % 获取电压矩阵的大小，m为样本数，n为时间步数
I = ones(m, 1) * I0;  % 初始化电流矩阵，每个样本的初始电流为I0

% 对每个样本（行）进行迭代
for i = 1:m
    % 对每个时间步（列）进行迭代
    for j = 1:n
        % 根据电压计算电流变化（applyV函数）
        I(i, j + 1) = I(i, j) + applyV(I(i, j), voltage_list(i, j));
        % 调制过程（modulation函数）
        I(i, j + 1) = modulation(I(i, j + 1), voltage_list(i, j), modu);
        % 运动过程（moveV函数）
        I(i, j + 1) = I(i, j + 1) + moveV(I(i, j + 1), voltage_list(i, j));
        % 放松过程（relaxation函数）
        I(i, j + 1) = relaxation(I(i, j + 1), relax);
    end
end

% 返回去除初始电流值后的电流输出
current_output = I(:, 2:end);
end

% -------------- 以下是计算电流的辅助函数 --------------

% applyV函数：根据输入电压V计算电流变化
function I = applyV(I0, V)
if V < 0
    % 电压小于0时，应用apply_a_1和apply_b_1模型
    I = apply_a_1(V) + apply_b_1(V) * I0;
else
    % 电压大于等于0时，应用apply_a_2和apply_b_2模型
    I = apply_a_2(V) + apply_b_2(V) * I0;
end
end

% apply_a_1函数：电压V小于0时的电流变化模型
function y = apply_a_1(x)
y0 = 3.45112;  % 参数值
A1 = -y0;
t1 = 1.34941;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% apply_b_1函数：电压V小于0时的电流变化模型
function y = apply_b_1(x)
y0 = -0.01151;  % 参数值
A1 = -y0;
t1 = 1.27601;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% apply_a_2函数：电压V大于等于0时的电流变化模型
function y = apply_a_2(x)
y = -4.82412 * x;  % 线性关系
end

% apply_b_2函数：电压V大于等于0时的电流变化模型
function y = apply_b_2(x)
y = 0.01625 * x;  % 线性关系
end

% modulation函数：电流调制过程，根据电压和时间常数调节电流
function I = modulation(I0, V, t)
if V < 0
    bias = 230.4;  % 偏置电流
    % 电压小于0时应用modu_quasi、modu_amplitude和modu_tau模型
    I = modu_quasi(V) + modu_amplitude(V) * exp(-(I0 - bias) / modu_tau(V));
else
    % 电压大于等于0时，电流随时间线性增长
    I = I0 + modu_b(V) * t;
end
end

% modu_quasi函数：电压V小于0时的调制过程（快速变化）
function y = modu_quasi(x)
y0 = 0.10667;
A1 = -y0;
t1 = 0.74558;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% modu_amplitude函数：电压V小于0时的调制幅度
function y = modu_amplitude(x)
y0 = -2.18377;
A1 = -y0;
t1 = 1.38766;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% modu_tau函数：电压V小于0时的调制时间常数
function y = modu_tau(x)
y0 = 10.96015;
A1 = 2.53295e-4;
t1 = 0.3041;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% modu_b函数：电压V大于等于0时的调制系数
function y = modu_b(x)
y0 = 8.57408E-4;
A1 = -8.51028E-4;
t1 = 0.47445;
y = y0 + A1 * exp(x / t1);  % 指数增长模型
end

% moveV函数：电流由于电压引起的运动效应
function I = moveV(I0, V)
if V < 0
    % 电压小于0时，应用move_a_1和move_b_1模型
    I = move_a_1(V) + move_b_1(V) * I0;
else
    % 电压大于等于0时，应用move_a_2和move_b_2模型
    I = move_a_2(V) + move_b_2(V) * I0;
end
end

% move_a_1函数：电压V小于0时的运动过程（快速变化）
function y = move_a_1(x)
y0 = 3.45112;
A1 = -y0;
t1 = 1.34941;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% move_b_1函数：电压V小于0时的运动过程（快速变化）
function y = move_b_1(x)
y0 = -0.01151;
A1 = -y0;
t1 = 1.27601;
y = y0 + A1 * exp(-x / t1);  % 指数衰减模型
end

% move_a_2函数：电压V大于等于0时的运动过程（线性增长）
function y = move_a_2(x)
y = 4.03075 * x;  % 线性关系
end

% move_b_2函数：电压V大于等于0时的运动过程（线性变化）
function y = move_b_2(x)
y = -0.0139 * x;  % 线性关系
end

% relaxation函数：电流放松过程
function I = relaxation(I1, t)
I = I1 + relax_amplitude(I1) * (1 - exp(-t / relax_tau(I1)));  % 放松模型
end

% relax_amplitude函数：放松过程的幅度，使用Boltzmann拟合
function y = relax_amplitude(x)
A1 = 9.40829;
A2 = 0;
x0 = 205.72315;
dx = 7.73827;
y = A2 + (A1 - A2) / (1 + exp((x - x0) / dx));  % Boltzmann拟合
end

% relax_tau函数：放松时间常数
function y = relax_tau(x)
a = 178.62049;
b = -0.57756;
y = a + b * x;  % 线性模型
end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
