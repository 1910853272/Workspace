<<<<<<< HEAD
function current_output = device_sim(voltage_list)
% 模拟设备响应，根据输入的电压列表计算对应的电流输出。
% 输入：voltage_list - 电压列表，每行代表一组电压值。
% 输出：current_output - 对应的电流输出。

I0 = 230;  % 初始电流，假设为230 A
[m, n] = size(voltage_list);  % 获取电压列表的大小，m为电压列表的行数，n为列数
relax = 100.5;  % 放松时间常数
I = ones(m, 1) * I0;  % 初始化电流数组，设定所有电流初始为I0

% 遍历电压列表，计算每个电压点对应的电流值
for i = 1:m
    for j = 1:n
        % 根据电压值更新电流值
        I(i, j + 1) = I(i, j) + applyV(I(i, j), voltage_list(i, j));  % 应用电压，计算电流
        I(i, j + 1) = modulation(I(i, j + 1), voltage_list(i, j), 18.5);  % 调制电流
        I(i, j + 1) = I(i, j + 1) + moveV(I(i, j + 1), voltage_list(i, j));  % 移动电流
        I(i, j + 1) = relaxation(I(i, j + 1), relax);  % 放松电流
    end
end

% 返回计算得到的电流输出（去掉第一个初始电流值）
current_output = I(:, 2:end);
end

% 计算电流的函数，应用电压V对初始电流I0的影响
function I = applyV(I0, V)
if V < 0
    I = apply_a_1(V) + apply_b_1(V) * I0;  % 电压为负时，使用负电压的模型
else
    I = apply_a_2(V) + apply_b_2(V) * I0;  % 电压为正时，使用正电压的模型
end
end

% 负电压下的模型函数
function y = apply_a_1(x)
y0 = 3.45112;  % 常数值
A1 = -y0;
t1 = 1.34941;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 负电压下的模型函数
function y = apply_b_1(x)
y0 = -0.01151;  % 常数值
A1 = -y0;
t1 = 1.27601;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 正电压下的模型函数
function y = apply_a_2(x)
y = -4.82412 * x;  % 电流与电压成线性关系
end

% 正电压下的模型函数
function y = apply_b_2(x)
y = 0.01625 * x;  % 电流与电压成线性关系
end

% 电流调制函数，根据电压V和调制时间t调整电流
function I = modulation(I0, V, t)
if V < 0
    bias = 230.4;  % 偏置电流
    I = modu_quasi(V) + modu_amplitude(V) * exp(-(I0 - bias) / modu_tau(V));  % 负电压下的调制
else
    I = I0 + modu_b(V) * t;  % 正电压下的调制
end
end

% 负电压下的准静态调制函数
function y = modu_quasi(x)
y0 = 0.10667;  % 常数值
A1 = -y0;
t1 = 0.74558;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 负电压下的幅度调制函数
function y = modu_amplitude(x)
y0 = -2.18377;  % 常数值
A1 = -y0;
t1 = 1.38766;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 电流调制的时间常数函数
function y = modu_tau(x)
y0 = 10.96015;  % 常数值
A1 = 2.53295e-4;  % 常数值
t1 = 0.3041;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 电流调制中的b函数
function y = modu_b(x)
y0 = 8.57408E-4;  % 常数值
A1 = -8.51028E-4;  % 常数值
t1 = 0.47445;  % 时间常数
y = y0 + A1 * exp(x / t1);  % 计算结果
end

% 电流移动函数，根据电压V对电流I进行移动
function I = moveV(I0, V)
if V < 0
    I = move_a_1(V) + move_b_1(V) * I0;  % 负电压时的移动
else
    I = move_a_2(V) + move_b_2(V) * I0;  % 正电压时的移动
end
end

% 负电压下的移动函数
function y = move_a_1(x)
y0 = 3.45112;  % 常数值
A1 = -y0;
t1 = 1.34941;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 负电压下的移动函数
function y = move_b_1(x)
y0 = -0.01151;  % 常数值
A1 = -y0;
t1 = 1.27601;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 正电压下的移动函数
function y = move_a_2(x)
y = 4.03075 * x;  % 电流与电压线性关系
end

% 正电压下的移动函数
function y = move_b_2(x)
y = -0.0139 * x;  % 电流与电压线性关系
end

% 放松函数，根据当前电流和放松时间计算新的电流值
function I = relaxation(I1, t)
I = I1 + relax_amplitude(I1) * (1 - exp(-t / relax_tau(I1)));  % 放松电流的计算
end

% 放松振幅函数，基于Boltzmann拟合
function y = relax_amplitude(x)
A1 = 9.40829;  % 常数值
A2 = 0;  % 常数值
x0 = 205.72315;  % 中心点
dx = 7.73827;  % 宽度
y = A2 + (A1 - A2) / (1 + exp((x - x0) / dx));  % 计算放松振幅
end

% 放松时间常数函数，基于线性拟合
function y = relax_tau(x)
a = 178.62049;  % 常数值
b = -0.57756;  % 常数值
y = a + b * x;  % 计算放松时间常数
end
=======
function current_output = device_sim(voltage_list)
% 模拟设备响应，根据输入的电压列表计算对应的电流输出。
% 输入：voltage_list - 电压列表，每行代表一组电压值。
% 输出：current_output - 对应的电流输出。

I0 = 230;  % 初始电流，假设为230 A
[m, n] = size(voltage_list);  % 获取电压列表的大小，m为电压列表的行数，n为列数
relax = 100.5;  % 放松时间常数
I = ones(m, 1) * I0;  % 初始化电流数组，设定所有电流初始为I0

% 遍历电压列表，计算每个电压点对应的电流值
for i = 1:m
    for j = 1:n
        % 根据电压值更新电流值
        I(i, j + 1) = I(i, j) + applyV(I(i, j), voltage_list(i, j));  % 应用电压，计算电流
        I(i, j + 1) = modulation(I(i, j + 1), voltage_list(i, j), 18.5);  % 调制电流
        I(i, j + 1) = I(i, j + 1) + moveV(I(i, j + 1), voltage_list(i, j));  % 移动电流
        I(i, j + 1) = relaxation(I(i, j + 1), relax);  % 放松电流
    end
end

% 返回计算得到的电流输出（去掉第一个初始电流值）
current_output = I(:, 2:end);
end

% 计算电流的函数，应用电压V对初始电流I0的影响
function I = applyV(I0, V)
if V < 0
    I = apply_a_1(V) + apply_b_1(V) * I0;  % 电压为负时，使用负电压的模型
else
    I = apply_a_2(V) + apply_b_2(V) * I0;  % 电压为正时，使用正电压的模型
end
end

% 负电压下的模型函数
function y = apply_a_1(x)
y0 = 3.45112;  % 常数值
A1 = -y0;
t1 = 1.34941;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 负电压下的模型函数
function y = apply_b_1(x)
y0 = -0.01151;  % 常数值
A1 = -y0;
t1 = 1.27601;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 正电压下的模型函数
function y = apply_a_2(x)
y = -4.82412 * x;  % 电流与电压成线性关系
end

% 正电压下的模型函数
function y = apply_b_2(x)
y = 0.01625 * x;  % 电流与电压成线性关系
end

% 电流调制函数，根据电压V和调制时间t调整电流
function I = modulation(I0, V, t)
if V < 0
    bias = 230.4;  % 偏置电流
    I = modu_quasi(V) + modu_amplitude(V) * exp(-(I0 - bias) / modu_tau(V));  % 负电压下的调制
else
    I = I0 + modu_b(V) * t;  % 正电压下的调制
end
end

% 负电压下的准静态调制函数
function y = modu_quasi(x)
y0 = 0.10667;  % 常数值
A1 = -y0;
t1 = 0.74558;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 负电压下的幅度调制函数
function y = modu_amplitude(x)
y0 = -2.18377;  % 常数值
A1 = -y0;
t1 = 1.38766;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 电流调制的时间常数函数
function y = modu_tau(x)
y0 = 10.96015;  % 常数值
A1 = 2.53295e-4;  % 常数值
t1 = 0.3041;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 电流调制中的b函数
function y = modu_b(x)
y0 = 8.57408E-4;  % 常数值
A1 = -8.51028E-4;  % 常数值
t1 = 0.47445;  % 时间常数
y = y0 + A1 * exp(x / t1);  % 计算结果
end

% 电流移动函数，根据电压V对电流I进行移动
function I = moveV(I0, V)
if V < 0
    I = move_a_1(V) + move_b_1(V) * I0;  % 负电压时的移动
else
    I = move_a_2(V) + move_b_2(V) * I0;  % 正电压时的移动
end
end

% 负电压下的移动函数
function y = move_a_1(x)
y0 = 3.45112;  % 常数值
A1 = -y0;
t1 = 1.34941;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 负电压下的移动函数
function y = move_b_1(x)
y0 = -0.01151;  % 常数值
A1 = -y0;
t1 = 1.27601;  % 时间常数
y = y0 + A1 * exp(-x / t1);  % 计算结果
end

% 正电压下的移动函数
function y = move_a_2(x)
y = 4.03075 * x;  % 电流与电压线性关系
end

% 正电压下的移动函数
function y = move_b_2(x)
y = -0.0139 * x;  % 电流与电压线性关系
end

% 放松函数，根据当前电流和放松时间计算新的电流值
function I = relaxation(I1, t)
I = I1 + relax_amplitude(I1) * (1 - exp(-t / relax_tau(I1)));  % 放松电流的计算
end

% 放松振幅函数，基于Boltzmann拟合
function y = relax_amplitude(x)
A1 = 9.40829;  % 常数值
A2 = 0;  % 常数值
x0 = 205.72315;  % 中心点
dx = 7.73827;  % 宽度
y = A2 + (A1 - A2) / (1 + exp((x - x0) / dx));  % 计算放松振幅
end

% 放松时间常数函数，基于线性拟合
function y = relax_tau(x)
a = 178.62049;  % 常数值
b = -0.57756;  % 常数值
y = a + b * x;  % 计算放松时间常数
end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
