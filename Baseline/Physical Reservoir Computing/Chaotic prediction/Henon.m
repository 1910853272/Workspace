<<<<<<< HEAD
function [x, y] = Henon(datasize)
% Henon函数：生成Henon映射数据序列，通常用于研究混沌系统。
% 输入：
%   datasize：生成的数据序列长度，指定x和y序列的长度。
% 输出：
%   x：Henon映射生成的x序列。
%   y：Henon映射生成的y序列。

% 初始化x和y序列的第一项
x(1, 1) = 0;  % x序列的初始值
y(1, 1) = 0;  % y序列的初始值

% Henon映射的参数
a = 1.4;  % Henon映射中的参数a
b = 0.3;  % Henon映射中的参数b

% 迭代生成Henon映射的x和y序列
for i = 1:datasize+1
    % 计算当前步骤的x值
    x(1, i+1) = 1 + y(1, i) - a * x(1, i)^2;  % Henon映射的x更新公式

    % 计算当前步骤的y值
    y(1, i+1) = b * x(1, i);  % Henon映射的y更新公式
end

% 截取x和y的第3项到最后一项，以去除初始值的影响
x = x(1, 3:end);  % x序列去除前两个值
y = y(1, 3:end);  % y序列去除前两个值

end
=======
function [x, y] = Henon(datasize)
% Henon函数：生成Henon映射数据序列，通常用于研究混沌系统。
% 输入：
%   datasize：生成的数据序列长度，指定x和y序列的长度。
% 输出：
%   x：Henon映射生成的x序列。
%   y：Henon映射生成的y序列。

% 初始化x和y序列的第一项
x(1, 1) = 0;  % x序列的初始值
y(1, 1) = 0;  % y序列的初始值

% Henon映射的参数
a = 1.4;  % Henon映射中的参数a
b = 0.3;  % Henon映射中的参数b

% 迭代生成Henon映射的x和y序列
for i = 1:datasize+1
    % 计算当前步骤的x值
    x(1, i+1) = 1 + y(1, i) - a * x(1, i)^2;  % Henon映射的x更新公式

    % 计算当前步骤的y值
    y(1, i+1) = b * x(1, i);  % Henon映射的y更新公式
end

% 截取x和y的第3项到最后一项，以去除初始值的影响
x = x(1, 3:end);  % x序列去除前两个值
y = y(1, 3:end);  % y序列去除前两个值

end
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
