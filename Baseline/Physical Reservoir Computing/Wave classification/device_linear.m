function output = device_linear(input)
% 该函数模拟一个线性设备响应，将输入的矩阵按元素累加计算输出。
% 输入：input - 输入矩阵，大小为m x n，其中m为行数，n为列数
% 输出：output - 处理后的输出矩阵，大小为m x (n-1)

[m, n] = size(input);  % 获取输入矩阵的行列数，m为行数，n为列数
out = zeros(m, 1);  % 初始化一个m行1列的零矩阵out，用于存储累计结果

% 遍历输入矩阵的每一行
for i = 1:m
    for j = 1:n
        % 累加输入矩阵的每个元素，并将结果存入out的相应位置
        out(i, j + 1) = out(i, j) + input(i, j);  % 累加输入矩阵元素到out中
    end
end

% 输出矩阵是累加过程的结果，从第二列开始（即去掉第一列的初始值）
output = out(:, 2:end);  % 返回out矩阵的第2列到第n列的部分
end
