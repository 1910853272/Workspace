function output = device_linear(input)
% device_linear函数：用于模拟一个线性设备的响应
% 输入：
%   input：一个矩阵，表示输入信号，大小为m x n，其中m是样本数，n是每个样本的特征数。
% 输出：
%   output：一个列向量，表示设备的输出。

% 获取输入矩阵的行数m和列数n
[m, n] = size(input);

% 初始化一个全零的列向量out，用于存储每个样本的累积和（线性响应）
out = zeros(m, 1);

% 遍历所有样本（行）
for i = 1:m
    % 遍历当前样本的每个特征（列）
    for j = 1:n
        % 累加当前特征的值到out(i)中
        out(i, j + 1) = out(i, j) + input(i, j);
    end
end

% 输出结果：去除out的第一个列（因为out的第一列是全零的）
output = out(:, 2:end);
end
