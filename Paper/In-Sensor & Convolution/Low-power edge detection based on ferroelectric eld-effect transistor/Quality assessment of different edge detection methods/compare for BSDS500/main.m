clc;
clear;

% 指定结果文件夹路径
folderPath = 'data'; % 修改为你的文件夹路径

% 获取文件夹中所有的.mat文件
matFiles = dir(fullfile(folderPath, '*.mat'));

% 初始化一个单元格数组用于存储所有结果
allResults = {};

% 遍历每个.mat文件
for i = 1:length(matFiles)
    % 获取.mat文件的完整路径
    filePath = fullfile(folderPath, matFiles(i).name);
    
    % 加载.mat文件
    data = load(filePath);
    
    % groundTruth 是一个结构体数组
    groundTruth = data.groundTruth;

    % 初始化一个空矩阵用于存储所有groundTruth的合集
    boundaries_sum = false(size(groundTruth{1}.Boundaries));

    % 遍历每个groundTruth结构体，进行合集操作
    for j = 1:length(groundTruth)
        % 读取boundaries信息
        boundaries = groundTruth{j}.Boundaries;

        % 将当前boundaries与合集进行逻辑“或”操作
        boundaries_sum = boundaries_sum | boundaries;
    end

    % 对合集后的边界图像进行反相处理
    inverted_boundaries = ~boundaries_sum;

    % 获取.mat文件名的前缀
    [~, name, ~] = fileparts(matFiles(i).name);

    % 读取对应的CSV文件
    laplacian_of_gaussian = csvread(fullfile(folderPath, [name '_Laplacian_of_Gaussian.csv']));
    musan = csvread(fullfile(folderPath, [name '_Musan.csv']));
    prewitt = csvread(fullfile(folderPath, [name '_Prewitt.csv']));
    roberts = csvread(fullfile(folderPath, [name '_Roberts.csv']));
    sobel = csvread(fullfile(folderPath, [name '_Sobel.csv']));
    susan = csvread(fullfile(folderPath, [name '_Susan.csv']));

    % 将所有的检测结果存入一个结构体数组
    detection_results = struct('name', {}, 'edges', {});

    detection_results(1).name = 'Laplacian of Gaussian';
    detection_results(1).edges = laplacian_of_gaussian;

    detection_results(2).name = 'Prewitt';
    detection_results(2).edges = prewitt;

    detection_results(3).name = 'Roberts';
    detection_results(3).edges = roberts;

    detection_results(4).name = 'Sobel';
    detection_results(4).edges = sobel;

    detection_results(5).name = 'Susan';
    detection_results(5).edges = susan;

    detection_results(6).name = 'Musan';
    detection_results(6).edges = musan;

    % 设置容忍度
    tolerance = 2;

    % 计算每个检测结果的Precision、Recall、FSIM和PFOM
    for k = 1:length(detection_results)
        % 获取当前的检测结果
        detected_edges = detection_results(k).edges;

        % 计算检测结果的相位一致性图
        PC_detected = calculate_phase_congruency(detected_edges);

        % 计算ground truth的相位一致性图
        PC_true = calculate_phase_congruency(inverted_boundaries);

        % 计算带有容忍度的加权Precision和Recall
        [precision_value, recall_value] = edge_detection_precision_recall_with_tolerance(detected_edges, inverted_boundaries, tolerance, PC_detected, PC_true);

        % 计算FSIM
        fsim_value = fsim(detected_edges, inverted_boundaries);

        % 计算PFOM
        pfom_value = pfom(detected_edges, inverted_boundaries, tolerance);

        % 存储结果
        allResults{end+1, 1} = name;
        allResults{end, 2} = detection_results(k).name;
        allResults{end, 3} = precision_value;
        allResults{end, 4} = recall_value;
        allResults{end, 5} = fsim_value;
        allResults{end, 6} = pfom_value;
    end
end

% 将结果写入CSV文件
resultsTable = cell2table(allResults, 'VariableNames', {'File', 'Method', 'Precision', 'Recall', 'FSIM', 'PFOM'});
writetable(resultsTable, 'summary_results.csv');

% 计算每种方法的平均值
uniqueMethods = unique(resultsTable.Method);
methodMeans = cell(length(uniqueMethods), 5);

for m = 1:length(uniqueMethods)
    method = uniqueMethods{m};
    methodData = resultsTable(strcmp(resultsTable.Method, method), :);
    
    meanPrecision = mean(methodData.Precision);
    meanRecall = mean(methodData.Recall);
    meanFSIM = mean(methodData.FSIM);
    meanPFOM = mean(methodData.PFOM);
    
    methodMeans{m, 1} = method;
    methodMeans{m, 2} = meanPrecision;
    methodMeans{m, 3} = meanRecall;
    methodMeans{m, 4} = meanFSIM;
    methodMeans{m, 5} = meanPFOM;
end

% 将平均值结果写入CSV文件
meanResultsTable = cell2table(methodMeans, 'VariableNames', {'Method', 'Mean_Precision', 'Mean_Recall', 'Mean_FSIM', 'Mean_PFOM'});
writetable(meanResultsTable, 'method_mean_results.csv');