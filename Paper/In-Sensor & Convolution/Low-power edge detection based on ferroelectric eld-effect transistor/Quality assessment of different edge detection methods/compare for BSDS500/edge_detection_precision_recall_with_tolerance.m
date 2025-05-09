function [precision, recall] = edge_detection_precision_recall_with_tolerance(detected_edges, true_edges, tolerance, PC_detected, PC_true)
    % 找到检测到的边缘点的坐标
    [detected_y, detected_x] = find(detected_edges == 0);
    
    % 找到真实边缘点的坐标
    [true_y, true_x] = find(true_edges == 0);
    

    TP = 0;


    
    % 遍历每个检测到的边缘点，检查它是否在容忍范围内接近任何真实边缘点
    for i = 1:length(detected_x)
        % 计算当前检测边缘点与所有真实边缘点的距离
        distances = sqrt((true_x - detected_x(i)).^2 + (true_y - detected_y(i)).^2);
        
        % 如果存在距离在容忍范围内的真实边缘点，则认为是TP
        if any(distances <= tolerance)
            TP = TP + 1;
        end
    end
    FP = length(detected_x) - TP;
    % 计算False Negative (FN)
    FN = 0;
  
    for j = 1:length(true_x)
        distances = sqrt((detected_x - true_x(j)).^2 + (detected_y - true_y(j)).^2);
        if all(distances > tolerance)
            FN = FN + 1;
            
        end
    end
    
   precision = TP / (TP + FP);

    recall = TP / (TP+ FN);
end