function pfom_value = pfom(detected_edges, true_edges, tolerance)
    % 获取图像尺寸
    [rows, cols] = size(detected_edges);
    
    % 初始化PFOM值
    pfom_value = 0;
    
    % 初始化变量统计实际边缘点数和检测到的边缘点数
    actual_edge_count = 0;
    detected_edge_count = 0;
    
    % 遍历真实边缘图像中的每个像素
    for y = 1:rows
        for x = 1:cols
            if true_edges(y, x) == 0  % 如果是真实边缘点
                actual_edge_count = actual_edge_count + 1;
                
                % 找到距离此真实边缘点最近的检测到的边缘点
                min_distance = inf;
                for j = max(1, y-tolerance):min(rows, y+tolerance)
                    for i = max(1, x-tolerance):min(cols, x+tolerance)
                        if detected_edges(j, i) == 0
                            distance = sqrt((x - i)^2 + (y - j)^2);
                            if distance < min_distance
                                min_distance = distance;
                            end
                        end
                    end
                end
                
                % 累加到PFOM值
                if min_distance <= tolerance
                    pfom_value = pfom_value + 1 / (1 + min_distance);
                end
            end
        end
    end
    
    % 计算检测到的边缘点数
    detected_edge_count = sum(detected_edges(:) == 0);
    
    % 归一化PFOM值
    pfom_value = pfom_value / max(actual_edge_count, detected_edge_count);
end