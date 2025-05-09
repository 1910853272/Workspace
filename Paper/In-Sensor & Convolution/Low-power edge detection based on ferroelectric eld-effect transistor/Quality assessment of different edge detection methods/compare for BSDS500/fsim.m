function fsim_value = fsim(img1, img2)
    % 将图像转换为灰度图
    if size(img1, 3) == 3
        img1 = rgb2gray(img1);
    end
    if size(img2, 3) == 3
        img2 = rgb2gray(img2);
    end

    % 计算图像梯度幅值和相位一致性
    G1 = calculate_gradient_magnitude(img1);
    G2 = calculate_gradient_magnitude(img2);

    PC1 = calculate_phase_congruency(img1);
    PC2 = calculate_phase_congruency(img2);

    % FSIM核心公式的实现
    T1 = 0.85;
    T2 = 160;

    S_G = (2 * G1 .* G2 + T1) ./ (G1.^2 + G2.^2 + T1);
    S_PC = (2 * PC1 .* PC2 + T2) ./ (PC1.^2 + PC2.^2 + T2);

    % 最终的FSIM计算
    FSIM_map = S_G .* S_PC;
    PC_max = max(PC1, PC2);
    fsim_value = sum(FSIM_map(:) .* PC_max(:)) / sum(PC_max(:));
end