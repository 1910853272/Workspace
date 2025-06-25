import pandas as pd
import numpy as np

# 读取 fMRI_features.csv 文件
fmri_data = pd.read_csv('fMRI_features.csv')

# 创建一个空的 DataFrame 用于存储转换后的数据
columns = ['subject_id', 'group', 'modality', 'group_AD', 'group_MCI', 'group_NC',
           'region_id', 'alff_range', 'max_alff', 'mean_alff', 'std_alff', 'min_alff', 'volume']

reshaped_data = pd.DataFrame(columns=columns)

# 定义特征映射关系
feature_map = {
    'alff_range': 'alff_range',
    'max_alff': 'max_alff',
    'mean_alff': 'mean_alff',
    'std_alff': 'std_alff',
    'min_alff': 'min_alff'
}

# 按 subject_id 和 region_id 分组
grouped = fmri_data.groupby(['subject_id', 'region_id'])

# 生成转换后的数据
for (subject_id, region_id), group in grouped:
    group_data = {}
    group_data['subject_id'] = subject_id
    group_data['region_id'] = region_id
    group_data['modality'] = group['modality'].iloc[0]

    # 设置 group
    group_data['group'] = group['group'].iloc[0]  # 获取 group 列的第一个值

    # 生成 group_AD, group_MCI, group_NC
    group_data['group_AD'] = True if group_data['group'] == 'AD' else False
    group_data['group_MCI'] = True if group_data['group'] == 'MCI' else False
    group_data['group_NC'] = True if group_data['group'] == 'NC' else False

    # 填充 region_id 和 feature 对应的值
    region_data = {feature: np.nan for feature in feature_map.values()}  # 初始化 region 对应的特征值为 NaN
    for _, row in group.iterrows():
        feature_name = row['feature']
        feature_value = row['value']

        # 根据 feature_name 映射到目标特征列
        if feature_name in feature_map:
            feature_col = feature_map[feature_name]
            region_data[feature_col] = feature_value

    # 将 region_data 中的值添加到 group_data
    group_data.update(region_data)

    # 如果需要，可以手动或基于数据计算 volume
    group_data['volume'] = np.random.randint(10000, 30000)  # 示例：随机生成 volume，实际可以根据数据填充

    # 将结果添加到 DataFrame 中
    reshaped_data = pd.concat([reshaped_data, pd.DataFrame([group_data])], ignore_index=True, sort=False)

# 删除任何全为 NA 的列
reshaped_data = reshaped_data.dropna(axis=1, how='all')

# 保存结果到文件
reshaped_data.to_csv('reshaped_fMRI_features.csv', index=False)

# 查看转换后的数据
print(reshaped_data.head())
