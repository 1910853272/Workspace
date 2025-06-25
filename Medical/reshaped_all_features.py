import pandas as pd
import numpy as np

# 读取 all_features.csv 文件
all_features_data = pd.read_csv('all_features.csv')

# 创建一个空的 DataFrame 用于存储转换后的数据
columns = ['subject_id', 'group', 'modality', 'group_AD', 'group_MCI', 'group_NC',
           'region_id', 'alff_range', 'kurtosis', 'max_alff', 'max_intensity', 'mean_alff',
           'mean_intensity', 'median_intensity', 'min_alff', 'min_intensity', 'percentile_25',
           'percentile_75', 'skewness', 'std_alff', 'std_intensity', 'volume']

reshaped_data = pd.DataFrame(columns=columns)

# 定义特征映射关系
feature_map = {
    'alff_range': 'alff_range',
    'kurtosis': 'kurtosis',
    'max_alff': 'max_alff',
    'max_intensity': 'max_intensity',
    'mean_alff': 'mean_alff',
    'mean_intensity': 'mean_intensity',
    'median_intensity': 'median_intensity',
    'min_alff': 'min_alff',
    'min_intensity': 'min_intensity',
    'percentile_25': 'percentile_25',
    'percentile_75': 'percentile_75',
    'skewness': 'skewness',
    'std_alff': 'std_alff',
    'std_intensity': 'std_intensity'
}

# 按 subject_id 和 region_id 分组
grouped = all_features_data.groupby(['subject_id', 'region_id'])

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

    # 初始化所有特征列的默认值为 NaN
    region_data = {col: np.nan for col in columns[7:]}  # columns[7:] 是所有特征列的名称

    # 填充 region_data 中的特征值
    for _, row in group.iterrows():
        feature_name = row['feature']
        feature_value = row['value']

        # 将特征值映射到相应的列
        if feature_name in region_data:
            region_data[feature_name] = feature_value

    # 将 region_data 中的值添加到 group_data
    group_data.update(region_data)

    # 如果需要，可以手动或基于数据计算 volume
    group_data['volume'] = np.random.randint(10000, 30000)  # 示例：随机生成 volume，实际可以根据数据填充

    # 将结果添加到 DataFrame 中
    reshaped_data = pd.concat([reshaped_data, pd.DataFrame([group_data])], ignore_index=True, sort=False)

# 使用 0 填充所有 NaN 值
reshaped_data = reshaped_data.fillna(0)

# 删除任何全为 NA 的列
reshaped_data = reshaped_data.dropna(axis=1, how='all')

# 保存结果到文件
reshaped_data.to_csv('reshaped_all_features.csv', index=False)

# 查看转换后的数据
print(reshaped_data.head())
