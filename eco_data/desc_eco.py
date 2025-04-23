import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 文件路径
file_path = os.path.join(os.path.dirname(__file__), 'economy_raw.xlsx')

# 读取 Excel 文件
data = pd.read_excel(file_path)

# 确保第一列是年份
data = data.rename(columns={data.columns[0]: 'Year'})

# 筛选 B-G 列和 N-R 列
columns_b_to_g = data.columns[1:7]  # 假设 B-G 对应第 2 列到第 7 列
columns_n_to_r = data.columns[13:18]  # 假设 N-R 对应第 14 列到第 18 列

# 筛选 2010 年和 2020 年的数据
filtered_data = data[data['Year'].isin([2010, 2020])]

# 创建一个包含两个子图的画布，左右排列
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 行 2 列的子图布局

# 绘制 B-G 列的条形图（左侧）
filtered_data_b_to_g = filtered_data[columns_b_to_g]
filtered_data_b_to_g.T.plot(kind='bar', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('列名')
axes[0].set_ylabel('值')
axes[0].tick_params(axis='x', rotation=15)  # 调整字体旋转角度
axes[0].legend(filtered_data['Year'].astype(str), title='年份')
axes[0].grid(axis='y')

# 绘制 N-R 列的条形图（右侧）
filtered_data_n_to_r = filtered_data[columns_n_to_r]
filtered_data_n_to_r.T.plot(kind='bar', ax=axes[1])
axes[1].set_title('')
axes[1].set_xlabel('列名')
axes[1].set_ylabel('值')
axes[1].tick_params(axis='x', rotation=15)  # 调整字体旋转角度
axes[1].legend(filtered_data['Year'].astype(str), title='年份')
axes[1].grid(axis='y')

# 自动调整布局以避免重叠
plt.tight_layout()

# 显示图形
plt.show()