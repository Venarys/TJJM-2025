import pandas as pd

import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = './predict_full.xlsx'  # 确保文件路径正确
data = pd.read_excel(file_path)

# 检查数据格式
if data.empty or data.shape[1] < 2:
    raise ValueError("文件内容不符合要求，第一列应为年份，后续列应为数据")

# 提取年份和重金属数据
years = data.iloc[:, 0]  # 第一列为年份
metal_data = data.iloc[:, 1:]  # 后续列为重金属数据

# 绘制折线图
plt.figure(figsize=(10, 6))
for column in metal_data.columns:
    plt.plot(years, metal_data[column], label=column)

# 添加图例、标题和坐标轴标签
plt.title('Heavy Metal Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()