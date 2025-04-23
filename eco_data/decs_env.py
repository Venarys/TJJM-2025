import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取 Excel 文件
file_path = 'env_data/environment.xlsx'  # 请根据实际路径调整
data = pd.read_excel(file_path)

# 确保文件有数据
if data.empty:
    print("文件为空或路径错误，请检查文件内容和路径。")
else:
    # 设置年份为 x 轴
    years = data.iloc[:, 0]  # 第一列为年份
    data = data.iloc[:, 1:]  # 除去第一列的其他列数据

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(years, data[column], label=column)

    # 图表美化
    plt.title('重金属含量变化趋势')
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()