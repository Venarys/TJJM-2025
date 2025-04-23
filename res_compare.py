import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
methods = ['ARINMA', 'LSTM', 'SVR', '动态融合']
mae_values = [5.3594, 9.4856, 11.6710, 2.9930]
rmse_values = [5.6068, 11.2229, 19.6284, 4.1210]

# 设置组的位置（MAE在0， RMSE在1）
x = np.array([0, 1])

# 参数设置
num_methods = len(methods)
bar_width = 0.15  # 每个条形的宽度
offsets = np.linspace(
    -(num_methods - 1) * bar_width / 2, 
    (num_methods - 1) * bar_width / 2, 
    num_methods
)

# 颜色和标签设置
colors = ['#CCCCCC', '#B2DF8A', '#E5AB02', '#66A61E']  # 绿、红、橙、蓝
labels = ['MAE', 'RMSE']

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个方法的条形
for i in range(num_methods):
    ax.bar(
        x + offsets[i],  # 每个组内的偏移位置
        [mae_values[i], rmse_values[i]],  # 对应的值
        width=bar_width, 
        color=colors[i], 
        label=methods[i], 
        edgecolor='black'
    )

# 设置坐标轴和标签
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('MAE and RMSE', fontsize=12)
ax.set_title('')

# 添加网格和图例
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(
    bbox_to_anchor=(1.05, 1), 
    loc='upper left', 
    title='方法', 
    frameon=False
)

# 自动调整布局
plt.tight_layout()
plt.show()