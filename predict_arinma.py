import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载数据
file_path = './env_data/environment.xlsx'
data = pd.read_excel(file_path)

# 确保数据格式正确
data.set_index('Year', inplace=True)

# 存储结果
predictions = {}
metrics = {}

# 遍历每个重金属列
for column in data.columns:
    # 获取当前列数据
    series = data[column]
    
    # 拆分训练集和测试集
    train = series[:-5]
    test = series[-5:]
    
    # 拟合 ARIMA 模型
    model = ARIMA(train, order=(1, 1, 1))  # 可调整 order 参数
    model_fit = model.fit()
    
    # 预测未来 5 年
    forecast = model_fit.forecast(steps=5)
    predictions[column] = forecast.values
    
    # 计算误差指标
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    r2 = r2_score(test, forecast)
    n = len(test)  # 样本数量
    p = 1  # 自变量数量（ARIMA 模型中假设为 1）
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0  # 避免除以 0
    
    metrics[column] = {
        'RMSE': rmse,
        'MAE': mae,
        'Adjusted R^2': min(adjusted_r2, 1)  # 确保 Adjusted R^2 在 [0, 1] 范围内
    }

# 打印预测结果和误差指标
for column, forecast in predictions.items():
    print(f"Predictions for {column}: {forecast}")
    print(f"Metrics for {column}: {metrics[column]}")

# 保存预测结果到 Excel
output_df = pd.DataFrame(predictions, index=range(data.index[-1] + 1, data.index[-1] + 6))
output_df.to_excel('./env_data/predicted_environment_arinma.xlsx', index_label='Year')