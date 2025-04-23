import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# 文件路径
base_file = "env_data/environment.xlsx"
lstm_file = "env_data/environment_LSTM.xlsx"
svr_file = "predictions_svr.xlsx"
full_file = "predict_full.xlsx"

# 读取数据
base_data = pd.read_excel(base_file)
lstm_data = pd.read_excel(lstm_file)
svr_data = pd.read_excel(svr_file)
full_data = pd.read_excel(full_file)

# 过滤年份范围
start_year, end_year = 2003, 2023
base_data = base_data[(base_data['Year'] >= start_year) & (base_data['Year'] <= end_year)]
lstm_data = lstm_data[(lstm_data['Year'] >= start_year) & (lstm_data['Year'] <= end_year)]
svr_data = svr_data[(svr_data['Year'] >= start_year) & (svr_data['Year'] <= end_year)]
full_data = full_data[(full_data['Year'] >= start_year) & (full_data['Year'] <= end_year)]

# 提取ZN列
base_zn = base_data['ZN'].values
lstm_zn = lstm_data['ZN'].values
svr_zn = svr_data['ZN'].values
full_zn = full_data['ZN'].values

# 定义计算函数
def calculate_metrics(y_true, y_pred, n, p):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return mae, rmse, adjusted_r2

# 计算指标
n = len(base_zn)  # 样本数量
p = 1  # 自变量数量（这里只有ZN一列）

metrics_lstm = calculate_metrics(base_zn, lstm_zn, n, p)
metrics_svr = calculate_metrics(base_zn, svr_zn, n, p)
metrics_full = calculate_metrics(base_zn, full_zn, n, p)

# 打印结果
print("LSTM Metrics (MAE, RMSE, Adjusted R2):", metrics_lstm)
print("SVR Metrics (MAE, RMSE, Adjusted R2):", metrics_svr)
print("Full Metrics (MAE, RMSE, Adjusted R2):", metrics_full)