import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 加载数据
df = pd.read_excel('烟台经济.xlsx', engine='openpyxl')
data = df[['年份', 'GDP']].set_index('年份')

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# 创建时间序列数据集
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 5
X, y = create_dataset(scaled_data, look_back)

# 调整输入形状为 [samples, time_steps, features]
X = X.reshape(X.shape[0], look_back, 1)  # 修正为 (n,5,1)

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))  # 输入形状为 (5,1)
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# 训练模型
model.fit(X, y, epochs=200, batch_size=1, verbose=2)

# 预测未来五年
input_data = scaled_data[-look_back:]  # 形状为 (5,)
predictions_scaled = []
for _ in range(5):
    # 将输入数据 reshape 为 [batch_size, time_steps, features]
    x_pred = input_data.reshape(1, look_back, 1)  # 形状为 (1,5,1)
    pred = model.predict(x_pred)
    predictions_scaled.append(pred[0, 0])
    # 更新输入数据（滚动预测）
    input_data = np.roll(input_data, -1)
    input_data[-1] = pred

# 计算训练集RMSE
train_pred = model.predict(X)
original_data = data.values[look_back+1:].reshape(-1, 1)
pred_scaled_back = scaler.inverse_transform(train_pred)
original_scaled_back = scaler.inverse_transform(original_data)
train_rmse = np.sqrt(np.mean((pred_scaled_back - original_scaled_back)**2))
print(f"训练集RMSE: {train_rmse:.2f}")

# 反向缩放预测结果
predictions = scaler.inverse_transform(
    np.array(predictions_scaled).reshape(-1, 1)
)

# 输出结果
future_years = [2026, 2027, 2028, 2029, 2030]
for year, pred in zip(future_years, predictions):
    print(f"预测{year}年的GDP: {pred[0]:.2f} 亿元")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体
matplotlib.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['GDP'], label='历史GDP')
future_indices = np.arange(2021, 2031)
plt.plot(future_years, predictions, 'r--', label='预测GDP')
plt.legend()
plt.show()