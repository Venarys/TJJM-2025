import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 设置全局参数
LOOK_BACK = 5  # 根据之前分析建议设置为5
FUTURE_YEARS = 5  # 预测未来5年
PRED_FILE = 'prediction.xlsx'

def create_dataset(series, look_back=1):
    """将时间序列转换为监督学习格式"""
    X, y = [], []
    for i in range(len(series) - look_back - 1):
        X.append(series[i:(i+look_back)])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

def train_and_predict(column_data, look_back):
    """训练LSTM并预测未来n年"""
    # 归一化（修正：直接使用 column_data 而不是 column_data.values）
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(column_data.reshape(-1, 1))  # 移除 .values

    # 创建数据集
    X, y = create_dataset(scaled, look_back)
    X = X.reshape(X.shape[0], look_back, 1)

    # 构建并训练模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=200, batch_size=1, verbose=0)

    # 预测未来5年
    input_data = scaled[-look_back:].reshape(-1, 1)
    predictions = []
    for _ in range(FUTURE_YEARS):
        x_pred = input_data[-look_back:].reshape(1, look_back, 1)
        pred = model.predict(x_pred)
        predictions.append(pred[0, 0])
        input_data = np.append(input_data, pred)

    # 反向归一化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# 加载并预处理数据
df = pd.read_excel('烟台经济.xlsx', engine='openpyxl')
original_columns = df.columns.tolist()  # 保存原始列名

# 处理缺失值：使用线性插值填充
df = df.set_index('年份')
df = df.interpolate(method='linear', axis=0).ffill().bfill()  # 线性插值+前后向填充

# 获取所有数值列（排除非数值列）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 预测所有数值列
predictions = {}
for col in numeric_cols:
    print(f"正在预测列：{col}")
    series = df[col]
    # 修正：series.values 已经是 numpy 数组，直接传入
    pred = train_and_predict(series.values, look_back=LOOK_BACK)
    predictions[col] = pred

# 构建输出DataFrame
index = [f"第{i}年后" for i in range(1, FUTURE_YEARS+1)]
result_df = pd.DataFrame(predictions, index=index)

# 保存到Excel
result_df.to_excel(PRED_FILE, index=True, header=True)
print(f"预测结果已保存至：{PRED_FILE}")