import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 参数设置
n_steps_in = 5      # 输入时间步数（使用过去5年的数据）
n_steps_out = 5     # 预测未来5年
epochs = 200        # 训练轮次

def main():
    # 读取数据
    df = pd.read_excel("environment.xlsx", index_col="Year")
    
    # 标准化每个重金属列
    scalers = {}
    scaled_data = {}
    for col in df.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df[[col]])
        scaled_data[col] = scaled
        scalers[col] = scaler
    
    # 数据分割函数，滑动窗口
    def split_sequences(sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequences):
                break
            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    # 创建预测结果字典
    predictions = {}
    
    # 为每个重金属元素训练模型
    for col in df.columns:
        # 获取标准化后的数据
        data = scaled_data[col].flatten()
        
        # 分割训练数据
        X, y = split_sequences(data, n_steps_in, n_steps_out)
        
        # 检查数据是否足够
        if len(X) == 0:
            raise ValueError(f"数据不足，无法为{col}创建训练样本")
        
        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, 1)))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        
        # 训练数据
        model.fit(
            X.reshape(X.shape[0], n_steps_in, 1),
            y,
            epochs=epochs,
            verbose=1  # 显示每轮训练进度
        )
        
        # 获取最后n_steps_in年的数据作为输入
        input_sequence = data[-n_steps_in:].reshape(1, n_steps_in, 1)
        
        # 进行预测
        pred_scaled = model.predict(input_sequence).flatten()
        
        # 反标准化处理
        pred_original = scalers[col].inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()
        
        # 存储预测结果
        predictions[col] = pred_original
    
    # 创建预测结果DataFrame
    predicted_df = pd.DataFrame(
        predictions,
        index=[f"第{i+1}年后" for i in range(n_steps_out)]
    )
    
    # 保存结果
    predicted_df.to_excel("predicted_heavy_metal_LSTM.xlsx")
    
    print("预测完成！结果已保存到 predicted_heavy_metal_LSTM.xlsx")

if __name__ == "__main__":
    main()