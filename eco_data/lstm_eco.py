import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 参数设置
n_steps_in = 5      # 输入时间步数（使用过去5年的数据）
n_steps_out = 5     # 预测未来5年
epochs = 200        # 训练轮次

def main():
    # 读取数据
    df = pd.read_excel("eco_data/economy.xlsx", index_col="Year")
    original_years = df.index  # 保存原始数据的年份索引

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
    predictions_dict = {}  # 按年份存储预测结果
    
    # 为每个重金属元素训练模型
    for col in df.columns:
        # 获取标准化后的数据
        data = scaled_data[col].flatten()
        
        # 分割训练数据
        X_train, y_train = split_sequences(data, n_steps_in, n_steps_out) 
        if len(X_train) == 0:
            raise ValueError(f"数据不足，无法为{col}创建训练样本")
        
        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(n_steps_in, 1)))
        model.add(LSTM(20, activation='tanh'))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        
        # 训练模型
        model.fit(
            X_train.reshape(X_train.shape[0], n_steps_in, 1),
            y_train,
            epochs=epochs,
            verbose=1
        )
        
        print(f"{col}列训练完成。")
        
        # -----------------------
        # 生成训练数据的预测结果
        # -----------------------
        # 预测所有训练样本的输出
        pred_scaled_train = model.predict(X_train.reshape(X_train.shape[0], n_steps_in, 1))
        pred_original_train = scalers[col].inverse_transform(pred_scaled_train)
        
        # 创建字典存储每个年份的预测值（保留最后一个预测结果）
        col_predictions = {}
        
        for i in range(len(X_train)):
            start_year = original_years[i + n_steps_in]  # 当前预测的起始年份
            predicted_values = pred_original_train[i].flatten().tolist()
            predicted_years = list(range(start_year, start_year + n_steps_out))
            
            # 将预测结果存入字典，覆盖之前的值（保留最后一次预测）
            for yr, val in zip(predicted_years, predicted_values):
                col_predictions[yr] = val
        
        # -----------------------
        # 生成未来数据的预测结果
        # -----------------------
        # 使用最后n_steps_in年的数据作为输入
        future_input = data[-n_steps_in:].reshape(1, n_steps_in, 1)
        pred_scaled_future = model.predict(future_input)
        pred_original_future = scalers[col].inverse_transform(pred_scaled_future).flatten().tolist()
        
        # 添加未来预测的年份
        future_years = list(range(2024, 2024 + n_steps_out))
        for yr, val in zip(future_years, pred_original_future):
            col_predictions[yr] = val
        
        # 将当前列的预测结果合并到总字典中
        predictions_dict[col] = col_predictions
    
    # -----------------------
    # 构建最终预测结果DataFrame
    # -----------------------
    # 确保覆盖所有目标年份（2003-2028）
    all_years = list(range(2003, 2024 + n_steps_out))
    
    # 创建DataFrame
    final_data = {}
    for col in df.columns:
        col_data = []
        for yr in all_years:
            # 如果年份不在预测结果中，填充为NaN（或根据需求处理）
            col_data.append(predictions_dict[col].get(yr, np.nan))
        final_data[col] = col_data
    
    predicted_df = pd.DataFrame(final_data, index=all_years)
    predicted_df.index.name = "Year"
    
    # 保存结果
    predicted_df.to_excel("eco_data/economy_LSTM.xlsx")
    
    print("预测完成！结果已保存到 economy_LSTM.xlsx")

if __name__ == "__main__":
    main()