import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# 读取数据
economic_history = pd.read_excel("eco_data/economy.xlsx", index_col="Year")
heavy_metal_history = pd.read_excel("env_data/environment.xlsx", index_col="Year")
economic_future = pd.read_excel("eco_data/economy_LSTM.xlsx", index_col="Year")

# 合并历史数据（按年份对齐）
merged_data = pd.merge(
    economic_history,
    heavy_metal_history,
    left_index=True,
    right_index=True,
    how="inner",
)

# 提取特征列和目标列
economic_features = economic_history.columns.tolist()
target_columns = heavy_metal_history.columns.tolist()

# 标准化特征数据（SVM 对特征缩放敏感）
scaler = StandardScaler()
X = merged_data[economic_features]
X_scaled = scaler.fit_transform(X)
y = merged_data[target_columns]

# 存储所有重金属元素的模型
models = {}

# 训练每个重金属元素的 SVR 模型
for element in target_columns:
    # 划分训练集和测试集（这里使用随机分割，实际可考虑时间序列分割）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y[element], test_size=0.2, random_state=42
    )
    model = SVR(
        kernel="rbf",
        C=1.0,        # 初始惩罚参数（可调整）
        gamma="scale"
    )  # 使用 RBF 核函数
    model.fit(X_train, y_train)
    models[element] = model

# 准备预测数据
# 在 2023 年及之前，使用 economy.xlsx 的数据作为输入
# 在 2024 年及以后，使用 economy_LSTM.xlsx 的数据作为输入
future_years = list(range(2003, 2029))  # 2003-2028 年
economic_combined = pd.concat([
    economic_history.loc[2003:2023],  # 2003-2023 年的数据
    economic_future.loc[2024:]       # 2024-2028 年的数据
])

# 标准化预测数据
future_X = economic_combined[economic_features]
future_X_scaled = scaler.transform(future_X)  # 使用相同的标准化参数

# 创建预测结果字典
predictions = {}
for i, year in enumerate(future_years):
    # 预测该行数据
    prediction_row = {}
    for element in target_columns:
        model = models[element]
        pred = model.predict([future_X_scaled[i]])
        prediction_row[element] = pred[0]
    predictions[year] = prediction_row

# 将预测结果转为 DataFrame
pred_df = pd.DataFrame(predictions).T

# 设置索引为 2003-2028 年
pred_df.index.name = "Year"

# 保存结果到 Excel
output_file = "predictions_svr.xlsx"
pred_df.to_excel(output_file)

print(f"预测完成！结果已保存到 {output_file}")