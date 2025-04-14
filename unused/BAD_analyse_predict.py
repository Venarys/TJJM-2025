import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------
# 1. 定义常量和路径
# ---------------------------

# 使用线性回归，可是量纲相差过大，产生了权重消失的现象。
ECONOMIC_HISTORY_FILE = 'economy.xlsx'  # 经济历史数据
HEAVY_METAL_HISTORY_FILE = 'environment.xlsx'  # 重金属历史数据
ECONOMIC_FUTURE_FILE = 'economy_future.xlsx'      # 经济未来数据
WEIGHTS_FILE = 'weights.xlsx'                      # 权重输出文件
PREDICTIONS_FILE = 'predictions.xlsx'              # 预测结果输出文件

# 重金属元素列表（根据您的需求修改）
METALS = ['CU', 'PB', 'ZN', 'CR', 'CD', 'HG', 'AS']

# 经济指标列（需与历史经济数据的列名一致）
ECONOMIC_COLUMNS = [
    'GDP',
    '第一产业（亿）',
    '第二产业（亿）',	
    '第三产业（亿）',	
    '工业（亿）',	
    '建筑业（亿）',	
    '环境保护支出',	
    '废水排放量',	
    '废水排放量（工业）',	
    '二氧化硫排放量',	
    '粉尘排放量',	
    '工业固废产生量',
    '农业产值',
    '林业产值',
    '牧业产值',
    '渔业产值',
]  # 根据实际列名修改

# ---------------------------
# 2. 读取和合并历史数据
# ---------------------------
def load_and_merge_data():
    # 读取经济历史数据和重金属历史数据
    economic = pd.read_excel(ECONOMIC_HISTORY_FILE).set_index('Year')
    heavy_metal = pd.read_excel(HEAVY_METAL_HISTORY_FILE).set_index('Year')
    
    # 合并数据（按年份索引合并）
    merged_data = pd.merge(economic, heavy_metal, left_index=True, right_index=True, how='inner')
    
    return merged_data

# ---------------------------
# 3. 训练模型并提取权重
# ---------------------------
def train_models_and_save_weights(merged_data):
    # 初始化权重DataFrame（行：重金属，列：经济指标）
    weights = pd.DataFrame(index=METALS, columns=ECONOMIC_COLUMNS)
    
    # 遍历每个重金属元素
    for metal in METALS:
        # 提取自变量（经济指标）和因变量（重金属含量）
        X = merged_data[ECONOMIC_COLUMNS]
        y = merged_data[metal]
        
        # 训练线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 提取权重系数（排除截距项）
        coefficients = model.coef_
        
        # 将权重存入DataFrame
        weights.loc[metal] = coefficients
    
    # 保存权重文件
    weights.to_excel(WEIGHTS_FILE)
    print(f"权重文件已保存至 {WEIGHTS_FILE}")

# ---------------------------
# 4. 预测未来重金属含量
# ---------------------------
def predict_future_metals():
    # 读取未来经济数据
    future_economic = pd.read_excel(ECONOMIC_FUTURE_FILE).set_index('Year')
    
    # 读取权重文件
    weights = pd.read_excel(WEIGHTS_FILE, index_col=0)
    
    # 初始化预测结果DataFrame（行：未来年份，列：重金属元素）
    predictions = pd.DataFrame(index=future_economic.index, columns=METALS)
    
    # 遍历每个未来年份和每个重金属元素
    for year in future_economic.index:
        for metal in METALS:
            # 提取该年的经济指标数据
            s = future_economic.loc[year, ECONOMIC_COLUMNS]
            
            # 提取对应的权重系数
            w = weights.loc[metal]
            
            # 计算预测值（公式：A = w1*s1 + w2*s2 + wn*sn）
            predicted_value = np.dot(w, s)  # 向量点积
            
            # 保存结果
            predictions.loc[year, metal] = predicted_value
    
    # 保存预测结果
    predictions.to_excel(PREDICTIONS_FILE)
    print(f"预测结果已保存至 {PREDICTIONS_FILE}")

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    # 加载和合并数据
    merged_data = load_and_merge_data()
    
    # 训练模型并保存权重
    train_models_and_save_weights(merged_data)
    
    # 预测未来数据
    predict_future_metals()

if __name__ == "__main__":
    main()