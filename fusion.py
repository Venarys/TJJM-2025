import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    # 设置项目目录（根据实际情况修改）
    project_dir = '.'  # 如果当前脚本在项目根目录，否则改为实际路径
    
    # 定义目录结构
    env_data_dir = os.path.join(project_dir, 'env_data')
    # eco_data_dir = os.path.join(project_dir, 'eco_data')
    # svr_models_dir = os.path.join(project_dir, 'svr_models')
    
    # 读取预测文件
    lstm_pred_path = os.path.join(env_data_dir, 'environment_LSTM.xlsx')
    svr_pred_path = os.path.join(project_dir, 'predictions_svr.xlsx')
    
    lstm_df = pd.read_excel(lstm_pred_path)
    svr_df = pd.read_excel(svr_pred_path)
    
    # 读取真实环境数据
    real_env_path = os.path.join(env_data_dir, 'environment.xlsx')
    real_env_df = pd.read_excel(real_env_path)
    
    # 确保数据按年份排序
    lstm_df = lstm_df.sort_values('Year').reset_index(drop=True)
    svr_df = svr_df.sort_values('Year').reset_index(drop=True)
    real_env_df = real_env_df.sort_values('Year').reset_index(drop=True)
    
    # 定义重金属列表（根据实际文件调整）
    # metals = ['AS', 'CD', 'CR', 'CU', 'HG', 'PB', 'ZN']
    metals = ['CU', 'PB', 'ZN', 'CR', 'CD', 'HG', 'AS']
    
    # 初始化结果DataFrame（仅保留未来预测部分）
    future_years_mask = lstm_df['Year'] > 2023
    result_df = lstm_df[future_years_mask][['Year']].copy()
    for metal in metals:
        result_df[metal] = np.nan
    
    # 开始处理每个重金属
    for metal in metals:
        print(f"Processing metal: {metal}")
        
        # 提取训练数据（历史数据）
        train_lstm = lstm_df[lstm_df['Year'] <= 2023][metal].values
        train_svr = svr_df[svr_df['Year'] <= 2023][metal].values
        X_train = np.column_stack((train_lstm, train_svr))
        y_train = real_env_df[real_env_df['Year'] <= 2023][metal].values
        
        # 训练随机森林模型
        rf = RandomForestRegressor(
            n_estimators=100,  # 树的数量
            max_depth=None,    # 树的最大深度
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # 提取未来预测数据
        X_future = np.column_stack((
            lstm_df[future_years_mask][metal].values,
            svr_df[future_years_mask][metal].values
        ))
        
        # 进行融合预测
        fused_predictions = rf.predict(X_future)
        result_df[metal] = fused_predictions
    
    # 保存结果到Excel
    output_path = os.path.join(project_dir, 'predict.xlsx')
    result_df.to_excel(output_path, index=False)
    print(f"Fused predictions saved to {output_path}")

if __name__ == "__main__":
    main()