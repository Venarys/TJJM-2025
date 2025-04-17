import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    project_dir = '.'  # 根据实际情况修改路径
    env_data_dir = os.path.join(project_dir, 'env_data')
    eco_data_dir = os.path.join(project_dir, 'eco_data')
    
    # 读取预测文件
    lstm_df = pd.read_excel(os.path.join(env_data_dir, 'environment_LSTM.xlsx'))
    svr_df = pd.read_excel(os.path.join(project_dir, 'predictions_svr.xlsx'))
    
    # 读取真实环境数据（用于训练模型）
    real_env_df = pd.read_excel(os.path.join(env_data_dir, 'environment.xlsx'))
    
    # 确保 Year 列是整数类型
    for df in [lstm_df, svr_df, real_env_df]:
        df['Year'] = df['Year'].astype(int)
    
    metals = ['CU', 'PB', 'ZN', 'CR', 'CD', 'HG', 'AS']
    
    # 初始化结果DataFrame（包含2003-2028年所有年份）
    result_df = lstm_df[(lstm_df['Year'] >= 2003) & (lstm_df['Year'] <= 2028)][['Year']].copy()
    for metal in metals:
        result_df[metal] = np.nan
    
    # 定义训练时间范围（2003-2023）
    train_start = 2003
    train_end = 2023
    
    for metal in metals:
        print(f"Processing metal: {metal}")
        
        # 提取训练数据（2003-2023）
        train_lstm = lstm_df[
            (lstm_df['Year'] >= train_start) & 
            (lstm_df['Year'] <= train_end)
        ][metal].values
        
        train_svr = svr_df[
            (svr_df['Year'] >= train_start) & 
            (svr_df['Year'] <= train_end)
        ][metal].values
        
        y_train = real_env_df[
            (real_env_df['Year'] >= train_start) & 
            (real_env_df['Year'] <= train_end)
        ][metal].values
        
        # 检查训练数据长度一致性
        if len(train_lstm) != len(train_svr) or len(train_svr) != len(y_train):
            print(f"⚠️ {metal} 训练数据长度不一致！")
            continue
        
        X_train = np.column_stack((train_lstm, train_svr))
        
        # 训练随机森林模型
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # 提取所有年份（2003-2028）的预测数据
        X_all = np.column_stack((
            lstm_df[
                (lstm_df['Year'] >= 2003) & 
                (lstm_df['Year'] <= 2028)
            ][metal].values,
            
            svr_df[
                (svr_df['Year'] >= 2003) & 
                (svr_df['Year'] <= 2028)
            ][metal].values
        ))
        
        # 生成融合预测
        fused_predictions = rf.predict(X_all)
        result_df[metal] = fused_predictions
    
    # 保存结果到Excel
    output_path = os.path.join(project_dir, 'predict_full.xlsx')
    result_df.to_excel(output_path, index=False)
    print(f"包含2003-2028年的融合预测结果已保存到 {output_path}")

if __name__ == "__main__":
    main()