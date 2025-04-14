import pandas as pd
import numpy as np
import os
from glob import glob

def main():
    metals = ['CU', 'PB', 'ZN', 'CR', 'CD', 'HG', 'AS']
    years = list(range(1998, 2025))
    
    data_frames = []
    
    for file in glob("*.xlsx"):
        try:
            df = pd.read_excel(file, index_col=0)
            df.columns = [col.strip().upper() for col in df.columns]
            
            df_reset = df.reset_index()
            id_col = df_reset.columns[0]
            df_melt = df_reset.melt(
                id_vars=id_col,
                var_name='金属',
                value_name='值'
            )
            
            df_melt.rename(columns={id_col: '年份'}, inplace=True)
            df_melt['年份'] = pd.to_numeric(df_melt['年份'], errors='coerce')
            df_melt = df_melt.dropna(subset=['年份', '值'])
            df_melt['年份'] = df_melt['年份'].astype(int)
            df_melt['金属'] = df_melt['金属'].str.upper().str.strip()
            
            data_frames.append(df_melt)
            print(f"成功处理文件 {file}，数据量：{len(df_melt)}")
            
        except Exception as e:
            print(f"文件 {file} 处理失败：{str(e)}")
    
    # 合并数据
    combined = pd.concat(data_frames, ignore_index=True)
    combined = combined[combined['金属'].isin(metals)]
    
    # 分组计算平均值
    grouped = combined.groupby(['年份', '金属'])['值'].mean().reset_index()
    
    # 转换为宽格式并插值
    pivot_df = grouped.pivot(index='年份', columns='金属', values='值')
    
    # 1. 使用NaN填充缺失值（原fill_value=0改为np.nan）
    pivot_df = pivot_df.reindex(index=years, columns=metals, fill_value=np.nan)
    
    # 2. 线性插值填补NaN（按年份方向插值）
    pivot_df = pivot_df.interpolate(method='linear', axis=0).ffill().bfill()
    
    # 3. 保留三位小数
    pivot_df = pivot_df.round()
    
    # 保存结果
    pivot_df.to_excel("enviroments.xlsx")
    print("文件已保存！")

if __name__ == "__main__":
    main()