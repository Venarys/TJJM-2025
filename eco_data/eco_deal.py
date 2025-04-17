import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def poly_extrapolate(df, year_col='Year', degree=10):
    """
    使用多项式插值填补经济指标列的开头缺失值。
    
    参数：
    df: pandas.DataFrame
        包含年份和经济指标的DataFrame，第一列为年份，其他列为指标。
    year_col: str
        年份列的列名，默认为'Year'。
    degree: int
        多项式阶数，默认为2（二次多项式）。可根据数据趋势调整（如1=线性，3=三次）。
        
    返回：
    pandas.DataFrame
        处理后的DataFrame，开头缺失值被填补。
    """
    df[year_col] = df[year_col].astype(int)
    years = df[year_col].values
    
    for col in df.columns:
        if col == year_col:
            continue
            
        data = df[col].values
        known_mask = ~np.isnan(data)
        
        if not any(known_mask):
            print(f"警告：列 '{col}' 全部缺失，无法处理！")
            continue
        
        known_years = years[known_mask]
        known_data = data[known_mask]
        
        try:
            # 动态选择多项式阶数（避免阶数过高）
            n_points = len(known_years)
            if n_points < 2:
                print(f"警告：列 '{col}' 有效数据点不足（{n_points}个），无法拟合！")
                continue
            current_degree = min(degree, n_points - 1)
            
            # 拟合多项式
            coeffs = np.polyfit(known_years, known_data, current_degree)
            poly = np.poly1d(coeffs)
            
            # 外推到缺失的开头年份
            missing_indices = np.nonzero(np.isnan(data))[0]
            if len(missing_indices) > 0:
                first_missing_year = years[missing_indices[0]]
                first_valid_year = known_years[0]
                
                if first_missing_year < first_valid_year:
                    min_year = years[0]
                    extrapolate_years = np.arange(min_year, first_valid_year)
                    
                    # 计算外推值并约束
                    extrapolate_values = poly(extrapolate_years)
                    extrapolate_values = np.maximum(extrapolate_values, 0)  # 非负约束
                    
                    # 添加数据爆炸保护（如设置上限为最大值的2倍）
                    max_value = np.max(known_data) * 2
                    extrapolate_values = np.minimum(extrapolate_values, max_value)
                    
                    for i, yr in enumerate(extrapolate_years):
                        idx = np.where(years == yr)[0][0]
                        df.at[idx, col] = extrapolate_values[i]
            
            print(f"列 '{col}' 成功拟合：阶数={current_degree}, 系数={coeffs.round(2)}")
        except Exception as e:
            print(f"警告：列 '{col}' 拟合失败！数据点：{known_years}, {known_data}")
            print(f"错误信息：{str(e)}")
    
    return df    
    
# ===== 主程序 =====
if __name__ == "__main__":
    # 1. 读取Excel文件
    input_file = "eco_data/economy_raw.xlsx"  # 替换为你的文件名
    output_file = "eco_data/economy.xlsx"
    
    df = pd.read_excel(input_file)
    
    # 执行多项式插值填补（默认二次多项式）
    df_filled = poly_extrapolate(df, year_col='Year', degree=5)
    
    # 4. 保存结果
    df_filled.to_excel(output_file, index=False)
    print(f"处理完成！结果已保存到 {output_file}")