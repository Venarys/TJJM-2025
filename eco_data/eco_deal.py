import pandas as pd

def interpolate_excel(file_path, output_path):
    """
    对Excel文件中的缺失值进行LSTM插值处理
    因为这是初始几行，所以无法进行线性插值
    :param file_path: 原始Excel文件路径
    :param output_path: 处理后的输出文件路径
    """
    # 读取Excel文件（假设第一行是列名，第一列是数据列）
    df = pd.read_excel(file_path)
    
    # 替换特殊缺失值（根据实际数据修改）
    df = df.replace({'-': pd.NA, 'N/A': pd.NA})
    
    # 筛选出数值列（排除非数值列）
    numeric_cols = df.select_dtypes(include=['number']).columns
    print("正在处理的数值列:", numeric_cols)
    
    # 复制原始数据以保留非数值列
    df_interpolated = df.copy()
    
    # 线性插值设置（双向插值，允许填充顶部和底部缺失值）
    df_interpolated[numeric_cols] = df[numeric_cols].interpolate(
        method='linear',
        axis=0,
        limit_direction='both',  # 允许双向插值
        inplace=False
    )
    
    # 保存处理后的数据
    df_interpolated.to_excel(output_path, index=False)
    print(f"数据已保存至：{output_path}")

if __name__ == "__main__":
    input_file = 'economy.xlsx'          # 输入文件路径
    output_file = 'data_interpolated.xlsx'  # 输出文件路径
    interpolate_excel(input_file, output_file)