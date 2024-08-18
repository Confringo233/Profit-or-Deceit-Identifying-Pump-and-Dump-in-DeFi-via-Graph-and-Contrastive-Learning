import pandas as pd
import os


def merge_csv(output_file_path, input_dir, base_name):
    # 获取目录中所有相关的CSV文件
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                 f.startswith(base_name) and f.endswith('.csv')]
    all_files.sort()  # 确保按顺序合并

    # 读取和合并所有文件
    combined_csv = pd.concat([pd.read_csv(f, encoding='utf-8') for f in all_files])

    # 保存合并后的CSV文件
    combined_csv.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Merged file saved as {output_file_path}")


output_file_path = 'data/pd.csv'
input_dir = 'data'
base_name = 'pd_part'  # 根据分割文件的命名规则来
merge_csv(output_file_path, input_dir, base_name)


