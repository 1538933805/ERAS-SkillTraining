import csv
import random

def process_csv(input_file, output_file):
    # 读取CSV文件的所有行
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
    
    # 提取第三列数据并转换为浮点数
    third_column = []
    for row in rows:
        if len(row) < 3:
            third_column.append(0.0)  # 处理列不足的情况
        else:
            try:
                third_column.append(float(row[2]))
            except ValueError:
                third_column.append(0.0)  # 处理无效数据
    
    # 处理第三列数据
    divisor_num = 100000  # 对应1.0
    divisor_num_0 = divisor_num
    max_divisor_num = 110000  # 对应2.0
    start_index = 309999  # 第350000个数的索引（0-based）
    
    if len(third_column) > start_index:
        for i in range(start_index, len(third_column)):
            current_value = third_column[i]
            if (current_value > 0.01):
                if random.uniform(0.0,1.0) < 0.6:
                    divisor = divisor_num / divisor_num_0
                    third_column[i] = current_value / divisor
                # 递增除数，不超过max_divisor_num
                if divisor_num < max_divisor_num:
                    divisor_num += 1
    
    # 将处理后的数据写回行列表
    for i in range(len(rows)):
        if i < len(third_column):
            # 确保第三列存在
            if len(rows[i]) < 3:
                rows[i].extend([''] * (3 - len(rows[i])))
            rows[i][2] = str(third_column[i])
    
    # 写入新的CSV文件
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

# 示例用法
process_csv('input.csv', 'loss.csv')