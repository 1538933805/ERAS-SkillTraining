import pandas as pd
import numpy as np



# 加载CSV文件
df = pd.read_csv(".\\_PolicyProject1\\_teach1_dataset\\action_原_360错误.csv")

def normalize_angle_deg(angles):
    # 将角度差值归一化到[-pi, pi]范围内
    normalized_angles = []
    for angle in angles:
        while angle > 180:
            angle -= 2 * 180
        while angle < -180:
            angle += 2 * 180
        normalized_angles.append(angle)
    return np.array(normalized_angles)

# 提取第3到6列
cols_of_interest = df.iloc[:, 3:6]

# 对每一行进行处理
for index, row in cols_of_interest.iterrows():
    # 这里你可以添加自己的处理逻辑，例如：
    processed_row = normalize_angle_deg(row.values)
    
    # 更新原始DataFrame
    df.loc[index, cols_of_interest.columns] = processed_row

# 保存处理过的CSV文件
df.to_csv((".\\_PolicyProject1\\_teach1_dataset\\action.csv"), index=False)



# 加载CSV文件
df = pd.read_csv(".\\_PolicyProject1\\_teach1_dataset\\agent_obs_原_360错误.csv")

def normalize_angle_rad(angles):
    # 将角度差值归一化到[-pi, pi]范围内
    normalized_angles = []
    for angle in angles:
        # print(angle, np.pi)
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        normalized_angles.append(angle)
        # print(normalized_angles)
    return np.array(normalized_angles)

# 提取第3到6列
cols_of_interest = df.iloc[:, 3:6]

# 对每一行进行处理
for index, row in cols_of_interest.iterrows():
    # 这里你可以添加自己的处理逻辑，例如：
    processed_row = normalize_angle_rad(row.values)
    # print(processed_row)
    
    # 更新原始DataFrame
    df.loc[index, cols_of_interest.columns] = processed_row

# 保存处理过的CSV文件
df.to_csv((".\\_PolicyProject1\\_teach1_dataset\\agent_obs.csv"), index=False)
