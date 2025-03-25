import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from random import uniform
# plt.rcParams['font.family'] = 'SimHei'  # 使用黑体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为宋体（SimSun）
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class ForceSensorFilter_Average:
    def __init__(self, window_size=10):
        self.queue = deque(maxlen=window_size)  # 创建固定大小的队列
        self.average_value = None

    def average_filter(self, FT):
        # 将新的测量值（六维向量）添加到队列中
        self.queue.append(FT)
        
        # 如果队列未满，则更新平均值
        if len(self.queue) < self.queue.maxlen:
            if self.average_value is None:
                self.average_value = FT
            else:
                self.average_value = np.mean(self.queue, axis=0)
        else:  # 如果队列已满，更新平均值并考虑移除的值
            removed_value = self.queue[0]  # 移除的是队列最前面的值
            self.average_value = ((self.average_value * len(self.queue)) 
                                  - removed_value + FT) / len(self.queue)
        return self.average_value



class ForceSensorFilter_Kalman:
    def __init__(self, Q, R):
        self.fx_last = None
        self.fy_last = None
        self.fz_last = None
        self.tx_last = None
        self.ty_last = None
        self.tz_last = None
        
        self.fx_p_last = Q
        self.fy_p_last = Q
        self.fz_p_last = Q
        self.tx_p_last = Q
        self.ty_p_last = Q
        self.tz_p_last = Q
        
        self.Q = Q
        self.R = R

    def kalman_filter(self, FT):
        # Initialize state estimates if they are not set yet
        if self.fx_last is None:
            self.fx_last = FT[0]
            self.fy_last = FT[1]
            self.fz_last = FT[2]
            self.tx_last = FT[3]
            self.ty_last = FT[4]
            self.tz_last = FT[5]

        # Perform Kalman filtering for each dimension
        fx_mid = self.fx_last 
        fx_p_mid = self.fx_p_last + self.Q 
        fx_kg = fx_p_mid / (fx_p_mid + self.R) 
        fx_now = fx_mid + fx_kg * (FT[0] - fx_mid)       
        fx_p_now = (1 - fx_kg) * fx_p_mid
        self.fx_p_last = fx_p_now  
        self.fx_last = fx_now 
        
        fy_mid = self.fy_last 
        fy_p_mid = self.fy_p_last + self.Q 
        fy_kg = fy_p_mid / (fy_p_mid + self.R) 
        fy_now = fy_mid + fy_kg * (FT[1] - fy_mid)       
        fy_p_now = (1 - fy_kg) * fy_p_mid
        self.fy_p_last = fy_p_now  
        self.fy_last = fy_now 
        
        fz_mid = self.fz_last 
        fz_p_mid = self.fz_p_last + self.Q 
        fz_kg = fz_p_mid / (fz_p_mid + self.R) 
        fz_now = fz_mid + fz_kg * (FT[2] - fz_mid)       
        fz_p_now = (1 - fz_kg) * fz_p_mid
        self.fz_p_last = fz_p_now  
        self.fz_last = fz_now 
        
        tx_mid = self.tx_last 
        tx_p_mid = self.tx_p_last + self.Q 
        tx_kg = tx_p_mid / (tx_p_mid + self.R) 
        tx_now = tx_mid + tx_kg * (FT[3] - tx_mid)       
        tx_p_now = (1 - tx_kg) * tx_p_mid
        self.tx_p_last = tx_p_now  
        self.tx_last = tx_now 
        
        ty_mid = self.ty_last 
        ty_p_mid = self.ty_p_last + self.Q 
        ty_kg = ty_p_mid / (ty_p_mid + self.R) 
        ty_now = ty_mid + ty_kg * (FT[4] - ty_mid)       
        ty_p_now = (1 - ty_kg) * ty_p_mid
        self.ty_p_last = ty_p_now  
        self.ty_last = ty_now 
        
        tz_mid = self.tz_last 
        tz_p_mid = self.tz_p_last + self.Q 
        tz_kg = tz_p_mid / (tz_p_mid + self.R) 
        tz_now = tz_mid + tz_kg * (FT[5] - tz_mid)       
        tz_p_now = (1 - tz_kg) * tz_p_mid
        self.tz_p_last = tz_p_now  
        self.tz_last = tz_now 
        
        return np.array([fx_now, fy_now, fz_now, tx_now, ty_now, tz_now])







if __name__ == '__main__':
    "均值滤波"
    filter_average = ForceSensorFilter_Average(window_size=30)
    "卡尔曼滤波"
    Q = 0.001  # Process noise variance
    R = 0.1    # Measurement noise variance
    filter_kalman = ForceSensorFilter_Kalman(Q=Q, R=R)

    "FT"
    FT = [np.array([uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1), uniform(-1,1)]) for i in range(1, 500)]  # 六维测量值

    raw_data = []
    filtered_data_average = []
    filtered_data_kalman = []

    for idx, measurement in enumerate(FT):
        raw_data.append(measurement[0])  # 只记录第一维的数据
        filtered_data_average.append(filter_average.average_filter(measurement)[0])  # 记录滤波后的第一维数据
        filtered_data_kalman.append(filter_kalman.kalman_filter(measurement)[0])  # 记录滤波后的第一维数据
        
    "绘制图表"
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(range(len(raw_data)), raw_data, label='Raw Data', marker='', linewidth=1, color=(0,0.5,1))
    plt.plot(range(len(filtered_data_average)), filtered_data_average, label='average_filter 均值滤波(window=30)', linestyle='-', marker='', linewidth=1.5, color=(1,0,0))
    plt.plot(range(len(filtered_data_kalman)), filtered_data_kalman, label='kalman_filter 卡尔曼滤波(Q=0.001, R=0.1)', linestyle='-', marker='', linewidth=1.5, color=(1,0.5,0))
    plt.title('Force Filter 力传感器滤波')
    plt.xlabel('Measurement Index')
    plt.ylabel('Force Value')
    plt.legend()
    plt.grid(True)


    plt.show()