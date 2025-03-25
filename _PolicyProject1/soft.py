import math
import numpy as np


class Soft():
    def __init__(self, m, k, m1, k1, dt, xi = 0.8,
                 F_threshold = 8, T_threshold = 2):
        self.m = m
        self.k = k      
        self.m1 = m1
        self.k1 = k1

        self.last_e = np.zeros(6)
        self.last_de = np.zeros(6)
        self.last_dde = np.zeros(6)
        
        #设置阻尼系数
        self.xi = xi
        self.b = math.sqrt(m*k)*2* xi
        self.b1 = math.sqrt(m1*k1)*2* xi
        self.dt = dt

        self.M = np.array([[self.m,0,0,0,0,0],
                           [0,self.m,0,0,0,0],
                           [0,0,self.m,0,0,0],
                           [0,0,0,self.m1,0,0],
                           [0,0,0,0,self.m1,0],
                           [0,0,0,0,0,self.m1]])
        
        self.B = np.array([[self.b,0,0,0,0,0],
                           [0,self.b,0,0,0,0],
                           [0,0,self.b,0,0,0],
                           [0,0,0,self.b1,0,0],
                           [0,0,0,0,self.b1,0],
                           [0,0,0,0,0,self.b1]])
        
        self.K = np.array([[self.k,0,0,0,0,0],
                           [0,self.k,0,0,0,0],
                           [0,0,self.k,0,0,0],
                           [0,0,0,self.k1,0,0],
                           [0,0,0,0,self.k1,0],
                           [0,0,0,0,0,self.k1]])
        self.force_threshold = np.array([F_threshold,F_threshold,F_threshold,T_threshold,T_threshold,T_threshold])
        self.out_threshold = np.array([0.004,0.004,0.004,0.001,0.001,0.001])
        self.velocity_threhold = np.array([0.8,0.8,0.8,0.1,0.1,0.1])
        
    def soft_control(self, input_pos_ort, FT, now_pos_ort):
        for i in range(6):
            if FT[i]>self.force_threshold[i]:
                FT[i]-=self.force_threshold[i]
            elif FT[i]<-self.force_threshold[i]:
                FT[i]+=self.force_threshold[i]
            else:
                FT[i]=0
        for i in range(3, 6):
                FT[i]*=8

        e_now = now_pos_ort - input_pos_ort

        dde = (FT - self.B * self.last_de - self.K * e_now) / self.M
        # 检查 dde 的形状并进行相应的处理
        if isinstance(dde, np.ndarray) and dde.ndim == 2 and dde.shape == (6, 6):
            # 如果 dde 是 6x6 的矩阵，则取主对角线元素
            dde = dde.diagonal()
        elif isinstance(dde, np.ndarray) and dde.ndim == 1 and dde.shape == (6,):
            # 如果 dde 已经是一维数组，则不需要进一步处理
            pass
        else:
            raise ValueError("dde 的形状不符合预期，应为 (6,) 或 (6, 6)")
        
        de = dde*self.dt + self.last_de
        for i in range(6):
            if de[i]>0.8:
                de[i]=0.8
            if de[i]<-0.8:
                de[i]=-0.8   
        e = (de + self.last_de)/2*self.dt + self.last_e
        
        self.last_dde = dde
        self.last_de = de
        self.last_e = e
        
        out_pos_ort = e + input_pos_ort

        return out_pos_ort, e

