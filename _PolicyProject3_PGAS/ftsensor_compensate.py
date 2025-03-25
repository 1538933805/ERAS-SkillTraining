# -*- coding:utf-8 -*-
import time
import math
import numpy as np
from random import uniform
import real_control

def rpy_to_rotation_xyz(roll, pitch ,yaw):
    R_x = np.array([[1,                            0,               0],
                    [0,               math.cos(roll), -math.sin(roll)],
                    [0,               math.sin(roll),  math.cos(roll)]])              
    R_y = np.array([[ math.cos(pitch),             0,  math.sin(pitch)],
                    [               0,             1,               0],
                    [-math.sin(pitch),             0,  math.cos(pitch)]])        
    R_z = np.array([[math.cos(yaw),   -math.sin(yaw),               0],
                    [math.sin(yaw),    math.cos(yaw),               0],
                    [            0,                0,               1]])      
    rot = np.dot(R_x, np.dot(R_y, R_z))
    return rot

def rpy_to_rotation_zyx(roll, pitch ,yaw):
    R_x = np.array([[1,                            0,               0],
                    [0,               math.cos(roll), -math.sin(roll)],
                    [0,               math.sin(roll),  math.cos(roll)]])              
    R_y = np.array([[ math.cos(pitch),             0,  math.sin(pitch)],
                    [               0,             1,               0],
                    [-math.sin(pitch),             0,  math.cos(pitch)]])        
    R_z = np.array([[math.cos(yaw),   -math.sin(yaw),               0],
                    [math.sin(yaw),    math.cos(yaw),               0],
                    [            0,                0,               1]])      
    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class FTsensor_Compensate(object):
    def __init__(self, Gt=10,
                 o_x=0, o_y=0, o_z=0.1):
        """_summary_
        Args:
            对于夹爪及夹取工件的对应量: 
            质量 Gt (float, optional): _description_.  Defaults to 10
            质心 o_x (_type_, optional): _description_. Defaults to 0
            质心 o_y (_type_, optional): _description_. Defaults to 0
            质心 o_z (_type_, optional): _description_. Defaults to 0.1
        """
        self.Gt = Gt
        self.o_x = o_x
        self.o_y = o_y
        self.o_z = o_z
        self.F_Gb = np.array([[0],[0],[-self.Gt]])    # 工具重力在基坐标系

    def compensate(self, FT_raw, ftsensor_ort, isPrintLog=False):
        r = ftsensor_ort[0]
        p = ftsensor_ort[1]
        y = ftsensor_ort[2]
        # 计算工具重力的分量
        R_sb = rpy_to_rotation_zyx(r, p, y)     # 基坐标系和传感器坐标系的旋转矩阵
        F_Gs = np.dot(np.linalg.inv(R_sb), self.F_Gb)
        o_xyz = np.array([[        0, -self.o_z,  self.o_y],
                            [ self.o_z,         0, -self.o_x],
                            [-self.o_y,  self.o_x,         0]])
        T_Gs = np.dot(o_xyz, F_Gs)

        # 计算标定后的力矩数据
        fx_calib = FT_raw[0] - F_Gs[0][0]
        fy_calib = FT_raw[1] - F_Gs[1][0]
        fz_calib = FT_raw[2] - F_Gs[2][0]
        tx_calib = FT_raw[3] - T_Gs[0][0]
        ty_calib = FT_raw[4] - T_Gs[1][0]
        tz_calib = FT_raw[5] - T_Gs[2][0]

        if isPrintLog is True:
            print("\n------------------力矩传感器重力补偿-----------------------")
            print("补偿前: x={:.3f},y={:.3f},z={:.3f}\n力矩:x={:.3f},y={:.3f},z={:.3f}"
                .format(FT_raw[0], FT_raw[1], FT_raw[2], FT_raw[3], FT_raw[4], FT_raw[5]))
            print("补偿后: x={:.3f},y={:.3f},z={:.3f}\n力矩:x={:.3f},y={:.3f},z={:.3f}"
                .format(fx_calib, fy_calib, fz_calib, tx_calib, ty_calib, tz_calib))
            print("补偿后z轴的力：{:.3f}".format(fz_calib))
            print("----------------------------------------------------\n")
        return np.array([fx_calib, fy_calib,fz_calib,tx_calib,ty_calib,tz_calib])
    
    
if __name__ == '__main__':
    "运用空载时对应的参数"
    Gt=9.2 # 设置质量
    o_x=0.00458305; o_y=-0.00878349; o_z=0.77668899# 设置质心
    FTsensor_Compensator = FTsensor_Compensate(Gt=Gt, o_x=o_x, o_y=o_y, o_z=o_z)
    UR5=real_control.UR5_Real()
    UR5.moveIK(pos=[5.91e-3, -645.81e-3, 160e-3], ort=[-0., -3.1415, 0.])
    
    while True:
        Fx, Fy, Fz, Tx, Ty, Tz = UR5.readFTsensor()
        FT_output = np.array([Fx, Fy, Fz, Tx, Ty, Tz])
        now_pos_ort = UR5.getRealPosOrt()
        now_ort = now_pos_ort[1]
        FTsensor_Compensator.compensate(FT_raw=FT_output, ftsensor_ort=now_ort, isPrintLog=True)
        time.sleep(0.5)
    
