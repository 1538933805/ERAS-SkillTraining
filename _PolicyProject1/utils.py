# -*- coding: utf-8 -*-

import math
import numpy as np


def rpy_to_rotation(roll, pitch ,yaw):
    # UR5采用XYZ正向旋转欧拉角,则xyz依次左乘
    rot = rpy_to_rotation_zyx(roll=roll, pitch=pitch, yaw=yaw)
    # rot = rpy_to_rotation_xyz(roll=roll, pitch=pitch, yaw=yaw)
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


def rot2euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
 
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
 
    return np.array([x, y, z])


def control_rad(orientation):
    for i in range(len(orientation)):
        if orientation[i] < -1*math.pi:
            orientation[i] = orientation[i]+2*math.pi
        elif orientation[i] > math.pi:
            orientation[i] = orientation[i]-2*math.pi
        elif 3.13 <= abs(orientation[i]) <= math.pi:
            orientation[i] = abs(orientation[i])
    return orientation


"六维位姿向量 to 4*4齐次矩阵"
def PosOrt_to_HomogeneousMatrix(pos_ort):
    rot = rpy_to_rotation(pos_ort[3], pos_ort[4], pos_ort[5])
    p = pos_ort[0:3]
    # print('rot: ', rot)
    # print('p: ', p)
    # 创建齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = p
    return T

"4*4齐次矩阵 to 六维位姿向量"
def HomogeneousMatrix_to_PosOrt(homogeneous_matrix):
    rot = homogeneous_matrix[:3, :3]
    p = homogeneous_matrix[:3, 3]
    # print('rot: ', rot)
    # print('p: ', p)
    # 从旋转矩阵中提取RPY角
    rpy = rot2euler(rot)
    pos_ort = np.concatenate([p, rpy], axis=0)
    return pos_ort

"np求逆方法计算齐次矩阵的逆"
def inverse_homogeneous_matrix_np(T_ab):
    # 直接使用numpy计算逆矩阵
    T_ba = np.linalg.inv(T_ab)
    return T_ba

"解析方法计算齐次矩阵的逆"
def inverse_homogeneous_matrix_robotics(T_ab):
    # 首先确保输入是一个4x4的矩阵
    assert T_ab.shape == (4, 4), "Input must be a 4x4 homogeneous transformation matrix."
    # 提取旋转和平移部分
    R = T_ab[:3, :3]
    p = T_ab[:3, 3]
    # 计算旋转矩阵的转置
    R_inv = R.T
    # 计算新的平移向量
    p_inv = -np.dot(R_inv, p)
    # 构建逆变换矩阵
    T_ba = np.eye(4)
    T_ba[:3, :3] = R_inv
    T_ba[:3, 3] = p_inv
    return T_ba


if __name__ == "__main__":
    (roll, pitch ,yaw) = (3.14,0,-0)
    print(rpy_to_rotation(roll, pitch ,yaw))
    print("\n")
    pos_ort = [5.91e-3, -645.81e-3, 160e-3, -0.57, -3.1415, 0.57]
    print("\npos_ort:\n", pos_ort)
    T = PosOrt_to_HomogeneousMatrix(pos_ort)
    print("\n齐次矩阵:\n", T)
    pos_ort_restore = HomogeneousMatrix_to_PosOrt(T)
    print("\n还原的pos_ort_restore:\n", pos_ort_restore)
    T1 = inverse_homogeneous_matrix_np(T)
    print("\n逆矩阵(np法):\n", T1)
    T1_rob = inverse_homogeneous_matrix_robotics(T)
    print("\n逆矩阵(解析法):\n", T1_rob)
    T_restore = inverse_homogeneous_matrix_np(T1)
    print("\n逆矩阵恢复(np法):\n", T_restore)
    T_restore_rob = inverse_homogeneous_matrix_np(T1_rob)
    print("\n逆矩阵恢复(解析法):\n", T_restore_rob)

