# -*- coding: utf-8 -*-
import time
import math
import numpy as np
from random import uniform
import real_control
# from utils import rpy_to_rotation

class ParamIdentify(object):
    
    def __init__(self):
        self.RAD2DEG = 180/math.pi
        self.UR5 = real_control.UR5_Real()

    def move(self):
        goal_joints = [0, 0, 0, 0, 0, 0]
        for j in range(6): 
            goal_joints[j] = uniform(-2, 2)
        self.UR5.moveFK(joints_positions=goal_joints, relative=True)

    def run(self):
        F_x = []
        F_y = []
        F_z = []
        T_x = []
        T_y = []
        T_z = []
        self.UR5.moveIK(pos=[5.91e-3, -545.81e-3, 260e-3], ort=[-0., -3.1415, 0.])
        
        for i in range(30):
            self.move()
            time.sleep(2.0)

            force = [0, 0, 0]
            torque = [0, 0, 0]

            num = 100

            for j in range(num):
                FT = self.UR5.readFTsensor()
                force[0] += FT[0]
                force[1] += FT[1]
                force[2] += FT[2]
                torque[0] += FT[3]
                torque[1] += FT[4]
                torque[2] += FT[5]
            F_x.append(force[0]/num)
            F_y.append(force[1]/num)
            F_z.append(force[2]/num)
            T_x.append(torque[0]/num)
            T_y.append(torque[1]/num)
            T_z.append(torque[2]/num)

            F_xyz = np.array([[      0,  F_z[i], -F_y[i], 1, 0, 0],
                              [-F_z[i],       0,  F_x[i], 0, 1, 0],
                              [ F_y[i], -F_x[i],       0, 0, 0, 1]])
            T_xyz = np.array([[T_x[i]], [T_y[i]], [T_z[i]]])

            if i == 0:
                F_matrix = F_xyz
                T_matrix = T_xyz
            else:
                F_matrix = np.concatenate((F_matrix, F_xyz), axis=0)
                T_matrix = np.concatenate((T_matrix, T_xyz), axis=0)

        a = np.dot(np.dot(np.linalg.inv(np.dot(F_matrix.T, F_matrix)), F_matrix.T), T_matrix)
        print("A = ", a)
        
        
if __name__ == '__main__':  
    param_identify = ParamIdentify()
    param_identify.run()