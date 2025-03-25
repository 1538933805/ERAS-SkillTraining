import real_control
import time
import numpy as np
import math
import threading
import soft
import force_filter
import ftsensor_compensate
import utils
import sys
import signal
"脚本ctrl+c退出时调用"
def signal_handler(signal, frame):
    print('Caught Ctrl+C / SIGINT signal')
    UR5.close()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
    
    
UR5 = real_control.UR5_Real()
UR5.RTDE_SOFT_MODE = False
"力滤波器参数设置"
FT_filter = force_filter.ForceSensorFilter_Average(window_size=2)
"重力补偿参数设置"
Gt=12 # 设置质量
o_x=0.00458305; o_y=-0.00878349; o_z=0.77668899# 设置质心
FTsensor_Compensator = ftsensor_compensate.FTsensor_Compensate(Gt=Gt, o_x=o_x, o_y=o_y, o_z=o_z)


is_soft = True

if is_soft is False:
    UR5.moveFK([1.5107, -1.8707, 1.5107, -1.5107, -1.5107, 0], isDEG=False, acc=0.3, vel=0.6)

    print("走到初始位置")
    UR5.moveIK(pos=[5.91e-3, -645.81e-3, 200e-3], ort=[-3, 0.138, 3])
    
else:
    dt = 0.02
    
    isAssembleTest = False
    if isAssembleTest is True:
        softer = soft.Soft(m=5000,k=1000,
                        m1=1e9,k1=0,
                        dt=dt, xi=0.8)
        input_pos_ort = np.array([5.91e-3, -645.81e-3, 75e-3, 3.1415, 0., 0.])
    else:
        softer = soft.Soft(m=1000,k=200,
                        m1=3000,k1=600,
                        dt=dt, xi=2)
        # input_pos_ort = np.array([5.91e-3, -645.81e-3, 160e-3, -0.57, -3.1415, 0.57])
        input_pos_ort = np.array([5.91e-3, -645.81e-3, 160e-3, -3., 0.138, 3])
    # softer.b *= 1
    
    UR5.moveFK([1.5107, -1.8707, 1.5107, -1.5107, -1.5107, 0], isDEG=False)
    UR5.moveIK(pos=input_pos_ort[0:3], ort=input_pos_ort[3:6])
    
    while True:
        start_time = time.time()  # 记录开始时间
        
        if isAssembleTest is True:
            if  input_pos_ort[2]>0.04: input_pos_ort[2] -= 0.002
            # input_pos_ort[2] -= 0.002
        
        now_pos_ort = UR5.getRealPosOrt()
        now_ort = now_pos_ort[3:6]
        FT_output = UR5.readFTsensor()
        "重力补偿"
        isGravityCompensate = True
        if isGravityCompensate:
            FT_output = FTsensor_Compensator.compensate(FT_raw=FT_output, ftsensor_ort=now_ort, isPrintLog=True)
        "力滤波"
        # FT_output = FT_filter.average_filter(FT_output)
        
        "input_pos_ort以及now_pos_ort转换到传感器坐标系下"
        FTsensor_height = 0.15
        T_sensor_on_nowTcp = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, -FTsensor_height],
                                       [0, 0, 0, 1]])
        T_nowTcp_on_sensor = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, FTsensor_height],
                                       [0, 0, 0, 1]])
        T_inputTcp_on_base = utils.PosOrt_to_HomogeneousMatrix(input_pos_ort)
        T_nowTcp_on_base = utils.PosOrt_to_HomogeneousMatrix(now_pos_ort)
        T_sensor_on_base = np.dot(T_nowTcp_on_base, T_sensor_on_nowTcp)
        
        # T_base_on_sensor = utils.inverse_homogeneous_matrix_robotics(T_sensor_on_base)
        T_base_on_sensor = utils.inverse_homogeneous_matrix_np(T_sensor_on_base)
        
        T_inputTcp_on_sensor = np.dot(T_base_on_sensor, T_inputTcp_on_base)
        print('T_inputTcp_on_sensor:\n', np.array2string(T_inputTcp_on_sensor, precision=3, separator=', ', max_line_width=np.inf, suppress_small=True))
        
        PosOrt_nowTcp_on_sensor = utils.HomogeneousMatrix_to_PosOrt(T_nowTcp_on_sensor)
        PosOrt_inputTcp_on_sensor = utils.HomogeneousMatrix_to_PosOrt(T_inputTcp_on_sensor)
        
        "-----------打印------------"
        posOrt_sensor_on_base = utils.HomogeneousMatrix_to_PosOrt(T_sensor_on_base)
        posOrt_nowTcp_on_base = utils.HomogeneousMatrix_to_PosOrt(T_nowTcp_on_base)
        formatted_posOrt_sensor_on_base = np.array2string(posOrt_sensor_on_base, precision=3, separator=', ', max_line_width=np.inf, suppress_small=True)
        formatted_posOrt_nowTcp_on_base = np.array2string(posOrt_nowTcp_on_base, precision=3, separator=', ', max_line_width=np.inf, suppress_small=True)
        formatted_posOrt_nowTcp_on_sensor = np.array2string(PosOrt_nowTcp_on_sensor, precision=3, separator=', ', max_line_width=np.inf, suppress_small=True)
        formatted_posOrt_inputTcp_on_sensor = np.array2string(PosOrt_inputTcp_on_sensor, precision=3, separator=', ', max_line_width=np.inf, suppress_small=True)
        print("PosOrt_sensor_on_base: ", formatted_posOrt_sensor_on_base)
        print("PosOrt_nowTcp_on_base: ", formatted_posOrt_nowTcp_on_base)
        print("PosOrt_nowTcp_on_sensor: ", formatted_posOrt_nowTcp_on_sensor)
        print("PosOrt_inputTcp_on_sensor: ", formatted_posOrt_inputTcp_on_sensor)
        "---------------------------"
        
        SoftPosOrt_tcp_on_sensor, e = softer.soft_control(input_pos_ort=PosOrt_inputTcp_on_sensor , FT=FT_output, now_pos_ort=PosOrt_nowTcp_on_sensor)
        SoftT_tcp_on_sensor = utils.PosOrt_to_HomogeneousMatrix(SoftPosOrt_tcp_on_sensor)
        SoftT_tcp_on_base = np.dot(T_sensor_on_base, SoftT_tcp_on_sensor)
        SoftPosOrt_tcp_on_base = utils.HomogeneousMatrix_to_PosOrt(SoftT_tcp_on_base)
        soft_pos_ort = np.copy(SoftPosOrt_tcp_on_base)
        
        "设置仅姿态"
        # soft_pos_ort[3:6] = np.copy(input_pos_ort[3:6])
        # print("soft_pos_ort: ", soft_pos_ort)
        UR5.moveIK(pos=soft_pos_ort[0:3], ort=soft_pos_ort[3:6], threshold_time=0.5)
        # UR5.moveIK(pos=input_pos_ort[0:3], ort=input_pos_ort[3:6])
        # print("测试:\ninput_pos_ort: ", input_pos_ort, "\nnow_pos_ort: ", now_pos_ort, "\n")
        # test_now_pos_ort = utils.HomogeneousMatrix_to_PosOrt(T_nowTcp_on_base)
        # print("test_now_pos_ort: ", test_now_pos_ort)
        # UR5.moveIK(test_now_pos_ort[0:3], test_now_pos_ort[3:6])
        # UR5.moveIK(now_pos_ort[0:3], now_pos_ort[3:6])
        # UR5.setTCPPos(pos=soft_pos_ort)
        
        end_time = time.time()  # 记录结束时间
        dt = end_time - start_time  # 计算本次迭代耗时
        print("本次迭代耗时:", dt, "秒")
        softer.dt = dt
        
        # time.sleep(dt)

    