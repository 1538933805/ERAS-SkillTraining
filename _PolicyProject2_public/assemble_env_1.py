# circular shaft with rounded corners
import real_control
import utils
import realsenseD435_multi
import numpy as np
from random import uniform, choice
import math
import time
import keyboard

class AssembleEnv(object):
    def __init__(self, dt = 0., max_steps = 1000,
                 img_obs_height=128, img_obs_width=128, img_obs_channel=3, img_obs_offsetX=45, img_obs_offsetY=-125,
                 agent_obs_dim=12, action_dim=7, obs_horizon=4, action_horizon=8, pred_horizon=8):
        self.task_name = "circular shaft with rounded corners"
        #|o|o|                             observations: 2
        #| |a|a|a|a|a|a|a|a|               actions executed: 8
        #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.dt = dt #每步动作的间隔时间_重要!
        self.RAD2DEG = 180/math.pi
        self.DEG2RAD = math.pi/180
        self.rob = real_control.UR5_Real()
        self.camera=realsenseD435_multi.RealsenseD435Multi()
        self.max_steps = max_steps
        
        """observation"""
        self.obs_horizon = obs_horizon
        # agent_obs 部分
        """
        机器人的直接编码观察量
        - 前6维: 末端位姿(可不考虑计算相对,端到端)
        - 后6维: 末端力矩传感器量(可不考虑重力补偿,端到端)
        """
        self.agent_obs_dim = agent_obs_dim
        self.agent_obs = np.zeros(self.agent_obs_dim)
        self.agent_obs_list = np.zeros((self.obs_horizon, self.agent_obs_dim))
        # img_obs部分
        """
        机器人的视觉图像观察量
        - 形状: (self.img_obs_height, self.img_obs_width, self.img_obs_channel)
        """
        self.img_obs_height = np.array([img_obs_height])
        self.img_obs_width = np.array([img_obs_width])
        self.img_obs_channel = img_obs_channel
        self.img_obs_offsetX = np.array([img_obs_offsetX])
        self.img_obs_offsetY = np.array([img_obs_offsetY])
        self.img_obs = np.zeros((self.img_obs_height[0], self.img_obs_width[0], self.img_obs_channel))
        self.img_obs_list = np.zeros((self.obs_horizon, self.img_obs_height[0], self.img_obs_width[0], self.img_obs_channel))
        # 整体obs部分
        self.obs = {
            'agent_obs': self.agent_obs,
            'img_obs': self.img_obs,
            'task_name': self.task_name
        }
        self.latest_pos_ort = self.rob.getRealPosOrt()
        self.now_pos_ort = self.rob.getRealPosOrt()

        """action"""
        self.action_dim = action_dim
        """
        动作为7维
        - 前6维: 末端运动量
        _ 第7维: 对于夹爪 -- |1->夹| |0->松| 
        """
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.action = np.zeros(self.action_dim)
        self.latest_action = np.zeros(self.action_dim)
        self.action_list = np.zeros((self.action_horizon, self.action_dim))
        self.pred_list = np.zeros((self.pred_horizon, self.action_dim))
        
        """reward"""
        self.reward = 0.
        self.episode_reward = 0.
        self.mean_reward = 0.
        
        """step"""
        self.step_num = 0
        
        """flags"""
        self.done = False #回合结束的标志:装配最终成功，或者装配失败时，done都从False到True
        self.success_flag = False #最终所有装配是否完成的标志
        self.state_assemble = 0 #装配阶段
        
        """others"""
        self.rotation_limit_list = [0.5 ,0.5, 4]  # 旋转限制（单位：度）
        self.is_rotation_limit = np.array([True, True, True])
        
        return
    
    
    def _get_obs(self, isPrintInfo=False, isShowImgs=True):
        self.latest_pos_ort = np.copy(self.now_pos_ort)
        self.now_pos_ort = self.rob.getRealPosOrt()
        self.agent_obs[0:6] = np.copy(self.now_pos_ort)
        FT_output = self.rob.readFTsensor()
        self.agent_obs[6:12] = np.copy(FT_output)
        color_images, depth_images, gray_depth_images = self.camera.get_data(sample_skip=0,
                                                                    get_w=self.img_obs_width, get_h=self.img_obs_height,
                                                                    offset_x=self.img_obs_offsetX, offset_y=self.img_obs_offsetY,
                                                                    is_show_window=isShowImgs)
        self.img_obs = np.copy(color_images[0])
        self.obs = {
            'agent_obs': self.agent_obs,
            'img_obs': self.img_obs,
            'task_name': self.task_name
        }
        if isPrintInfo: print("obs['agent_obs']的形状: ", self.obs['agent_obs'].shape, "\n"
                              "obs['img_obs']的形状: ", self.obs['img_obs'].shape)
        return self.obs
    
    
    def reset(self, isAddError=True, is_drag_mode=False, error_scale_pos=1, error_scale_ort=1,
              initial_pos_ort=np.array([29.77e-3, -440.28e-3, 128e-3, 0, -3.1415, 0])):
        self.initial_pos_ort = np.copy(initial_pos_ort)
        "设置位置误差"
        if isAddError:
            error_rad = 180.0 * uniform(-1.0, 1.0) * self.DEG2RAD * error_scale_pos
            self.initial_pos_ort[0] += 0.01 * math.cos(error_rad) * error_scale_pos * uniform(0.4, 1.0)
            self.initial_pos_ort[1] += 0.01 * math.sin(error_rad) * error_scale_pos * uniform(0.4, 1.0)
            error_direction = choice([-1, 1])
            self.initial_pos_ort[5] += uniform(0,10) * error_direction * self.DEG2RAD * error_scale_ort
        if is_drag_mode is True:
            self.rob.moveIK_urx(self.initial_pos_ort[0:3], self.initial_pos_ort[3:6], wait=True)
        else:
            self.rob.moveIK(self.initial_pos_ort[0:3], self.initial_pos_ort[3:6], wait=True)
            # self.rob.moveIK_urx(self.initial_pos_ort[0:3], self.initial_pos_ort[3:6], wait=True)
        self.obs = self._get_obs()
        self.update_list_env()
        self.reward = 0.
        self.episode_reward = 0.
        self.step_num = 0.
        self.success_flag = False
        self.done = False
        "step阶段的运动限制"
        self.cumulative_rotation = np.zeros(3)  # 累积旋转量
        return self.obs
    
    def step(self, action, isPrintInfo=False):
        """step运动
        Args:
            action (7维数组): 
                - 前3维 平动量 单位mm
                - 中3维 转动量 单位deg
                - 最后1维 夹爪 1->夹 0->松
        """
        self.action = np.copy(action)
        # 检查并调整旋转量
        for i in range(3):
            if self.is_rotation_limit[i] is True:
                if abs(self.cumulative_rotation[i]+action[3 + i]) > abs(self.rotation_limit_list[i]):
                    action[3 + i] = 0
                self.cumulative_rotation[i] += action[3 + i]
        
        action_real = np.zeros(7)
        action_real[0:3] = [x * 0.001 for x in action[0:3]]  
        action_real[3:6] = [x * self.DEG2RAD for x in action[3:6]]  # 将度转换为弧度，并应用
        action_real[6] = 1
        # last_pos_ort = self.rob.getRealPosOrt()
        if action is not None:
            self.rob.moveIK_changePosOrt_onEnd(d_pos=action_real[0:3], d_ort=action_real[3:6], threshold_time=0.5, wait=True)
            if (self.action[6] > 0.5) and (self.latest_action[6] < 0.5): self.rob.closeRG2()
            if (self.action[6] < 0.5) and (self.latest_action[6] > 0.5): self.rob.openRG2()
            time.sleep(self.dt)
        # now_pos_ort = self.rob.getRealPosOrt()
        # d_pos_ort = self.rob.get_d_pos_ort_onEnd(last_pos_ort, now_pos_ort)
        # print("命令的较末端的装配运动为:")
        # utils.print_array_with_precision(action_real, n=3)
        # print("计算的较末端的差分装配运动为:")
        # utils.print_array_with_precision(d_pos_ort, n=3)
        self.obs = self._get_obs(isPrintInfo=False)
        self.update_list_env()
        self.get_reward(isPrintInfo=isPrintInfo)
        self.check_episodeEnd(isPrintInfo=isPrintInfo)
        self.latest_action = np.copy(action)
        self.step_num += 1
        if isPrintInfo:
            print("当前第{}次逆运动学动作  动作为{}  执行后(指令)位置为{}".format(self.step_num,  [f"{x:.2f}" for x in self.action[:3]], [f"{x:.4f}" for x in self.agent_obs]))
        return self.obs, self.action, self.reward, self.success_flag, self.done
    
    
    def step_by_drag(self, isPrintInfo=False):
        """拖动示教下的step运动
        Args:
            action (7维数组): 
                - 前3维 平动量 单位mm
                - 中3维 转动量 单位deg
                - 最后1维 夹爪 1->夹 0->松
        """
        # self.rob.closeRG2()
        # time.sleep(0.5)
        time.sleep(0.1)
        self.obs = self._get_obs(isPrintInfo=False)
        "原先是相对于基坐标系的"
        # self.action[0:6] = np.copy(self.now_pos_ort-self.latest_pos_ort)
        "改为相对于末端执行器自己位姿的动作"
        self.action[0:6] = self.rob.get_d_pos_ort_onEnd(last_pos_ort=np.copy(self.latest_pos_ort),
                                                        now_pos_ort=np.copy(self.now_pos_ort))
        self.action[0:3] = [x * 1000 for x in self.action[0:3]]
        self.action[3:6] = utils.normalize_angles(self.action[3:6])
        self.action[3:6] = [x * self.RAD2DEG for x in self.action[3:6]]  # 将弧度转换为度，并应用  
        self.action[6] = 1

        self.update_list_env()
        self.get_reward(isPrintInfo=isPrintInfo)
        self.check_episodeEnd(isPrintInfo=isPrintInfo)
        
        self.latest_action = np.copy(self.action)
        self.step_num += 1
        
        if isPrintInfo:
            print("拖动示教第{}次逆运动学动作  动作为{}  执行后(指令)位置为{}".format(self.step_num,  [f"{x:.2f}" for x in self.action[:3]], [f"{x:.4f}" for x in self.agent_obs]))
        return self.obs, self.action, self.reward, self.success_flag, self.done
    
    
    def get_reward(self, isPrintInfo=False):
        self.reward = 0.
        self.assembleDepth = np.linalg.norm([self.initial_pos_ort[2]-self.agent_obs[2]], ord=2)
        depth_reward = 50 * self.assembleDepth
        self.reward = depth_reward
        self.episode_reward += self.reward
        if isPrintInfo:
            print("reward: {}\t深度奖励: {}".format(self.reward, depth_reward))
            
    def check_episodeEnd(self, isPrintInfo=False):
        if (self.assembleDepth > 0.4):
            self.success_flag = True
            self.done = True
        elif self.step_num>=self.max_steps-1:
            self.success_flag = False
            self.done = True
            if isPrintInfo: print("本回合step已达上限")
        if self.done:
            self.mean_reward = self.episode_reward / (self.step_num+1)
            if isPrintInfo: print("本回合结束,任务完成情况:{}\t平均奖励为: {}\t回合奖励为: {}".format(self.success_flag, self.mean_reward, self.episode_reward))
            
    
    def update_list_env(self):
        self.agent_obs_list = self.update_list(self.agent_obs, self.agent_obs_list, self.obs_horizon, info="agent_obs", isPrintInfo=False)
        self.img_obs_list = self.update_list(self.img_obs, self.img_obs_list, self.obs_horizon, info="img_obs", isPrintInfo=False)
        # img_obs_list_flattern = self.img_obs_list.reshape(self.obs_horizon, -1)
        # print(np.linalg.norm(img_obs_list_flattern[0] - img_obs_list_flattern[1]), "\t(测试:采样两步图像信息差的L2范数)") # 计算 L2 范数
        self.action_list = self.update_list(self.action, self.action_list, self.action_horizon, info="action", isPrintInfo=False)
    
    def update_list(self, data, data_list, horizon, info="", isPrintInfo=False):
        # data_list = np.zeros((horizon, data.shape[0]))
        if self.step_num < 1:
            for i in range(horizon):
                data_list[i] = np.copy(data)
        else:
            for i in range(horizon-1):
                data_list[i] = np.copy(data_list[i+1])
            data_list[horizon-1] = np.copy(data)
        if isPrintInfo:
            print("得到的",info, "的data_list为:\n", data_list)
        return data_list
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    env = AssembleEnv()
    is_drag_mode=False
    isPrintStepInfo = True
    while True:
        print("重置.......")
        # env.reset(isAddError=False)
        env.reset(isAddError=False, initial_pos_ort=np.array([29.77e-3, -440.28e-3, 133.83e-3, 0.4, -3.1415, 0.4]))
        env.is_rotation_limit = np.array([False, False, False])
        print("环境重置完成...\t状态观测为{}".format([f"{x:.4f}" for x in env.agent_obs]))
        if is_drag_mode:
            print("按s开始本回合,注意手动打开onRobot拖动示教模式\n启用后若提前完成,则按d终止示教回合")
            while True:
                env._get_obs(isPrintInfo=False)
                if keyboard.is_pressed("s"): break
        else:
            print("按s开始本回合")
            while True:
                env._get_obs(isPrintInfo=False)
                if keyboard.is_pressed("s"): break
        while True:
            print("---------------------------------------------------------------------")
            env.obs = env._get_obs()
            if is_drag_mode is False:
                env.step([uniform(-2,2), uniform(-2,2), uniform(-1,1), 
                          uniform(-1,1), uniform(-1,1), uniform(-1,1), 1], isPrintInfo=isPrintStepInfo)
                env.step([0, 0, -2, 0, 0, 2, 1], isPrintInfo=isPrintStepInfo)
                # env.step([1, 1, -2, -2, 0, 0, 1], isPrintInfo=isPrintStepInfo)
                # env.step([1, 1, -2, 0, 2, 0, 1], isPrintInfo=isPrintStepInfo)
                if env.done: break
            else:
                if env.done:
                    print("先关闭onRobot拖动示教模式!!! 不要再拖动,否则保护性停止 \n 然后按e结束本次示教")
                    while True:
                        if keyboard.is_pressed("e"): break
                    break
                else:
                    if keyboard.is_pressed("d"): env.done = True
                    env.step_by_drag(isPrintInfo=isPrintStepInfo)
        print("---------------------------------------------------------------------")
        print("---------------------------------------------------------------------")
    
    
    
    
    
    

    
    