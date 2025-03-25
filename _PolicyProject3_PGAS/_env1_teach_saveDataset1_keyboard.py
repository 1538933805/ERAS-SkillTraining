# %% [markdown]
# 圆轴（带圆角）装配1
# 
# 示教并存储示教数据集

# %%
import assemble_env_1
from random import uniform,choice
import numpy as np
import time
import os
import pandas as pd
import cv2
import zipfile
import h5py
import torch
import keyboard
dataset_path =      (".\\_PolicyProject3_PGAS\\_trained_models_env1\\_teach1_dataset")
dataset_zip_path =  (".\\_PolicyProject3_PGAS\\_trained_models_env1\\_teach1_dataset.zip")
dataset_hdf5_path = (".\\_PolicyProject3_PGAS\\_trained_models_env1\\_teach1_dataset.hdf5")

# %%
def try_to_csv(file_path, df, info="", index=False, header=False, mode='w', isPrintInfo=False):
    # 检查文件夹是否存在，如果不存在则创建
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    while True:
        try:
            df.to_csv(file_path, index=index, header=header, mode=mode)
            break
        except Exception as e:
            if isPrintInfo: print("本次"+info+"数据写入csv失败,尝试重新写入...")
    
def try_read_csv(file_path, info="", header=None, isPrintInfo=False):  
    while True:  
        try:  
            # 检查文件是否为空  
            if os.path.getsize(file_path) == 0:  
                if isPrintInfo: print(f"{info}文件为空。")
                return pd.DataFrame()
            # 读取文件的前几行以检查是否有有效数据  
            with open(file_path, 'r') as f:  
                first_line = f.readline().strip()  
                if not first_line:  
                    if isPrintInfo: print(f"{info}文件没有有效数据。")
                    return pd.DataFrame()      
            csv = pd.read_csv(file_path, header=header)
            # print(csv)  
            return csv  
        except Exception as e:  
            if isPrintInfo:  
                print(f"本次{info}读取csv失败, 错误信息: {e}，尝试重新读取...")  

def try_read_csv_and_toList(file_path, info="", header=None, isPrintInfo=False):
    csv_df = try_read_csv(file_path, info, header, isPrintInfo)
    # print(csv_df)
    if csv_df.empty:
        ans = np.array([])
    else:
        ans = csv_df.iloc[:].to_numpy()
    return ans.tolist()

"递归获取list维度的方法"
def get_dimensions(lst):  
    dimensions = []  
    def recursive_length(lst, level=0):  
        # 确保 dimensions 列表足够长  
        if level >= len(dimensions):  
            dimensions.append(0)  
        if isinstance(lst, list):  
            dimensions[level] = max(dimensions[level], len(lst))  
            for item in lst:  
                recursive_length(item, level + 1)  
    recursive_length(lst)  
    return dimensions

def try_to_image(image_data, output_path, height, width, channel, info="图像", isPrintInfo=False):
    # 确保输入数据形状正确
    expected_shape = (height, width, channel)
    if image_data.shape != expected_shape: raise ValueError(f"输入数据形状({image_data.shape})与预期形状({expected_shape})不符")
    if isPrintInfo: print(f"{info}原始数据形状:", image_data.shape)
    # 如果是灰度图，需要确保数据是单通道
    if channel == 1 and image_data.ndim == 3: image_data = image_data.squeeze(axis=-1)
    # 保存图像
    while True:
        try:
            if channel == 1:
                # 对于灰度图，使用cv2.IMWRITE_PNG_COMPRESSION设置压缩级别
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                success = cv2.imwrite(output_path, image_data, compression_params)
            else:
                # 对于彩色图，直接保存
                success = cv2.imwrite(output_path, image_data)
            if not success: raise IOError("保存图像时发生错误")
            if isPrintInfo: print(f"{info}已成功保存为PNG图像至 {output_path}")
            break
        except Exception as e:
            if isPrintInfo: print(f"{info}保存图像失败: {e}")
                
def create_zip_from_folder(folder_path, output_zip_path):  
    # 检查输入路径是否存在  
    if not os.path.exists(folder_path):  
        raise ValueError(f"指定的文件夹路径不存在: {folder_path}")  
    # 创建一个ZipFile对象，用于写入数据  
    try:  
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:  
            # 遍历指定文件夹中的所有文件  
            for root, dirs, files in os.walk(folder_path):  
                for file in files:  
                    # 构建完整的文件路径  
                    file_path = os.path.join(root, file)  
                    # 在ZIP文件中存储的文件名（移除路径前缀）  
                    arcname = os.path.relpath(file_path, start=folder_path)  
                    # 添加文件到ZIP文件中  
                    zipf.write(file_path, arcname)  
        print(f"成功创建ZIP文件: {output_zip_path}")  
    except Exception as e:  
        print(f"创建ZIP文件时出错: {e}")
        
def create_hdf5_from_folder(folder_path, output_hdf5_path):
    # 检查输入路径是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"指定的文件夹路径不存在: {folder_path}")
    try:
        # 创建一个HDF5文件
        with h5py.File(output_hdf5_path, 'w') as hdf5f:
            # 遍历指定文件夹中的所有文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 为每个文件创建一个数据集，数据集的名称为文件相对路径
                    relative_path = os.path.relpath(file_path, start=folder_path)
                    # 使用相对路径作为数据集的名称
                    dataset_name = relative_path.replace(os.sep, '/')
                    # 读取文件内容（例如图片、文本等），这里只是示范以读取为二进制文件
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    # 将数据存储到HDF5文件
                    hdf5f.create_dataset(dataset_name, data=np.void(data))  # 将二进制数据存储为 `np.void`
        print(f"成功创建HDF5文件: {output_hdf5_path}")
    except Exception as e:
        print(f"创建HDF5文件时出错: {e}")

# %%
# device transfer
device = torch.device('cuda')

env = assemble_env_1.AssembleEnv()
isPrintStepInfo = True
"设置是否启用示教模式"
isUseTeachAction = True
"设置是否存储示教数据"
isSaveTeachData = True

# max_episodes = 1000

# %%
print("示教文件夹的路径:\t", dataset_path)

# %%
create_zip_from_folder(dataset_path, dataset_zip_path)
create_hdf5_from_folder(dataset_path, dataset_hdf5_path)

# %%
"load episode_ends"
episode_ends_path = os.path.join(dataset_path, "episode_ends.csv")
print("episode_ends的路径:\t", episode_ends_path)
tmp_id = -1
episode_ends = try_read_csv_and_toList(episode_ends_path, isPrintInfo=True)
episode_ends = torch.tensor(episode_ends).to(device)
# print(episode_ends)
if len(episode_ends) > 0: tmp_id = int(episode_ends[-1][-1]) - 1
print("tmp_id:", tmp_id)   

# %%
"load agent_obs"
agent_obs_path = os.path.join(dataset_path, "agent_obs.csv")
print("agent_pos数据的路径:\t", agent_obs_path)
agent_obs_data = try_read_csv_and_toList(agent_obs_path, isPrintInfo=True)
agent_obs_data = torch.tensor(agent_obs_data).to(device)
print("agent_obs_data_长度", len(agent_obs_data))
# print("agent_obs_dimension:", get_dimensions(agent_obs_data))
print("agent_obs_shape:", agent_obs_data.shape)
# print(agent_obs_data)
agent_obs_tmpList = []

# %%
"load img_obs"
img_obs_1_path = os.path.join(dataset_path, "img_obs_1")
print("img_obs_1数据的路径:\t", img_obs_1_path)
img_obs_2_path = os.path.join(dataset_path, "img_obs_2")
print("img_obs_2数据的路径:\t", img_obs_2_path)

# %%
"load action"
action_path = os.path.join(dataset_path, "action.csv")
print("action数据的路径:\t", action_path)
action_data = try_read_csv_and_toList(action_path, isPrintInfo=True)
action_data = torch.tensor(action_data).to(device)
print("action_data_长度", len(action_data))
# print("action_dimension:", get_dimensions(action_data))
print("action_shape:", action_data.shape)
# print(action_data)
action_tmpList = []

# %%
"load reward"
reward_path = os.path.join(dataset_path, "reward.csv")
print("reward数据的路径:\t", reward_path)
reward_data = try_read_csv_and_toList(reward_path, isPrintInfo=True)
reward_data = torch.tensor(reward_data).to(device)
print("reward_data_长度", len(reward_data))
# print("reward_dimension:", get_dimensions(reward_data))
print("reward_shape:", reward_data.shape)
# print(reward_data)
reward_tmpList = []

# %%
"load task_name"
task_name_path = os.path.join(dataset_path, "task_name.csv")
print("task_name数据的路径:\t", task_name_path)
task_name_data = try_read_csv_and_toList(task_name_path, isPrintInfo=True)
task_name_data_array = np.array(task_name_data).reshape(-1, 1)
# task_name_data = torch.tensor(task_name_data).to(device)
print("task_name_data_长度", len(task_name_data))
# print("task_name_dimension:", get_dimensions(task_name_data))
print("task_name_shape:", task_name_data_array.shape)
# print(task_name_data)
task_name_tmpList = []

# %%
def get_teach_action():
    return env.action

# %%
env
env.IS_IMAGE_RAND_POS = True
# env.IS_IMAGE_RAND_POS = False
time.sleep(0.5)
is_drag_mode=False
isPrintStepInfo=True

if __name__ == '__main__':
    while True:
        print("重置.......")
        episode_start_id = tmp_id+1
        agent_obs_tmpList = []
        img_obs_1_tmpList = []
        img_obs_2_tmpList = []
        task_name_tmpList = []
        action_tmpList = []
        reward_tmpList = []
        env.reset(isAddError=False, is_drag_mode=is_drag_mode)
        env.reset(isAddError=True, is_drag_mode=is_drag_mode, error_scale_pos=4, error_scale_ort=2)

        print("环境重置完成...\t状态观测为{}".format([f"{x:.4f}" for x in env.agent_obs]))
        if is_drag_mode:
            print("按 s 开始本回合,注意手动打开onRobot拖动示教模式\n启用后若提前完成,则按 x 终止示教回合")
            while True:
                env._get_obs(isPrintInfo=False)
                if keyboard.is_pressed("s"): break
        else:
            print("按 s 开始本回合")
            while True:
                env._get_obs(isPrintInfo=False)
                if keyboard.is_pressed("s"): break
        time.sleep(0.5) # 开始拖动示教后0.5秒再记录动作!为的是避免开始时停顿的动作被学习了
        while True:
            print("---------------------------------------------------------------------")
            env.obs = env._get_obs()
            agent_obs_tmpList.append(np.copy(env.obs['agent_obs']))
            img_obs_1_tmpList.append(np.copy(env.obs['img_obs_1']))
            img_obs_2_tmpList.append(np.copy(env.obs['img_obs_2']))
            task_name_tmpList.append(env.obs['task_name'])
            
            if is_drag_mode is False:
                env.step_by_keyboard(isPrintInfo=isPrintStepInfo)
                action_tmpList.append(np.copy(env.action))
                reward_tmpList.append(np.copy([env.reward]))
                tmp_id += 1
                if keyboard.is_pressed("x"):
                    env.done = True
                    break
                if env.done: break
                
            else:
                env.step_by_drag(isPrintInfo=isPrintStepInfo)
                print(env.action)
                action_tmpList.append(np.copy(env.action))
                print(action_tmpList)
                reward_tmpList.append(np.copy([env.reward]))
                tmp_id += 1
                if keyboard.is_pressed("x"):
                    env.done = True
                    break
                if env.done: break
                    
        if is_drag_mode is True:
            print("先关闭onRobot拖动示教模式!!! 不要再拖动,否则保护性停止 \n 然后按 e 结束本次示教")
            while True:
                if keyboard.is_pressed("e"): break
                    

        
        """数据集存储模块"""
        isSave = False
        print("是否保存本回合示教数据? y/n (是请按y; 否请按n)")
        while True:
            if keyboard.is_pressed("y"): 
                isSave = True; 
                break
            elif keyboard.is_pressed("n"): 
                isSave = False; 
                break
        if isSave is True:
            episode_end_id = tmp_id
            # print(episode_ends,'\n',torch.tensor([[episode_end_id+1]]).to(device))
            episode_ends = torch.cat((episode_ends, torch.tensor([[episode_end_id+1]]).to(device)), dim=0)
            
            # print(np.array(episode_ends))
            try_to_csv(episode_ends_path, pd.DataFrame(episode_ends.cpu().numpy().reshape(-1,1)), info="episode_ends")
            
            # print(agent_obs_tmpList)
            for row in agent_obs_tmpList:
                agent_obs_data = torch.cat((agent_obs_data, torch.tensor([row]).to(device)), dim=0)
            try_to_csv(agent_obs_path, pd.DataFrame(agent_obs_data.cpu().numpy().reshape(-1,env.agent_obs_dim)), info="agent_obs_data")
            
            #print(img_obs_tmpList)
            for i in range(0, episode_end_id-episode_start_id+1):
                ".zfill(12)表示用12位数存储索引"
                try_to_image(np.array(img_obs_1_tmpList[i]), os.path.join(img_obs_1_path, "img_obs_1_"+str(i+episode_start_id).zfill(12)+".png"), 
                                    height=env.img_obs_height[0], width=env.img_obs_width[0], channel=env.img_obs_channel, info="img_obs_1_data")
                
            #print(img_obs_tmpList)
            for i in range(0, episode_end_id-episode_start_id+1):
                ".zfill(12)表示用12位数存储索引"
                try_to_image(np.array(img_obs_2_tmpList[i]), os.path.join(img_obs_2_path, "img_obs_2_"+str(i+episode_start_id).zfill(12)+".png"), 
                                    height=env.img_obs_height[1], width=env.img_obs_width[1], channel=env.img_obs_channel, info="img_obs_2_data")
            
            # print(action_tmpList)
            for row in action_tmpList:
                action_data = torch.cat((action_data, torch.tensor([row]).to(device)), dim=0)
            try_to_csv(action_path, pd.DataFrame(action_data.cpu().numpy().reshape(-1,env.action_dim)), info="action_data")
            
            # print(reward_tmpList)
            for row in reward_tmpList:
                reward_data = torch.cat((reward_data, torch.tensor([row]).to(device)), dim=0)
            try_to_csv(reward_path, pd.DataFrame(reward_data.cpu().numpy().reshape(-1,1)), info="reward_data")
            
            # print(task_name_tmpList)
            task_name_tmpList_array = np.array(task_name_tmpList).reshape(-1,1)
            # 检查 task_name_data_array 是否为空
            if task_name_data_array.size == 0:
                task_name_data_array = task_name_tmpList_array
            else:
                task_name_data_array = np.concatenate((task_name_data_array, task_name_tmpList_array), axis=0)
            try_to_csv(task_name_path, pd.DataFrame(task_name_data_array.reshape(-1,1)), info="task_name_data")
            
        else:
            "id回调"
            tmp_id = episode_start_id - 1
            episode_end_id = tmp_id
            print("本回合示教数据不保存")
        
        print("---------------------------------------------------------------------")
        print("---------------------------------------------------------------------")
        
    print("示教数据采集结束")


