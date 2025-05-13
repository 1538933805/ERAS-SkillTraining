#@title ### 网络性能测试脚本
import DiffusionPolicy_Networks
# module import
from random import uniform,choice
import numpy as np
import time
import os
import subprocess
import sys
import pandas as pd
import cv2
import zipfile
import torch
from torch import nn
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
img_height = [256,256]
img_width = [256,256]
img_channel = 3
agent_obs_dim = 12
action_dim = 7
pred_horizon = 8
obs_horizon = 1
action_horizon = 4
"记录标记!"
dataset_path =           (".\\_PolicyProject3_PGAS\\_trained_models_env1\\_teach1_dataset_amplify1")
MinMax_dir_path =        (".\\_PolicyProject3_PGAS\\_trained_models_env1\\MinMax")

ckpt_path =              (".\\_PolicyProject3_PGAS\\env1_train对比\\1_Attention\\model.ckpt")
output_dir =             (".\\_PolicyProject3_PGAS\\env1_train对比\\1_Attention")
training_loss_csv_path = (".\\_PolicyProject3_PGAS\\env1_train对比\\1_Attention\\loss.csv")
batch_reduce = 8 # GPU不足时,batch_size缩小倍数
max_iterations = 50000 * batch_reduce  # 目标最大迭代次数
"记录标记!"
"记录标记!"
# 测试参数配置
test_result_path = "./_PolicyProject3_PGAS/env1_train对比/_网络性能/1_Attention/network_performance.txt"  # 测试结果保存路径
batch_size = 2                   # 测试用batch size
warmup_iters = 5                 # 预热迭代次数
test_iters = 100                 # 正式测试迭代次数
repeat_times = 5                 # 重复测试次数取平均
"记录标记!"


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats, eps=1e-8):
    range_ = stats['max'] - stats['min'] # 归一化到 [0,1]
    range_[range_ < eps] = 1 # 当 range_ 小于 eps 时，视为常数 0
    ndata = (data - stats['min']) / range_ 
    ndata = ndata * 2 - 1 # 归一化到 [-1, 1]
    return ndata

def unnormalize_data(ndata, stats, eps=1e-8):
    ndata = (ndata + 1) / 2 # 反归一化回原始范围
    range_ = stats['max'] - stats['min']
    data = ndata * range_ + stats['min']
    return data

def normalize_data_byValue(data, max, min, eps=1e-8):
    range_ = max - min # 归一化到 [0,1]
    range_[range_ < eps] = 1 # 当 range_ 小于 eps 时，视为常数 0
    ndata = (data - min) / range_ 
    ndata = ndata * 2 - 1 # 归一化到 [-1, 1]
    return ndata

def unnormalize_data_byValue(ndata, max, min, eps=1e-8):
    ndata = (ndata + 1) / 2 # 反归一化回原始范围
    range_ = max - min
    data = ndata * range_ + min
    return data
# dataset
class PolicyProject1Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        
        self.dataset_path = dataset_path
        
        """
        # Read image files
        images_paths = sorted(os.listdir(os.path.join(self.dataset_path, "img_obs")))
        train_image_data = []
        for img_name in images_paths:
            img_path = os.path.join(self.dataset_path, "img_obs", img_name)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            img = img.transpose((2, 0, 1))  # Change the channel dimension to be the first one
            train_image_data.append(img)
        train_image_data = np.stack(train_image_data)
        train_image_data = train_image_data.astype(np.float32)
        print("train_image_data的形状及性质为:", train_image_data.shape, train_image_data.dtype)
        """
        # Store image paths  
        self.image_paths_1 = sorted(os.listdir(os.path.join(self.dataset_path, "img_obs_1")))
        self.image_paths_2 = sorted(os.listdir(os.path.join(self.dataset_path, "img_obs_2")))
        
        
        # Read CSV files
        agent_obs_csv = pd.read_csv(os.path.join(self.dataset_path, "agent_obs.csv"), header=None)
        action_csv = pd.read_csv(os.path.join(self.dataset_path, "action.csv"), header=None)
        episode_ends_csv = pd.read_csv(os.path.join(self.dataset_path, "episode_ends.csv"), header=None)

        # Remove the first row with indices
        agent_obs_csv = agent_obs_csv.iloc[0:]
        action_csv = action_csv.iloc[0:]
        episode_ends_csv = episode_ends_csv.iloc[0:]

        # Convert DataFrame to NumPy arrays and reshape them
        agent_obs = agent_obs_csv.to_numpy().astype(np.float32).reshape(-1, agent_obs_dim)
        action = action_csv.to_numpy().astype(np.float32).reshape(-1, action_dim)
        episode_ends = episode_ends_csv.to_numpy().astype(int).reshape(-1) #episode_ends需要是个1维数组
        print("episode_ends的形状及性质为:", episode_ends.shape, episode_ends.dtype)
        
        train_data = {
            'agent_obs': agent_obs,
            # (N, 6)
            'action': action
            # (N, 6)
        }
        print("agent_obs的形状及性质为:", agent_obs.shape, agent_obs.dtype)
        print("action的形状及性质为:", action.shape, action.dtype)
        
        """重写读取：结束"""

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
            print(key, "的最小:\n", stats[key]['min'], "\n的最大:\n", stats[key]['max'])
            
        # images are already normalized
        """
        normalized_train_data['image'] = train_image_data
        """

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        """
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        """
        
        """新增模块: 图像实时采样读取,而非预先加载所有的把内存爆了"""
        # Load images on-the-fly  
        nsample['image_1'] = np.zeros((self.obs_horizon, 3, img_height[0], img_width[0]), dtype=np.float32)  
        for i in range(self.obs_horizon):
            # print([buffer_start_idx, buffer_end_idx], [sample_start_idx, sample_end_idx])
            idx = max(min(i, sample_end_idx-1), sample_start_idx) - sample_start_idx
            img_path = os.path.join(self.dataset_path, "img_obs_1", self.image_paths_1[buffer_start_idx + idx])  
            img = Image.open(img_path).convert('RGB')  
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]  
            img = img.transpose((2, 0, 1))  # Change channel dimension to be first  
            nsample['image_1'][i] = img
        
        nsample['image_2'] = np.zeros((self.obs_horizon, 3, img_height[1], img_width[1]), dtype=np.float32)  
        for i in range(self.obs_horizon):
            # print([buffer_start_idx, buffer_end_idx], [sample_start_idx, sample_end_idx])
            idx = max(min(i, sample_end_idx-1), sample_start_idx) - sample_start_idx
            img_path = os.path.join(self.dataset_path, "img_obs_2", self.image_paths_2[buffer_start_idx + idx])  
            img = Image.open(img_path).convert('RGB')  
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]  
            img = img.transpose((2, 0, 1))  # Change channel dimension to be first  
            nsample['image_2'][i] = img
        
        nsample['agent_obs'] = nsample['agent_obs'][:self.obs_horizon,:]
        
        # print(nsample['image'].shape[0], nsample['action'].shape[0])
        # flag11 = np.array_equal(nsample['image'][0],nsample['image'][1])
        # flag12 = np.array_equal(nsample['agent_obs'][0],nsample['agent_obs'][1])
        # flag21 = np.array_equal(nsample['image'][-1],nsample['image'][-2])
        # flag22 = np.array_equal(nsample['agent_obs'][-1],nsample['agent_obs'][-2])
        # print("nsample测试1:\n", flag11, flag12)
        # print("nsample测试2:\n", flag21, flag22)
        # print("nsample测试3:\t", flag11 == flag12)
        # print("nsample测试4:\t", flag21 == flag22)
           
        return nsample










# 复用原始模型定义
import DiffusionPolicy_Networks as nets
# import DiffusionPolicy_Networks_VE1 as nets_VE1

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试

# 初始化模型 --------------------------------------------------
def create_network():
    # 视觉编码器
    vision_encoder_1 = nets.get_resnet_with_attention('resnet18')
    vision_encoder_2 = nets.get_resnet_with_attention('resnet18')
    
    # 替换BN为GN
    vision_encoder_1 = nets.replace_bn_with_gn(vision_encoder_1)
    vision_encoder_2 = nets.replace_bn_with_gn(vision_encoder_2)
    
    # 噪声预测网络
    obs_dim = 512 * 2 * obs_horizon  # 来自原始代码设置
    noise_pred_net = nets.ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    )
    
    model = nn.ModuleDict({
        'vision_encoder_1': vision_encoder_1,
        'vision_encoder_2': vision_encoder_2,
        'noise_pred_net': noise_pred_net
    }).to(device)
    
    # 启用训练模式以计算梯度
    model.train()
    return model

# 参数统计函数 --------------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# 内存计算函数 --------------------------------------------------
def model_memory_footprint(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size

# 显存测试函数（修正版）--------------------------------------------------
def profile_gpu_memory(model, dataloader):
    # 确保模型在训练模式
    model.train()
    
    # 准备数据
    batch = next(iter(dataloader))
    nimage_1 = batch['image_1'][:,:obs_horizon].to(device)
    nimage_2 = batch['image_2'][:,:obs_horizon].to(device)
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建需要梯度的噪声张量
    noise = torch.randn_like(batch['action'].to(device), requires_grad=True)
    timesteps = torch.zeros((batch_size,), device=device).long()
    
    # 前向传播
    # 视觉编码
    feat1 = model['vision_encoder_1'](nimage_1.flatten(end_dim=1))
    feat1 = feat1.reshape(*nimage_1.shape[:2], -1)
    feat2 = model['vision_encoder_2'](nimage_2.flatten(end_dim=1))
    feat2 = feat2.reshape(*nimage_2.shape[:2], -1)
    obs_cond = torch.cat([feat1, feat2], dim=-1).flatten(start_dim=1)
        
    # 噪声预测
    pred = model['noise_pred_net'](noise, timesteps, global_cond=obs_cond)
    
    # 创建需要梯度的目标张量
    target = torch.randn_like(pred, requires_grad=False)
    
    # 计算损失并反向传播
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    
    # 获取峰值显存
    peak_mem = torch.cuda.max_memory_allocated()
    
    # 清理
    del noise, pred, target, loss
    torch.cuda.empty_cache()
    
    return peak_mem

# 耗时测试函数（改进版）--------------------------------------------------
def profile_inference_time(model, dataloader, num_iters):
    # model.eval()
    model.train()
    batch = next(iter(dataloader))
    
    # 准备数据
    nimage_1 = batch['image_1'][:,:obs_horizon].to(device)
    nimage_2 = batch['image_2'][:,:obs_horizon].to(device)
    naction = batch['action'].to(device)
    
    # 预热
    print("正在进行预热...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model['vision_encoder_1'](nimage_1.flatten(end_dim=1))
            _ = model['vision_encoder_2'](nimage_2.flatten(end_dim=1))
    
    # 正式测试
    print("开始正式测试...")
    timings = []
    for _ in tqdm(range(num_iters), desc='性能测试进度'):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # 视觉编码
            feat1 = model['vision_encoder_1'](nimage_1.flatten(end_dim=1))
            feat1 = feat1.reshape(*nimage_1.shape[:2], -1)
            feat2 = model['vision_encoder_2'](nimage_2.flatten(end_dim=1))
            feat2 = feat2.reshape(*nimage_2.shape[:2], -1)
            
            # 条件拼接
            obs_cond = torch.cat([feat1, feat2], dim=-1).flatten(start_dim=1)
            
            # 噪声预测
            _ = model['noise_pred_net'](
                naction, 
                torch.zeros((batch_size,), device=device).long(), 
                global_cond=obs_cond
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    return np.mean(timings)*1000, np.std(timings)*1000  # 转换为毫秒

# 耗时训练函数（改进版）--------------------------------------------------
def profile_train_time(model, dataloader, num_iters):
    # model.eval()
    model.train()
    batch = next(iter(dataloader))
    
    # 准备数据
    nimage_1 = batch['image_1'][:,:obs_horizon].to(device)
    nimage_2 = batch['image_2'][:,:obs_horizon].to(device)
    naction = batch['action'].to(device)
    
    # 预热
    print("正在进行预热...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model['vision_encoder_1'](nimage_1.flatten(end_dim=1))
            _ = model['vision_encoder_2'](nimage_2.flatten(end_dim=1))
    
    # 正式测试
    print("开始正式训练...")
    timings = []
    for _ in tqdm(range(num_iters), desc='性能测试进度'):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # 视觉编码
        feat1 = model['vision_encoder_1'](nimage_1.flatten(end_dim=1))
        feat1 = feat1.reshape(*nimage_1.shape[:2], -1)
        feat2 = model['vision_encoder_2'](nimage_2.flatten(end_dim=1))
        feat2 = feat2.reshape(*nimage_2.shape[:2], -1)
        
        # 条件拼接
        obs_cond = torch.cat([feat1, feat2], dim=-1).flatten(start_dim=1)
        
        # 噪声预测
        pred = model['noise_pred_net'](
            naction, 
            torch.zeros((batch_size,), device=device).long(), 
            global_cond=obs_cond)
        # 创建需要梯度的目标张量
        target = torch.randn_like(pred, requires_grad=False)
        
        # 计算损失并反向传播
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    return np.mean(timings)*1000, np.std(timings)*1000  # 转换为毫秒

# 主测试流程 --------------------------------------------------
if __name__ == "__main__":
    print("=== 网络性能测试开始 ===")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请在GPU环境下运行测试")
    
    # 初始化模型
    print("初始化模型...")
    model = create_network()
    
    # 参数统计
    print("计算模型参数...")
    total_params, trainable_params = count_parameters(model)
    memory_footprint = model_memory_footprint(model)
    
    # 准备测试数据
    print("准备测试数据...")
    pred_horizon = 8
    obs_horizon = 1
    action_horizon = 4
    
    dataset = PolicyProject1Dataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    # 显存测试
    print("测试显存需求...")
    peak_mem = profile_gpu_memory(model, dataloader)
    
    # 耗时测试（多次取平均）
    print("测试推理时间...")
    time_results = []
    std_results = []
    time_results_train = []
    std_results_train = []
    for i in range(repeat_times):
        print(f"\n第 {i+1}/{repeat_times} 次测试...")
        avg_time, std_time = profile_inference_time(model, dataloader, test_iters)
        avg_time_train, std_time_train = profile_train_time(model, dataloader, test_iters)
        time_results.append(avg_time)
        std_results.append(std_time)
        time_results_train.append(avg_time_train)
        std_results_train.append(std_time_train)
    avg_inference_time = np.mean(time_results)
    avg_std = np.mean(std_results)
    avg_inference_time_train = np.mean(time_results_train)
    avg_std_train = np.mean(std_results_train)
    
    # 结果收集
    results = f"""=== 网络性能测试报告 ===

1. 参数统计:
   - 总参数数量: {total_params:,}
   - 可训练参数: {trainable_params:,}
   - 模型内存占用: {memory_footprint/1024**2:.2f} MB

2. GPU需求:
   - Batch Size = {batch_size} 时峰值显存: {peak_mem/1024**3:.2f} GB

3. 推理性能:
   - Batch Size = {batch_size} 时平均推理时间: {avg_inference_time:.2f} ± {avg_std:.2f} ms
   - 测试次数: {repeat_times} 次, 每次 {test_iters} 迭代
   训练性能:
   - Batch Size = {batch_size} 时平均训练时间: {avg_inference_time_train:.2f} ± {avg_std_train:.2f} ms
   - 测试次数: {repeat_times} 次, 每次 {test_iters} 迭代

4. 环境信息:
   - 设备名称: {torch.cuda.get_device_name(device)}
   - CUDA版本: {torch.version.cuda}
   - PyTorch版本: {torch.__version__}
   - 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

=== 测试结束 ===
"""
    
    # 保存结果
    with open(test_result_path, "w", encoding='utf-8') as f:
        f.write(results)
    
    # 打印结果
    print("\n" + "="*50)
    print(results)
    print(f"测试结果已保存至: {os.path.abspath(test_result_path)}")
    print("="*50)