import torch
import torch.nn as nn
import DiffusionPolicy_Networks as nets
output_dir = (".\\_PolicyProject3_PGAS\\_trained_models_env3\\模型结构")

# 构建 ResNet18 编码器
vision_encoder_1 = nets.get_resnet_with_attention('resnet18')
vision_encoder_2 = nets.get_resnet_with_attention('resnet18')

# 替换所有 BatchNorm 为 GroupNorm
vision_encoder_1 = nets.replace_bn_with_gn(vision_encoder_1)
vision_encoder_2 = nets.replace_bn_with_gn(vision_encoder_2)

# ResNet18 输出维度为 512
vision_feature_dim = 512
obs_dim = vision_feature_dim + 0

img_height = [256,256]
img_width = [256,256]
img_channel = 3

agent_obs_dim = 12
action_dim = 7

pred_horizon = 8
obs_horizon = 1
action_horizon = 4

# 创建网络对象
noise_pred_net = nets.ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * 2 * obs_horizon
)

# 将所有网络组件组合成一个 ModuleDict
nets = nn.ModuleDict({
    'vision_encoder_1': vision_encoder_1,
    'vision_encoder_2': vision_encoder_2,
    'noise_pred_net': noise_pred_net
})

# 打印参数数量
print("number of all parameters: {:e}".format(
    sum(p.numel() for p in nets['vision_encoder_1'].parameters()) +
    sum(p.numel() for p in nets['vision_encoder_2'].parameters()) +
    sum(p.numel() for p in nets['noise_pred_net'].parameters())
))

# 示例输入
image_1 = torch.zeros((1, obs_horizon, 3, img_height[0], img_width[0]))
image_2 = torch.zeros((1, obs_horizon, 3, img_height[1], img_width[1]))
noised_action = torch.randn((1, pred_horizon, action_dim))
diffusion_iter = torch.zeros((1,))

# 视觉编码器前向传播
with torch.no_grad():
    image_features_1 = nets['vision_encoder_1'](image_1.flatten(end_dim=1))
    image_features_1 = image_features_1.reshape(*image_1.shape[:2], -1)
    image_features_2 = nets['vision_encoder_2'](image_2.flatten(end_dim=1))
    image_features_2 = image_features_2.reshape(*image_2.shape[:2], -1)
    obs = torch.cat([image_features_1, image_features_2], dim=-1)

# 噪声预测网络前向传播
with torch.no_grad():
    noise = nets['noise_pred_net'](
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1)
    )

# 导出视觉编码器1为 ONNX 格式
torch.onnx.export(
    nets['vision_encoder_1'],
    image_1.flatten(end_dim=1),
    output_dir+"\\vision_encoder_1.onnx",
    opset_version=11,
    input_names=['input_image_1'],
    output_names=['output_features_1']
)

# 导出视觉编码器2为 ONNX 格式
torch.onnx.export(
    nets['vision_encoder_2'],
    image_2.flatten(end_dim=1),
    output_dir+"\\vision_encoder_2.onnx",
    opset_version=11,
    input_names=['input_image_2'],
    output_names=['output_features_2']
)

# 导出噪声预测网络为 ONNX 格式
torch.onnx.export(
    nets['noise_pred_net'],
    (noised_action, diffusion_iter, obs.flatten(start_dim=1)),
    output_dir+"\\noise_pred_net.onnx",
    opset_version=13,
    input_names=['noised_action', 'diffusion_iter', 'global_cond'],
    output_names=['noise']
)