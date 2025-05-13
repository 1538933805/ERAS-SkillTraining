import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")  # 白色背景+灰色网格 ("whitegrid", "darkgrid", "white", "ticks" 可选)
# sns.set_palette("deep")   # 使用柔和的颜色 ("pastel", "deep", "muted", "bright", "dark" 也可选)

# Set font style and size
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
# 在画图前添加Seaborn全局样式设置（只需加一次，对所有图生效）

# 使用matplotlib的tab10调色板生成6个颜色
colors = plt.cm.tab10.colors[:6]  # 获取tab10调色板的前6种颜色

# Paths for the 4 directories (update these paths as needed)
paths = [
    ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\1_Attention\\",
    ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\2_Transformer\\",
    ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\6_VE_SigLIP2\\",
    ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\1_Attention\\",
    ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\2_Transformer\\",
    ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\6_VE_SigLIP2\\"
]
names = ['MHAR + U-Net', 'MHAR + Transformer', 'SigLIP + U-Net', 
         'mobileViT + U-Net', 'DINOv2 + U-Net', 'Baseline']
path1 = (".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\_draw1\\")


# 计算所有数据的最大值和最小值
all_data = []

for path in paths:
    df = pd.read_csv(path+'info.csv', header=None, names=['step', 'depth', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
    df_abs = df.abs()
    force_data = df_abs[['fx', 'fy', 'fz']]
    torque_data = df_abs[['tx', 'ty', 'tz']]
    data = pd.concat([force_data, torque_data], axis=1)
    all_data.append(data)

# 计算所有数据的最大值和最小值
max_val = pd.concat(all_data).max().max()
min_val = pd.concat(all_data).min().min()
"手动设置"
max_val = 40
min_val = 0





"画1"
# Set up the subplots (1 row, 4 columns)
fig, axs = plt.subplots(2, 3, figsize=(24, 16), dpi=200)

# Loop over the paths
for i, path in enumerate(paths):
    # Read the CSV file
    df = pd.read_csv(path+'info.csv', header=None, 
                     names=['step', 'depth', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])

    # Absolute values for force and torque
    df_abs = df.abs()
    force_data = df_abs[['fx', 'fy', 'fz']]
    torque_data = df_abs[['tx', 'ty', 'tz']]

    # Combine force and torque data for plotting
    data = pd.concat([force_data, torque_data], axis=1)
    data.columns = ['F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']

    # Plot Violin plot
    sns.violinplot(data=data, ax=axs[i//3,i%3], inner='quart',
                   linewidth=1.5, saturation=0.8)
    axs[i//3,i%3].set_title(names[i])
    axs[i//3,i%3].set_ylabel('F/T Magnitude')
    axs[i//3,i%3].set_xlabel('Axes')
    axs[i//3,i%3].set_ylim(min_val, max_val)  # 设置统一的纵坐标范围
    axs[i//3,i%3].grid(True, linestyle='--', alpha=1, linewidth=2, axis='y')  # 添加虚线网格

# Adjust the layout to avoid overlapping
plt.tight_layout()
# Save the violin plot as PNG
plt.savefig(path1+'force_torque_distribution_violin_grid.png')
plt.show()





"画2"
# Set up the subplots (1 row, 4 columns) for boxplots
fig, axs = plt.subplots(2, 3, figsize=(24, 16), dpi=200)

# Loop again for boxplots
for i, path in enumerate(paths):
    # Read the CSV file
    df = pd.read_csv(path+'info.csv', header=None, 
                     names=['step', 'depth', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])

    # Absolute values for force and torque
    df_abs = df.abs()
    force_data = df_abs[['fx', 'fy', 'fz']]
    torque_data = df_abs[['tx', 'ty', 'tz']]

    # Combine force and torque data for plotting
    data = pd.concat([force_data, torque_data], axis=1)
    data.columns = ['F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']

    # Plot Box plot
    sns.boxplot(data=data, ax=axs[i//3,i%3],
                linewidth=1.5, saturation=0.8)
    axs[i//3,i%3].set_title(names[i])
    axs[i//3,i%3].set_ylabel('F/T Magnitude')
    axs[i//3,i%3].set_xlabel('Axes')
    axs[i//3,i%3].set_ylim(min_val, max_val)  # 设置统一的纵坐标范围
    axs[i//3,i%3].grid(True, linestyle='--', alpha=1, linewidth=2, axis='y')  # 添加虚线网格

# Adjust the layout to avoid overlapping
plt.tight_layout()
# Save the box plot as PNG
plt.savefig(path1+'force_torque_distribution_boxplot_grid.png')
plt.show()

