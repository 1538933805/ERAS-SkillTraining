import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======================== Global Parameters ========================
# Style
sns.set_style("darkgrid")
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
FONT_SIZE = 20
plt.rcParams['font.size'] = FONT_SIZE

# Plot settings
LINE_WIDTH = 1          # Global line width for all curves
MOVING_AVG_WINDOW = 100   # Window size for moving average
MAX_X_LENGTH = 49000      # Maximum data length to plot (truncate if longer)
X_LIM = (0, 500)        # x-axis limits (after downsampling if applied)
Y_LIM = (0.12, 0.6)          # y-axis limits
GRID_STYLE = {            # Grid style parameters
    'alpha': 1,
    'linestyle': '--',
    'linewidth': 2
}

base_path = ".\\_PolicyProject3_PGAS\\env2_train对比"
# Paths and names
paths_list = [
    [base_path+"\\2_Transformer\\loss.csv", 0, 'Transformer DP'],
    [base_path+"\\1_Attention\\loss.csv", 0, 'CNN DP'],
    [base_path+"\\3_VE_mobileViT1\\loss.csv", 1, 'MobileViT + U-Net'],
    [base_path+"\\6_VE_SigLIP2\\loss.csv", 0, 'SigLIP + U-Net'],
    [base_path+"\\7_DINOv2_224\\loss.csv", 1, 'DINOv2 + U-Net'],
    [base_path+"\\10_RDT_mobileViT\\loss.csv", 1, 'MobileViT + RDT'],
    [base_path+"\\9_RDT_SigLIP\\loss.csv", 1, 'SigLIP + RDT'],
    [base_path+"\\7_DINOv2_RDT\\loss.csv", 1, 'DINOv2 + RDT'],
]
paths = [item[0] for item in paths_list]
output_path = base_path+"\\_draw1\\4pics\\loss_comparison2.png"
names = [item[2] for item in paths_list]
colors = list(plt.cm.tab10.colors[:10])  # 转换为列表
colors[2], colors[3] = colors[3], colors[2]  # 交换位置
# colors = plt.cm.Set1.colors[:12]  # 使用SetX的前12种颜色
# colors = plt.cm.Accent.colors[:10]
# colors = plt.cm.Paired.colors[:10]
# colors = ['#00FFFF', '#0000FF', '#FF0000', '#800000', '#FF4500', '#008000', '#FFFF00', '#FFA500']

# ======================== Data Processing ========================
def moving_average(data, window_size):
    """Apply moving average smoothing to 1D data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def downsample(data, group_size=8):
    """Downsample data by averaging groups of `group_size` points."""
    n = len(data) // group_size
    return data[:n*group_size].reshape(-1, group_size).mean(axis=1)

# ======================== Plotting ========================
plt.figure(figsize=(9, 6), dpi=200)

for i, (path, name, color) in enumerate(zip(paths, names, colors)):
    try:
        # Read data (assuming 3rd column is loss)
        loss_data = pd.read_csv(path, header=None).iloc[:, 2].values
        
        # Special downsampling for MobileViT (i=3)
        if paths_list[i][1] == 1:
            processed_loss = downsample(loss_data)
        else:
            processed_loss = loss_data
        
        # Truncate if longer than MAX_X_LENGTH
        if len(processed_loss) > MAX_X_LENGTH:
            processed_loss = processed_loss[:MAX_X_LENGTH]
        
        # Apply moving average to ALL curves
        processed_loss = moving_average(processed_loss, MOVING_AVG_WINDOW)
        
        # Create x-axis values (adjusted for moving average)
        x = np.arange(len(processed_loss))
        
        # Plot with global line width
        if i == 2: order = 10
        else: order = len(paths_list)-i
        plt.plot(x, processed_loss, label=name, color=color, linewidth=LINE_WIDTH,
                 zorder=order  # 先画的曲线zorder更大，显示在上层
        )
    except Exception as e:
        print(f"{name}: 绘图失败")

# ======================== Plot Formatting ========================
plt.xlabel('Batch Number')
plt.ylabel('Loss')
legend = plt.legend(fontsize=int(FONT_SIZE*0.85))
legend.set_zorder(100)
plt.grid(**GRID_STYLE)
plt.xlim(X_LIM)
plt.ylim(Y_LIM)

# Save and show
# plt.tight_layout()
plt.savefig(output_path)
plt.show()