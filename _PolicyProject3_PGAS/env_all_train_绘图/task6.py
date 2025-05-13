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
MOVING_AVG_WINDOW = 1   # Window size for moving average
MAX_X_LENGTH = 49000      # Maximum data length to plot (truncate if longer)
X_LIM = (0, 10000)        # x-axis limits (after downsampling if applied)
Y_LIM = (0, 1)          # y-axis limits
GRID_STYLE = {            # Grid style parameters
    'alpha': 1,
    'linestyle': '--',
    'linewidth': 2
}

# Paths and names
paths_list = [
    [".\\_PolicyProject2_public\\_trained_models_env6\\training_loss_Attention.csv", 0],
]
paths = [item[0] for item in paths_list]
output_path = ".\\_PolicyProject3_PGAS\\env_all_train_绘图\\figs\\task6.png"
names = ['Noise action pred MSE']
colors = plt.cm.tab10.colors[:10]  # Using first 4 colors from tab10 palette

# ======================== Data Processing ========================
def moving_average(data, window_size):
    """Apply moving average smoothing to 1D data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def downsample(data, group_size=8):
    """Downsample data by averaging groups of `group_size` points."""
    n = len(data) // group_size
    return data[:n*group_size].reshape(-1, group_size).mean(axis=1)

# ======================== Plotting ========================
plt.figure(figsize=(6, 5), dpi=200)

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
        plt.plot(x, processed_loss, label=name, color=color, linewidth=LINE_WIDTH)
    except Exception as e:
        print(f"{name}: 绘图失败")

# ======================== Plot Formatting ========================
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.legend(fontsize=int(FONT_SIZE*0.85))
plt.grid(**GRID_STYLE)
plt.xlim(X_LIM)
plt.ylim(Y_LIM)

# Save and show
plt.tight_layout()
plt.savefig(output_path)
plt.show()