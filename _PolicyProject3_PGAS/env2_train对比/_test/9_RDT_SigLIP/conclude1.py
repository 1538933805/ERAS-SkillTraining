import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file (ensure the file path is correct)
path = ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\9_RDT_SigLIP\\"
df = pd.read_csv(path+'info.csv', header=None, 
                 names=['step', 'depth', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])

# Split into different episodes
episodes = []
current_episode = []
prev_step = 0

for _, row in df.iterrows():
    if row['step'] <= prev_step and len(current_episode) > 0:
        episodes.append(pd.DataFrame(current_episode))
        current_episode = []
    current_episode.append(row)
    prev_step = row['step']
if current_episode:
    episodes.append(pd.DataFrame(current_episode))

# Count successful episodes and success rate
successful = sum(1 for ep in episodes if ep['step'].max() < 48)
success_rate = successful / len(episodes) * 100  # Calculate success rate percentage

# Episode statistics
ep_stats = []
for ep in episodes:
    ep_abs = ep.abs()
    stats = {
        'steps': len(ep),
        'avg_step': ep['step'].mean(),
        'avg_depth': ep['depth'].mean(),
        'avg_fx': ep_abs['fx'].mean(),
        'avg_fy': ep_abs['fy'].mean(),
        'avg_fz': ep_abs['fz'].mean(),
        'avg_tx': ep_abs['tx'].mean(),
        'avg_ty': ep_abs['ty'].mean(),
        'avg_tz': ep_abs['tz'].mean()
    }
    ep_stats.append(stats)

# Overall statistics
df_abs = df.abs()
overall = {
    'avg_step': df['step'].mean(),
    'avg_depth': df['depth'].mean(),
    'avg_fx': df_abs['fx'].mean(),
    'avg_fy': df_abs['fy'].mean(),
    'avg_fz': df_abs['fz'].mean(),
    'avg_tx': df_abs['tx'].mean(),
    'avg_ty': df_abs['ty'].mean(),
    'avg_tz': df_abs['tz'].mean(),
    'avg_total_steps_per_episode': len(df) / len(episodes)  # Total steps divided by number of episodes
}

# Write results to the summary text file
with open(path+'summary.txt', 'w') as f:
    f.write(f"Successful episodes: {successful}\n")
    f.write(f"Success rate: {success_rate:.2f}%\n\n")
    
    f.write("Episode statistics:\n")
    for i, stats in enumerate(ep_stats, 1):
        f.write(f"Ep{i}: Steps={stats['steps']} "
                f"Average Depth={stats['avg_depth']:.4f} "
                f"Average Force (x|y|z)={stats['avg_fx']:.4f}|{stats['avg_fy']:.4f}|{stats['avg_fz']:.4f} "
                f"Average Torque (x|y|z)={stats['avg_tx']:.4f}|{stats['avg_ty']:.4f}|{stats['avg_tz']:.4f}\n")
    
    f.write("\nOverall statistics:\n")
    f.write(f"Average Steps: {overall['avg_step']:.2f}\n")
    f.write(f"Average Depth: {overall['avg_depth']:.4f}\n")
    f.write(f"Average Force (x|y|z): {overall['avg_fx']:.4f}|{overall['avg_fy']:.4f}|{overall['avg_fz']:.4f}\n")
    f.write(f"Average Torque (x|y|z): {overall['avg_tx']:.4f}|{overall['avg_ty']:.4f}|{overall['avg_tz']:.4f}\n")
    f.write(f"Average Total Steps per Episode: {overall['avg_total_steps_per_episode']:.2f}\n")

print("Processing completed! Results have been saved to summary.txt")

# Plotting: Force and Torque boxplot/violin plot
force_data = df_abs[['fx', 'fy', 'fz']]
torque_data = df_abs[['tx', 'ty', 'tz']]

# Combine force and torque data for plotting
data = pd.concat([force_data, torque_data], axis=1)
data.columns = ['Force_x', 'Force_y', 'Force_z', 'Torque_x', 'Torque_y', 'Torque_z']

# 画图
# Create a violin plot
plt.figure(figsize=(10, 6), dpi=200)
sns.violinplot(data=data)
plt.title('Force and Torque Distribution (Absolute Values)')
plt.ylabel('Force/Torque Magnitude')
plt.xlabel('Axes')
# Save the plot as PNG
plt.savefig(path+'force_torque_distribution_violin.png')
plt.show()

# Create a boxplot
plt.figure(figsize=(10, 6), dpi=200)
sns.boxplot(data=data)
plt.title('Force and Torque Distribution (Absolute Values)')
plt.ylabel('Force/Torque Magnitude')
plt.xlabel('Axes')
# Save the plot as PNG
plt.savefig(path+'force_torque_distribution_boxplot.png')
plt.show()