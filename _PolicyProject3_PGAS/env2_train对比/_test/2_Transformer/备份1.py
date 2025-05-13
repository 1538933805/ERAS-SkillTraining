import pandas as pd

# Read the CSV file (ensure the file path is correct)
path = ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\2_Transformer\\"
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

# Count successful episodes
successful = sum(1 for ep in episodes if ep['step'].max() < 48)

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
    'avg_force': df_abs[['fx', 'fy', 'fz']].stack().mean(),
    'avg_torque': df_abs[['tx', 'ty', 'tz']].stack().mean(),
    'avg_total_steps_per_episode': len(df) / len(episodes)  # Total steps divided by number of episodes
}
print(len(episodes))
print(len(df))
# Write results to the summary text file
with open(path+'summary.txt', 'w') as f:
    f.write(f"Successful episodes: {successful}\n\n")
    
    f.write("Episode statistics:\n")
    for i, stats in enumerate(ep_stats, 1):
        f.write(f"Ep{i}: Steps={stats['steps']} "
                f"Average Depth={stats['avg_depth']:.4f} "
                f"Average Force (x|y|z)={stats['avg_fx']:.4f}|{stats['avg_fy']:.4f}|{stats['avg_fz']:.4f} "
                f"Average Torque (x|y|z)={stats['avg_tx']:.4f}|{stats['avg_ty']:.4f}|{stats['avg_tz']:.4f}\n")
    
    f.write("\nOverall statistics:\n")
    f.write(f"Average Steps: {overall['avg_step']:.2f}\n")
    f.write(f"Average Depth: {overall['avg_depth']:.4f}\n")
    f.write(f"Overall Average Force: {overall['avg_force']:.4f}\n")
    f.write(f"Overall Average Torque: {overall['avg_torque']:.4f}\n")
    f.write(f"Average Total Steps per Episode: {overall['avg_total_steps_per_episode']:.2f}\n")

print("Processing completed! Results have been saved to summary.txt")
