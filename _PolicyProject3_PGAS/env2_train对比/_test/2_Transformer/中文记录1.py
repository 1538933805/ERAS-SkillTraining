import pandas as pd

# 读取CSV文件（请确保文件路径正确）
path = ".\\_PolicyProject3_PGAS\\env2_train对比\\_test\\2_Transformer\\"
df = pd.read_csv(path+'info.csv', header=None, 
                 names=['step', 'depth', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])

# 分割为不同episode
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

# 统计成功回合
successful = sum(1 for ep in episodes if ep['step'].max() < 48)

# 各回合统计
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

# 总体统计
df_abs = df.abs()
overall = {
    'avg_step': df['step'].mean(),
    'avg_depth': df['depth'].mean(),
    'avg_force': df_abs[['fx', 'fy', 'fz']].stack().mean(),
    'avg_torque': df_abs[['tx', 'ty', 'tz']].stack().mean()
}

# 写入结果文件
with open(path+'summary.txt', 'w') as f:
    f.write(f"成功回合数: {successful}\n\n")
    
    f.write("各回合统计:\n")
    for i, stats in enumerate(ep_stats, 1):
        f.write(f"Ep{i}: 步数={stats['steps']} "
                f"平均深度={stats['avg_depth']:.4f} "
                f"平均力(x|y|z)={stats['avg_fx']:.4f}|{stats['avg_fy']:.4f}|{stats['avg_fz']:.4f} "
                f"平均力矩(x|y|z)={stats['avg_tx']:.4f}|{stats['avg_ty']:.4f}|{stats['avg_tz']:.4f}\n")
    
    f.write("\n总体统计:\n")
    f.write(f"平均步数: {overall['avg_step']:.2f}\n")
    f.write(f"平均深度: {overall['avg_depth']:.4f}\n")
    f.write(f"总体平均力: {overall['avg_force']:.4f}\n")
    f.write(f"总体平均力矩: {overall['avg_torque']:.4f}")

print("处理完成！结果已保存至 summary.txt")