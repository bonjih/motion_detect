import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('df/data_log.csv')
plt.figure(figsize=(40, 15))

dfb = df[['motion', 'motion_sma']]

df_scaled_sma = (dfb['motion_sma'] - dfb['motion'].min()) / (dfb['motion'].max() - dfb['motion'].min())
df_scaled_motion = (dfb['motion'] - dfb['motion'].min()) / (dfb['motion'].max() - dfb['motion'].min())
df_scaled = pd.concat((df['ts'], df_scaled_motion, df_scaled_sma), axis=1)

print("Summary")
print(df.describe())
print(df_scaled.describe())
print(df_scaled.var())
plt.scatter(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], s=2, marker='+')
plt.scatter(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 2], s=3, marker='^')
plt.axhline(y=350000 / df['motion'].max(), color='orange', linestyle='--', linewidth=1)
plt.yticks(np.arange(0, 1, 0.01))
plt.xticks(np.arange(df_scaled.iloc[:, 0].min(), df_scaled.iloc[:, 0].max(), 10), rotation=90)

plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.savefig("threshold_plot.png")

# 0.07 -> 1.529747e7
# 0.08 -

# df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y/%m/%d %H:%M:%S.%f'))
