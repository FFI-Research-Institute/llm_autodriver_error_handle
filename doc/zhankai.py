import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. 生成瑞士卷数据（内在2D + 嵌入3D）
# -----------------------------
num_points = 1000
theta = np.linspace(0, 4 * np.pi, num_points)  # 螺旋角 (内在维度1)
z = np.linspace(-2, 2, num_points)            # 高度  (内在维度2)
r = 5                                         # 固定半径

# 生成三维坐标
x = r * np.sin(theta)
y = r * np.cos(theta)

# 加入较小的噪声
noise_x = np.random.normal(0, 0.1, num_points)
noise_y = np.random.normal(0, 0.1, num_points)
noise_z = np.random.normal(0, 0.1, num_points)

x += noise_x
y += noise_y
z += noise_z

# -------------------------------------------------
# 2. 在 2D 平面上展示 (theta, z) 的“展开”效果
# -------------------------------------------------
plt.figure(figsize=(13, 8), facecolor='#e6f4d3')  # 设置背景色
ax = plt.gca()  # 获取当前坐标轴
ax.set_facecolor('#e6f4d3')  # 设置背景色
sc_2d = plt.scatter(theta, z, c=theta, cmap=plt.cm.viridis, 
                    edgecolor='k', s=10, alpha=0.8)
plt.xlabel("Theta")
plt.ylabel("Z")
plt.colorbar(sc_2d, label='Theta')
plt.tight_layout()
plt.show()

# 添加 colorbar
plt.colorbar(sc_3d, ax=ax, shrink=0.5, aspect=10, label='Theta')
plt.show()
