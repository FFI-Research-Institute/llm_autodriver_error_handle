import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 生成随机数据
num_points = 1000
x = np.random.rand(num_points)
y = np.random.rand(num_points)
z = np.random.rand(num_points)

# 创建图表
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置背景颜色
ax.set_facecolor('#e6f4d3')  # 设置背景色

# 绘制散点图
ax.scatter(x, y, z, c='darkgreen', marker='o', alpha=0.6)

# 设置标签
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 设置标题


fig.patch.set_facecolor('#e6f4d3')

# 显示图表
plt.tight_layout()
plt.show()
