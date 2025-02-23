import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 创建圆环（Torus）
torus = trimesh.creation.torus(major_radius=2.0, minor_radius=0.5)

# 获取网格顶点和面
vertices = torus.vertices
faces = torus.faces

# 对顶点进行正弦扭曲变形
# 添加一个与 z 相关的正弦扰动
vertices[:, 0] += 0.3 * np.sin(2 * np.pi * vertices[:, 2])
vertices[:, 1] += 0.3 * np.cos(2 * np.pi * vertices[:, 2])

# 绘制扭曲后的圆环
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用 Matplotlib 三角化绘制网格
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, color='cyan', edgecolor='k', alpha=0.8)

# 添加坐标轴
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

# 设置标题
ax.set_title("Twisted Torus with Sine Distortion")

# 显示图形
plt.show()
