import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Generate a regular Swiss Roll point cloud
num_points = 1000
theta = np.linspace(0, 4 * np.pi, num_points)  # Spiral angle
z = np.linspace(-2, 2, num_points)             # Z-axis range
r = 5                                          # Fixed radius

# Generate Swiss Roll points
x = r * np.sin(theta)
y = r * np.cos(theta)

# Add small noise for complexity
x += np.random.normal(0, 0.1, num_points)
y += np.random.normal(0, 0.1, num_points)
z += np.random.normal(0, 0.1, num_points)

# Create the main figure and GridSpec layout
fig = plt.figure(figsize=(15, 8))  # Increased figure size for better layout
gs = GridSpec(3, 5, width_ratios=[3, 3, 2, 0.5, 0.3], height_ratios=[3, 2, 2], 
             wspace=0.4, hspace=0.4)


# Main 3D Plot
ax_main = fig.add_subplot(gs[:, :2], projection='3d')
sc = ax_main.scatter(x, y, z, c=theta, cmap='viridis', edgecolor='k', s=10, alpha=0.8)
ax_main.set_xlabel("X Axis")
ax_main.set_ylabel("Y Axis")
ax_main.set_zlabel("Z Axis")


# Top View (Projection onto XY plane)
ax_top = fig.add_subplot(gs[0, 2])
ax_top.scatter(x, y, c=theta, cmap='viridis', edgecolor='k', s=10, alpha=0.8)
ax_top.set_xlabel("X Axis")
ax_top.set_ylabel("Y Axis")
ax_top.set_title("Top View (XY Plane)", fontsize=12)
ax_top.set_aspect('equal')
ax_top.grid(True)

# Front View (Projection onto XZ plane)
ax_front = fig.add_subplot(gs[1, 2])
ax_front.scatter(x, z, c=theta, cmap='viridis', edgecolor='k', s=10, alpha=0.8)
ax_front.set_xlabel("X Axis")
ax_front.set_ylabel("Z Axis")
ax_front.set_title("Front View (XZ Plane)", fontsize=12)
ax_front.set_aspect('auto')
ax_front.grid(True)

# Side View (Projection onto YZ plane)
ax_side = fig.add_subplot(gs[2, 2])
ax_side.scatter(y, z, c=theta, cmap='viridis', edgecolor='k', s=10, alpha=0.8)
ax_side.set_xlabel("Y Axis")
ax_side.set_ylabel("Z Axis")
ax_side.set_title("Side View (YZ Plane)", fontsize=12)
ax_side.set_aspect('auto')
ax_side.grid(True)

# Synchronize the axis limits with the main 3D plot
ax_top.set_xlim(ax_main.get_xlim())
ax_top.set_ylim(ax_main.get_ylim())

ax_front.set_xlim(ax_main.get_xlim())
ax_front.set_ylim(ax_main.get_zlim())

ax_side.set_xlim(ax_main.get_ylim())
ax_side.set_ylim(ax_main.get_zlim())

# Add a color bar in its dedicated grid cell
cbar_ax = fig.add_subplot(gs[:, 4])
cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical', label='Theta')
cbar_ax.tick_params(labelsize=10)

fig.patch.set_facecolor('#e6f4d3')  # 为图形设置背景色
ax_main.set_facecolor('#e6f4d3')  # 为3D坐标轴设置背景色
ax_side.set_facecolor('#e6f4d3')
ax_top.set_facecolor('#e6f4d3')
ax_front.set_facecolor('#e6f4d3')

# Optionally, you can use the remaining grid cells for annotations or spacing
# For example, gs[0,3] and gs[1,3] can be left empty or used as needed

# Enhance layout to prevent overlapping
plt.tight_layout()
plt.show()
