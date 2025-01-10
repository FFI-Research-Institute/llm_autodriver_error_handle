import argparse
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple
from util import load_pkl_file, save_to_pkl
import numpy as np
import re
import sys
import ast
import math
import json

example_input = """
input: 
From Distance:
Score: 8
type: vehicle.car
position: (0.50, 9.57)
trajectory: [(0.51, 9.57), (0.51, 9.57), (0.51, 9.57), (0.51, 9.57), (0.52, 9.59), (0.52, 9.56)]
velocity: (0.00, -0.00)
acceleration: (0.00, -0.01)
direction: 315.00
distance: 9.58

Score: 7
type: movable_object.trafficcone
position: (5.25, 0.04)
trajectory: [(5.25, 0.04), (5.25, 0.04), (5.25, 0.04), (5.25, 0.04), (5.25, 0.04), (5.25, 0.01)]
velocity: (0.00, -0.01)
acceleration: (0.00, -0.01)
direction: 270.00
distance: 5.25

Score: 7
type: vehicle.truck
position: (3.59, 7.69)
trajectory: [(3.59, 7.71), (3.59, 7.72), (3.59, 7.74), (3.59, 7.75), (3.59, 7.77), (3.59, 7.79)]
velocity: (0.00, 0.02)
acceleration: (0.00, 0.00)
direction: 90.00
distance: 8.49

From Acceleration:
Score: 10
type: vehicle.car
position: (1.08, 15.43)
trajectory: [(1.12, 15.43), (1.12, 15.43), (1.12, 15.43), (1.08, 15.59), (1.12, 16.25), (1.16, 16.90)]
velocity: (0.01, 0.29)
acceleration: (0.01, 0.16)
direction: 88.44
distance: 15.47

Score: 1
type: movable_object.trafficcone
position: (5.25, 0.04)
trajectory: [(5.25, 0.04), (5.25, 0.04), (5.25, 0.04), (5.25, 0.04), (5.25, 0.04), (5.25, 0.01)]
velocity: (0.00, -0.01)
acceleration: (0.00, -0.01)
direction: 270.00
distance: 5.25

Score: 1
type: movable_object.trafficcone
position: (-13.47, 5.44)
trajectory: [(-13.47, 5.47), (-13.47, 5.47), (-13.47, 5.47), (-13.47, 5.47), (-13.47, 5.47), (-13.47, 5.40)]
velocity: (0.00, -0.01)
acceleration: (0.00, -0.02)
direction: 270.00
distance: 14.53

From Speed:
Score: 2
type: movable_object.trafficcone
position: (5.47, 3.14)
trajectory: [(5.46, 3.15), (5.46, 3.16), (5.46, 3.16), (5.45, 3.17), (5.45, 3.18), (5.46, 3.14)]
velocity: (0.00, -0.00)
acceleration: (0.00, -0.01)
direction: 270.00
distance: 6.31

Score: 2
type: movable_object.trafficcone
position: (-12.32, 9.92)
trajectory: [(-12.32, 9.92), (-12.32, 9.92), (-12.32, 9.92), (-12.32, 9.92), (-12.32, 9.92), (-12.32, 9.92)]
velocity: (0.00, 0.00)
acceleration: (0.00, 0.00)
direction: 0.00
distance: 15.82

Score: 1
type: movable_object.trafficcone
position: (-9.13, 19.80)
trajectory: [(-9.13, 19.81), (-9.13, 19.81), (-9.13, 19.81), (-9.13, 19.79), (-9.12, 19.78), (-9.12, 19.76)]
velocity: (0.00, -0.01)
acceleration: (0.00, -0.00)
direction: 281.31
distance: 21.80

From Direction:
Score: 4
type: movable_object.trafficcone
position: (-9.13, 19.80)
trajectory: [(-9.13, 19.81), (-9.13, 19.81), (-9.13, 19.81), (-9.13, 19.79), (-9.12, 19.78), (-9.12, 19.76)]
velocity: (0.00, -0.01)
acceleration: (0.00, -0.00)
direction: 281.31
distance: 21.80

Score: 4
type: movable_object.trafficcone
position: (-9.70, 10.84)
trajectory: [(-9.70, 10.84), (-9.70, 10.84), (-9.70, 10.84), (-9.70, 10.84), (-9.70, 10.82), (-9.71, 10.79)]
velocity: (-0.00, -0.01)
acceleration: (-0.00, -0.01)
direction: 258.69
distance: 14.55

Score: 4
type: vehicle.car
position: (1.08, 15.43)
trajectory: [(1.12, 15.43), (1.12, 15.43), (1.12, 15.43), (1.08, 15.59), (1.12, 16.25), (1.16, 16.90)]
velocity: (0.01, 0.29)
acceleration: (0.01, 0.16)
direction: 88.44
distance: 15.47


Ego-States:
  Velocity (vx, vy): (-0.00, 0.00)
  Heading Angular Velocity (v_yaw): -0.00
  Acceleration (ax, ay): (0.00, 0.00)
  Can Bus (vx, vy): (-0.03, 0.06)
  Heading Speed: 0.00
  Steering: -0.28

Historical Trajectory (last 2 seconds):
  1. (x: 0.00, y: 0.00)
  2. (x: 0.00, y: 0.00)
  3. (x: 0.00, y: 0.00)
  4. (x: 0.00, y: 0.00)

Mission Goal: FORWARD
"""

example_output = """
output:
Thoughts:
 - Notable Objects from Perception: None
   Potential Effects from Prediction: None
Meta Action: STOP
Trajectory:
[(-0.00,-0.00), (0.00,-0.00), (0.00,0.00), (0.00,0.00), (0.00,-0.00), (0.00,0.00)]
<\example>
"""

system_prompt_consider_distance = """
**System Prompt:**
You are a member of a multi-agent system responsible for evaluating the impact of obstacles on the ego vehicle. Only consider the distance between the obstacle and the ego vehicle for scoring. The score range is from 0 to 10, where 0 means the obstacle is very far and not important, and 10 means the obstacle is very close and extremely important.
Here is the input for the obstacle and ego vehicle information:
### Input Format:
1. **Obstacle Information:**
   - Type: `type`
   - Position: `position`
   - Distance: `distance` (distance between the obstacle and the ego vehicle)
2. **Ego Vehicle Information:**
   - Current position (detailed information is not necessary, only that the vehicle is present)
3. **Mission Goal:**
   - Goal: `Mission Goal` (e.g., FORWARD)
### Scoring Rules:
- **0-2 points**: The obstacle is very far, unlikely to pose a threat to the ego vehicle, distance is greater than 15 meters.
- **3-5 points**: The obstacle is somewhat distant, may pose a limited threat to the ego vehicle, distance is between 10 and 15 meters.
- **6-8 points**: The obstacle is relatively close, requires attention, distance is between 5 and 10 meters.
- **9-10 points**: The obstacle is very close to the ego vehicle, immediate action is required, distance is less than 5 meters.
### Example:
1. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-12.32, 9.92)
     - Distance: 15.82
   - **Ego Vehicle Information:**
     - Mission Goal: FORWARD

   **Output:**
   - Based on the distance between the obstacle and the ego vehicle, the score is: `4`

2. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-5.50, 3.10)
     - Distance: 5.0
   - **Ego Vehicle Information:**
     - Mission Goal: FORWARD

   **Output:**
   - Based on the distance between the obstacle and the ego vehicle, the score is: `7`

Please provide a score for the importance of the obstacle based on the distance between it and the ego vehicle.
"""

system_prompt_consider_speed = """
**System Prompt:**

You are a member of a multi-agent system responsible for evaluating the impact of obstacles on the ego vehicle. Only consider the speed of the ego vehicle for scoring. The score range is from 0 to 10, where 0 means the obstacle is unlikely to pose a threat based on the ego vehicle's speed, and 10 means the obstacle poses a significant risk based on the ego vehicle's speed.

Here is the input for the obstacle and ego vehicle information:

### Input Format:
1. **Obstacle Information:**
   - Type: `type`
   - Position: `position`
   - Distance: `distance` (distance between the obstacle and the ego vehicle)
   
2. **Ego Vehicle Information:**
   - Speed: `velocity (vx, vy)` (speed components of the ego vehicle in the x and y directions)
   
3. **Mission Goal:**
   - Goal: `Mission Goal` (e.g., FORWARD)

### Scoring Rules:
- **0-2 points**: The ego vehicle is moving very slowly or is stationary (e.g., low or zero speed). Obstacles at a reasonable distance are unlikely to pose a threat.
- **3-5 points**: The ego vehicle is moving at a moderate speed. Obstacles within a reasonable distance may pose a limited threat.
- **6-8 points**: The ego vehicle is moving at a high speed. Obstacles closer to the ego vehicle are a significant concern.
- **9-10 points**: The ego vehicle is moving very fast. Obstacles in close proximity are very dangerous and require immediate action.

### Example:
1. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-12.32, 9.92)
     - Distance: 15.82
   - **Ego Vehicle Information:**
     - Speed: (-0.20, 0.00)
     - Mission Goal: FORWARD

   **Output:**
   - Based on the ego vehicle's speed, the score is: `2`

2. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-5.50, 3.10)
     - Distance: 5.0
   - **Ego Vehicle Information:**
     - Speed: (4.00, 0.00)
     - Mission Goal: FORWARD

   **Output:**
   - Based on the ego vehicle's speed, the score is: `7`

Please provide a score for the importance of the obstacle based on the ego vehicle's speed.
"""

system_prompt_consider_acceleration = """
**System Prompt:**

You are a member of a multi-agent system responsible for evaluating the impact of obstacles on the ego vehicle. Only consider the acceleration of the ego vehicle for scoring. The score range is from 0 to 10, where 0 means the obstacle is unlikely to pose a threat based on the ego vehicle's acceleration, and 10 means the obstacle poses a significant risk based on the ego vehicle's acceleration.

Here is the input for the obstacle and ego vehicle information:

### Input Format:
1. **Obstacle Information:**
   - Type: `type`
   - Position: `position`
   - Distance: `distance` (distance between the obstacle and the ego vehicle)

2. **Ego Vehicle Information:**
   - Acceleration: `acceleration (ax, ay)` (acceleration components of the ego vehicle in the x and y directions)

3. **Mission Goal:**
   - Goal: `Mission Goal` (e.g., FORWARD)

### Scoring Rules:
- **0-2 points**: The ego vehicle has low or zero acceleration (i.e., no significant acceleration or deceleration). Obstacles at a reasonable distance are unlikely to pose a threat.
- **3-5 points**: The ego vehicle has moderate acceleration (e.g., gradual increase in speed or deceleration). Obstacles within a reasonable distance may pose a limited threat.
- **6-8 points**: The ego vehicle is experiencing high acceleration (e.g., fast acceleration or rapid deceleration). Obstacles closer to the ego vehicle are a significant concern.
- **9-10 points**: The ego vehicle has very high acceleration (e.g., very fast acceleration or emergency braking). Obstacles in close proximity are very dangerous and require immediate action.

### Example:
1. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-12.32, 9.92)
     - Distance: 15.82
   - **Ego Vehicle Information:**
     - Acceleration: (0.00, 0.50)
     - Mission Goal: FORWARD

   **Output:**
   - Based on the ego vehicle's acceleration, the score is: `3`

2. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-5.50, 3.10)
     - Distance: 5.0
   - **Ego Vehicle Information:**
     - Acceleration: (0.50, 0.00)
     - Mission Goal: FORWARD

   **Output:**
   - Based on the ego vehicle's acceleration, the score is: `7`

Please provide a score for the importance of the obstacle based on the ego vehicle's acceleration.
"""

system_prompt_consider_direction = """
**System Prompt:**

You are a member of a multi-agent system responsible for evaluating the impact of obstacles on the ego vehicle. Only consider the direction of the ego vehicle for scoring. The score range is from 0 to 10, where 0 means the obstacle is unlikely to pose a threat based on the ego vehicle's direction, and 10 means the obstacle poses a significant risk based on the ego vehicle's direction.

Here is the input for the obstacle and ego vehicle information:

### Input Format:
1. **Obstacle Information:**
   - Type: `type`
   - Position: `position`
   - Distance: `distance` (distance between the obstacle and the ego vehicle)

2. **Ego Vehicle Information:**
   - Direction: `direction` (the heading direction of the ego vehicle in radians)

3. **Mission Goal:**
   - Goal: `Mission Goal` (e.g., FORWARD)

### Scoring Rules:
- **0-2 points**: The ego vehicle's direction is such that the obstacle is not within its immediate path, or the obstacle is in an irrelevant direction (e.g., 90° or 270° to the ego vehicle's heading). The obstacle is unlikely to pose a threat.
- **3-5 points**: The ego vehicle's direction indicates the obstacle is somewhat within the vehicle's path, but not a direct threat (e.g., 45° or 135° to the ego vehicle's heading). The obstacle may pose a limited threat depending on further movement.
- **6-8 points**: The ego vehicle's direction aligns relatively closely with the obstacle's position (e.g., 0° or 180° to the ego vehicle's heading). The obstacle is on a direct path or close to it and requires attention.
- **9-10 points**: The ego vehicle's direction is directly aligned with the obstacle (e.g., 0° to 10° or 170° to 180°). The obstacle is in the vehicle's direct path and poses an immediate danger.

### Example:
1. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-12.32, 9.92)
     - Distance: 15.82
   - **Ego Vehicle Information:**
     - Direction: 0.00 (Heading directly forward)
     - Mission Goal: FORWARD

   **Output:**
   - Based on the ego vehicle's direction, the score is: `2`

2. **Input:**
   - **Obstacle Information:**
     - Type: `movable_object.trafficcone`
     - Position: (-5.50, 3.10)
     - Distance: 5.0
   - **Ego Vehicle Information:**
     - Direction: 0.00 (Heading directly forward)
     - Mission Goal: FORWARD

   **Output:**
   - Based on the ego vehicle's direction, the score is: `8`

Please provide a score for the importance of the obstacle based on the ego vehicle's direction.
"""

system_prompt_final_decision = """
**Autonomous Driving Trajectory Planner**

**Role**: You are the brain of an autonomous vehicle. Plan a safe and physically feasible 3-second driving trajectory, avoiding critical obstacles, ensuring smoothness, and considering the vehicle's dynamic constraints. **The coordinate changes between waypoints must be smooth and must not exhibit sudden, physically unrealistic changes**. The output format must be strictly fixed as: `[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]`.

### **Context:**
- **Coordinate System**: The X-axis is perpendicular, and the Y-axis is parallel to the direction you're facing. You are at point (0, 0).
- **Objective**: Generate a smooth and physically valid 3-second route using 6 waypoints (one every 0.5 seconds), ensuring no collisions and avoiding any sudden, physically unrealistic changes in coordinates.

### **Inputs:**

1. **Obstacle Information:**
   - **Obstacle Selection Criteria**: Select and evaluate the most important obstacles based on the following four perspectives:
     - **From Distance**: The top 3 obstacles based on proximity to the ego vehicle.
     - **From Velocity**: The top 3 obstacles based on relative speed with respect to the ego vehicle.
     - **From Direction**: The top 3 obstacles based on their relative direction (heading) to the ego vehicle's heading.
     - **From Acceleration**: The top 3 obstacles based on their acceleration/deceleration relative to the ego vehicle.

   Each obstacle is described by:
   - **Type**: `str`
   - **Position**: `(x, y)`
   - **Trajectory**: A list of predicted positions over time.
   - **Velocity**: `(vx, vy)` — The velocity components of the obstacle.
   - **Acceleration**: `(ax, ay)` — The acceleration components of the obstacle.
   - **Direction**: The heading of the obstacle (in degrees).
   - **Distance**: The distance between the obstacle and the ego vehicle.

2. **Ego Vehicle State:**
   - **Velocity**: `(vx, vy)` — The velocity components of the ego vehicle.
   - **Heading Angular Velocity**: `v_yaw` — The rate of change of the vehicle's heading.
   - **Acceleration**: `(ax, ay)` — The acceleration components of the ego vehicle.
   - **CAN Bus**: `(vx, vy)` — The vehicle's speed components from the CAN bus.
   - **Heading Speed**: The speed of the vehicle's heading (angular speed).
   - **Steering**: The steering angle of the vehicle.
   - **Physical Constraints**: Consider the vehicle's maximum acceleration, turning radius, and other physical limitations to ensure the generated trajectory is smooth and feasible.

3. **Mission Goal:**
   - **Goal**: The target location or direction for the vehicle over the next 3 seconds (e.g., **FORWARD**).

### **Task:**
- **Thought Process**:
  - **Notable Obstacles**: Identify the most critical obstacles based on the four perspectives (distance, velocity, direction, and acceleration).
  - **Potential Effects**: Evaluate how these obstacles might impact the ego vehicle's trajectory, considering their speed, direction, and future predicted positions.

- **Action Plan**: 
  - Adjust speed, steering, or trajectory to avoid collisions. Ensure that all actions are within the dynamic limits of the ego vehicle (e.g., maximum acceleration, steering angle).
  - Smoothly transition between waypoints to avoid sudden speed or direction changes that could destabilize the vehicle. **Ensure that the coordinate changes between waypoints are physically realistic and avoid any sudden, unrealistic jumps.**

- **Trajectory Planning**: 
  - Generate a safe, feasible, and smooth 3-second trajectory that avoids all obstacles and respects the physical constraints of the ego vehicle. The trajectory should consist of **6 waypoints** (one every 0.5 seconds), with each point being a calculated safe position. **The coordinate changes between waypoints must be continuous and physically plausible, without any sudden, unrealistic jumps or sharp changes**.

### **Output**:

- **Thoughts**:
  - **Notable Obstacles**: Describe the most critical obstacles selected based on the four criteria and their relative risk to the ego vehicle.
  - **Potential Effects**: Analyze how the selected obstacles might influence trajectory planning, considering their distance, speed, direction, and acceleration.

- **Meta Action**:
  - Provide the actions you would take to ensure the ego vehicle remains safe, such as adjusting speed, steering, or initiating evasive maneuvers. Ensure these actions are physically feasible and the trajectory remains smooth.

- **Trajectory (MOST IMPORTANT)**: 
  - A list of 6 waypoints in the fixed format: **`[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]`**. Ensure the path is smooth, collision-free, and within the vehicle's physical constraints. **The changes in coordinates between waypoints must adhere to physical laws, avoiding any sudden, unrealistic jumps or abrupt shifts.**

<example>
""" + example_input + example_output

DEBUG = False

K = 3


class CarInfo:
    def __init__(self, info_text: str):
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.angular_velocity: float = 0.0
        self.acceleration: Tuple[float, float] = (0.0, 0.0)
        self.can_bus: Tuple[float, float] = (0.0, 0.0)
        self.heading_speed: float = 0.0
        self.steering: float = 0.0
        self.history_trajectory: List[Tuple[float, float]] = []
        self.mission_goal: str = ""
        info_text = f"Ego-States:{info_text.split('Ego-States:')[-1]}"
        self.parse(info_text)

    def parse(self, info_text: str):
        # 定义正则表达式模式
        patterns = {
            'velocity': r'-\s*Velocity\s*\(vx,vy\):\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)',
            'angular_velocity': r'-\s*Heading Angular Velocity\s*\(v_yaw\):\s*\((-?\d+\.\d+)\)',
            'acceleration': r'-\s*Acceleration\s*\(ax,ay\):\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)',
            'can_bus': r'-\s*Can Bus:\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)',
            'heading_speed': r'-\s*Heading Speed:\s*\((-?\d+\.\d+)\)',
            'steering': r'-\s*Steering:\s*\((-?\d+\.\d+)\)',
            'historical_trajectory': r'Historical Trajectory.*?:\s*\[(.*?)\]',
            'mission_goal': r'Mission Goal:\s*(\w+)'
        }

        # 提取各个属性
        velocity_match = re.search(patterns['velocity'], info_text)
        if velocity_match:
            self.velocity = (float(velocity_match.group(1)), float(velocity_match.group(2)))

        angular_velocity_match = re.search(patterns['angular_velocity'], info_text)
        if angular_velocity_match:
            self.angular_velocity = float(angular_velocity_match.group(1))

        acceleration_match = re.search(patterns['acceleration'], info_text)
        if acceleration_match:
            self.acceleration = (float(acceleration_match.group(1)), float(acceleration_match.group(2)))

        can_bus_match = re.search(patterns['can_bus'], info_text)
        if can_bus_match:
            self.can_bus = (float(can_bus_match.group(1)), float(can_bus_match.group(2)))

        heading_speed_match = re.search(patterns['heading_speed'], info_text)
        if heading_speed_match:
            self.heading_speed = float(heading_speed_match.group(1))

        steering_match = re.search(patterns['steering'], info_text)
        if steering_match:
            self.steering = float(steering_match.group(1))

        historical_trajectory_match = re.search(patterns['historical_trajectory'], info_text, re.DOTALL)
        if historical_trajectory_match:
            traj_str = historical_trajectory_match.group(1)
            # 提取所有的 (x, y) 坐标
            points = re.findall(r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)', traj_str)
            self.history_trajectory = [(float(x), float(y)) for x, y in points]

        mission_goal_match = re.search(patterns['mission_goal'], info_text)
        if mission_goal_match:
            self.mission_goal = mission_goal_match.group(1)

    def to_json(self) -> str:
        # 将类的属性转换为字典
        data = {
            "Ego-States": {
                "Velocity": {
                    "vx": self.velocity[0],
                    "vy": self.velocity[1]
                },
                "Heading Angular Velocity": {
                    "v_yaw": self.angular_velocity
                },
                "Acceleration": {
                    "ax": self.acceleration[0],
                    "ay": self.acceleration[1]
                },
                "Can Bus": {
                    "vx": self.can_bus[0],
                    "vy": self.can_bus[1]
                },
                "Heading Speed": self.heading_speed,
                "Steering": self.steering
            },
            "Historical Trajectory (last 2 seconds)": self.history_trajectory,
            "Mission Goal": self.mission_goal
        }
        return json.dumps(data, indent=4, ensure_ascii=False)

    def get_info(self) -> str:
        output_lines = []
        output_lines.append("Ego-States:")
        output_lines.append(f"  Velocity (vx, vy): ({self.velocity[0]:.2f}, {self.velocity[1]:.2f})")
        output_lines.append(f"  Heading Angular Velocity (v_yaw): {self.angular_velocity:.2f}")
        output_lines.append(f"  Acceleration (ax, ay): ({self.acceleration[0]:.2f}, {self.acceleration[1]:.2f})")
        output_lines.append(f"  Can Bus (vx, vy): ({self.can_bus[0]:.2f}, {self.can_bus[1]:.2f})")
        output_lines.append(f"  Heading Speed: {self.heading_speed:.2f}")
        output_lines.append(f"  Steering: {self.steering:.2f}")

        output_lines.append("\nHistorical Trajectory (last 2 seconds):")
        if self.history_trajectory:
            for idx, (x, y) in enumerate(self.history_trajectory, start=1):
                output_lines.append(f"  {idx}. (x: {x:.2f}, y: {y:.2f})")
        else:
            output_lines.append("  No historical trajectory data available.")

        output_lines.append(f"\nMission Goal: {self.mission_goal}")

        # 将所有行组合成一个单一的字符串
        output_str = "\n".join(output_lines)

        # 打印到控制台
        if DEBUG:
            print(output_str)

        # 返回打印的字符串
        return output_str

    # def __repr__(self):
    #     return self.to_json()


class Obstacle:
    def __init__(self, trajectory: List[Tuple[float, float]], obstacle_type: str, position: Tuple[float, float]):
        """
        初始化障碍物对象。

        :param trajectory: 障碍物未来六秒内每秒的位置信息，列表中每个元素是一个 (x, y) 坐标元组。
        :param obstacle_type: 障碍物类型（字符串）。
        :param position: 障碍物的当前位置 (x, y) 坐标元组。
        """
        self.trajectory = trajectory  # list of (x, y)
        self.type = obstacle_type
        self.position = position

    def __eq__(self, other):
        return self.type == other.type and self.position == other.position

    def distance_to_hero(self) -> float:
        """
        计算障碍物当前位置与我们的车辆（位于原点 (0,0)）之间的欧氏距离。

        :return: 距离值（浮点数）。
        """
        x, y = self.position
        distance = math.hypot(x, y)
        if DEBUG:
            print(f"计算距离: sqrt({x}^2 + {y}^2) = {distance}")
        return distance

    def velocity(self) -> Tuple[float, float]:
        """
        计算障碍物的平均速度向量（单位：米/秒），基于未来六秒的轨迹。

        :return: 速度向量 (vx, vy)。
        """
        if len(self.trajectory) < 2:
            if DEBUG:
                print("轨迹数据不足，无法计算速度。返回 (0.0, 0.0)")
            return (0.0, 0.0)

        total_vx = 0.0
        total_vy = 0.0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][0] - self.trajectory[i - 1][0]
            dy = self.trajectory[i][1] - self.trajectory[i - 1][1]
            total_vx += dx
            total_vy += dy
            if DEBUG:
                print(f"第 {i} 秒速度分量: vx = {dx}, vy = {dy}")

        avg_vx = total_vx / (len(self.trajectory) - 1)
        avg_vy = total_vy / (len(self.trajectory) - 1)
        if DEBUG:
            print(f"平均速度向量: ({avg_vx}, {avg_vy})")
        return (avg_vx, avg_vy)

    def acceleration(self) -> Tuple[float, float]:
        """
        计算障碍物的平均加速度向量（单位：米/秒²），基于未来六秒的轨迹。

        :return: 加速度向量 (ax, ay)。
        """
        if len(self.trajectory) < 3:
            if DEBUG:
                print("轨迹数据不足，无法计算加速度。返回 (0.0, 0.0)")
            return (0.0, 0.0)

        # 首先计算每一秒的速度向量
        velocities = []
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][0] - self.trajectory[i - 1][0]
            dy = self.trajectory[i][1] - self.trajectory[i - 1][1]
            velocities.append((dx, dy))
            if DEBUG:
                print(f"第 {i} 秒速度向量: ({dx}, {dy})")

        # 计算速度向量的变化量，即加速度
        total_ax = 0.0
        total_ay = 0.0
        for i in range(1, len(velocities)):
            dax = velocities[i][0] - velocities[i - 1][0]
            day = velocities[i][1] - velocities[i - 1][1]
            total_ax += dax
            total_ay += day
            if DEBUG:
                print(f"第 {i} 秒加速度分量: ax = {dax}, ay = {day}")

        avg_ax = total_ax / (len(velocities) - 1)
        avg_ay = total_ay / (len(velocities) - 1)
        if DEBUG:
            print(f"平均加速度向量: ({avg_ax}, {avg_ay})")
        return (avg_ax, avg_ay)

    def direction(self) -> float:
        """
        计算障碍物的行驶方向（以度为单位，相对于正x轴的角度）。

        :return: 行驶方向角度（浮点数）。
        """
        vx, vy = self.velocity()
        if vx == 0 and vy == 0:
            if DEBUG:
                print("速度为零，方向不可定义。返回 0.0 度")
            return 0.0
        angle_rad = math.atan2(vy, vx)
        angle_deg = math.degrees(angle_rad)
        # 确保角度在 [0, 360) 范围内
        angle_deg = angle_deg % 360
        if DEBUG:
            print(f"行驶方向: {angle_deg} 度")
        return angle_deg

    def get_info(self):
        # 格式化位置
        position_formatted = f"({self.position[0]:.2f}, {self.position[1]:.2f})"

        # 格式化轨迹
        trajectory_formatted = "[" + ", ".join(f"({x:.2f}, {y:.2f})" for x, y in self.trajectory) + "]"

        # 格式化速度
        velocity_x, velocity_y = self.velocity()
        velocity_formatted = f"({velocity_x:.2f}, {velocity_y:.2f})"

        # 格式化加速度
        acceleration_x, acceleration_y = self.acceleration()
        acceleration_formatted = f"({acceleration_x:.2f}, {acceleration_y:.2f})"

        # 格式化方向
        direction_formatted = f"{self.direction():.2f}"

        # 格式化距离
        distance_formatted = f"{self.distance_to_hero():.2f}"

        # 打印信息
        info_str = (
            f"type: {self.type}\n"
            f"position: {position_formatted}\n"
            f"trajectory: {trajectory_formatted}\n"
            f"velocity: {velocity_formatted}\n"
            f"acceleration: {acceleration_formatted}\n"
            f"direction: {direction_formatted}\n"
            f"distance: {distance_formatted}\n"
        )
        if DEBUG:
            print(info_str)
        return info_str


class Obstacles:
    def __init__(self, input_text: str):
        self.obstacles = []
        for line in input_text.splitlines():
            if "Perception and Prediction" in line:
                continue
            if "Ego-States" in line:
                break
            stripped_line = line.strip()
            if stripped_line:  # 跳过空行
                obj = self.parse_obstacle_line(stripped_line)
                if obj:
                    self.obstacles.append(obj)

    @staticmethod
    def parse_obstacle_line(line: str) -> Obstacle:
        # 更新正则表达式，允许行首有空白字符，并严格匹配整行
        pattern = r"^\s*-\s+([\w\.]+)\s+at\s+\(([-\d.]+),([-\d.]+)\)\.\s+Future trajectory:\s+(\[.*\])$"
        match = re.match(pattern, line)
        if match:
            obj_type = match.group(1)  # 障碍物名称
            x = float(match.group(2))  # 当前X坐标
            y = float(match.group(3))  # 当前Y坐标
            traj_str = match.group(4)  # 轨迹字符串

            try:
                # 将 'UN' 替换为 'None' 以便 ast.literal_eval 可以解析
                traj_str_clean = traj_str.replace('UN', 'None')

                # 使用 ast.literal_eval 解析轨迹字符串
                traj = ast.literal_eval(traj_str_clean)

                # 确保轨迹是列表且每个元素是元组
                if isinstance(traj, list) and all(isinstance(t, tuple) and len(t) == 2 for t in traj):
                    # 将 'None' 保留为 None，数值保持为浮点数
                    traj_processed = []
                    for px, py in traj:
                        # px 和 py 可能是 float 或 None
                        px_val = float(px) if isinstance(px, (int, float)) else None
                        py_val = float(py) if isinstance(py, (int, float)) else None
                        traj_processed.append((px_val, py_val))
                    traj = traj_processed
                else:
                    raise ValueError("轨迹格式不正确")
            except Exception as e:
                print(f"解析轨迹时出错: {e} 在行: {line}")
                traj = []
                sys.exit(-1)

            return Obstacle(traj, obj_type, (x, y))
        else:
            print(f"未匹配行: {line}")
            sys.exit(-1)

    def get_obstacles(self) -> List[Obstacle]:
        return self.obstacles


class LLMMultiAgentDriver:
    def __init__(self, llm_name: str, temperature: float = 0.0):
        self.model = OllamaLLM(model=llm_name,
                               temperature=temperature,
                               num_ctx=4096,
                               timeout=100,
                               num_predict=4096)

    @staticmethod
    def parse_score(response: str) -> int:
        """
        解析模型响应，提取0到10之间的整数分数。返回最后一个分数。
        """
        matches = re.findall(r'\b([0-9]|10)\b', response)

        if matches:
            score = int(matches[-1])  # 获取最后一个匹配的分数
            return score
        else:
            # 如果无法提取到有效分数，可以设置一个默认值或引发异常
            print(f"无法从模型响应中提取分数。响应内容: {response}")
            return 0

    @staticmethod
    def extract_last_trajectory(input_string: str) -> np.ndarray | None:
        """
        从输入的字符串中提取所有六个坐标对的轨迹，并返回最后一个匹配的轨迹数据。

        参数:
        input_string (str): 输入的字符串，包含一个或多个轨迹数据。

        返回:
        np.ndarray 或 None: 如果匹配到轨迹数据，返回最后一个匹配的轨迹；否则返回 None。
        """
        # 正则表达式，确保捕获包含六个坐标对的轨迹
        pattern = r"\[\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)\s*(?:,\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)\s*){5}\s*\]"

        # 使用 re.finditer 查找所有匹配的轨迹
        matches = re.finditer(pattern, input_string, re.DOTALL)

        # 最后一个匹配项
        last_match = None

        # 遍历所有匹配结果，更新 last_match 为最后一个匹配的结果
        for match in matches:
            last_match = match

        # 如果找到了最后一个匹配项
        if last_match:
            # 获取最后一个匹配到的轨迹字符串
            traj = last_match.group(0)

            # 使用 ast.literal_eval 安全地解析字符串为 Python 对象
            traj = ast.literal_eval(traj)

            # 将解析出的轨迹转换为 NumPy 数组
            traj = np.array(traj)

            # 返回处理后的轨迹数据
            return traj
        else:
            # 如果没有匹配到任何轨迹，返回 None
            return None

    def select_top_k(self, sys_prompt, obstacles: Obstacles, car_status: CarInfo, k: int = K) -> List[
        Tuple[Obstacle, int]]:
        score_list = list()
        prompt_template = ChatPromptTemplate([
            ("system", sys_prompt),
            ("user", "{input}")
        ])

        for obstacle in obstacles.get_obstacles():
            user_input = f"{obstacle.get_info()} \n{car_status.get_info()}"
            chain = prompt_template | self.model | StrOutputParser()
            output_str = chain.invoke({"input": user_input})
            num = self.parse_score(output_str)
            score_list.append((obstacle, num))
            # DEBUG = True
            if DEBUG:
                print(f"input: {user_input}")
                print(f"output: {output_str}")
        return sorted(score_list, key=lambda x: x[1], reverse=True)[:k]

    def driver_decision(self,
                        distance_select_list: List[Tuple['Obstacle', int]],
                        acceleration_select_list: List[Tuple['Obstacle', int]],
                        speed_select_list: List[Tuple['Obstacle', int]],
                        direction_select_list: List[Tuple['Obstacle', int]],
                        car_info: 'CarInfo') -> np.ndarray | None:
        # Step 1: Process the obstacles' information
        obstacles_info = ""

        # Function to process each list of selected obstacles
        def process_obstacle_list(obstacle_list: List[Tuple['Obstacle', int]], category: str):
            nonlocal obstacles_info
            obstacles_info += f"{category}:\n"
            for obstacle, score in obstacle_list:
                obstacles_info += f"Score: {score}\n"
                obstacles_info += obstacle.get_info() + "\n"

        # Process obstacles for each category
        process_obstacle_list(distance_select_list, "From Distance")
        process_obstacle_list(acceleration_select_list, "From Acceleration")
        process_obstacle_list(speed_select_list, "From Speed")
        process_obstacle_list(direction_select_list, "From Direction")

        # Step 2: Process the car's information
        car_info_str = car_info.get_info()

        # Step 3: Combine all the information and return as the final input string
        input_str = obstacles_info + "\n" + car_info_str
        if DEBUG:
            print(input_str)

        prompt_template = ChatPromptTemplate([
            ("system", system_prompt_final_decision),
            ("user", "{input}")
        ])

        chain = prompt_template | self.model | StrOutputParser()
        output_str = chain.invoke({"input": input_str})
        if DEBUG:
            print(f"input: \n{input_str}")
            print(f"output: \n{output_str}")

        return self.extract_last_trajectory(output_str)

    def run(self, input_text: str) -> str:
        obs = Obstacles(input_text)
        car_info = CarInfo(input_text)

        if DEBUG:
            print(car_info.get_info())
            for x in obs.get_obstacles():
                print(x.get_info())

        distance_select = self.select_top_k(system_prompt_consider_distance, obs, car_info)
        acceleration_select = self.select_top_k(system_prompt_consider_acceleration, obs, car_info)
        speed_select = self.select_top_k(system_prompt_consider_speed, obs, car_info)
        direction_select = self.select_top_k(system_prompt_consider_direction, obs, car_info)
        decision = self.driver_decision(distance_select, acceleration_select, speed_select, direction_select, car_info)
        return decision


if __name__ == "__main__":
    test_text = """
Perception and Prediction:
- movable_object.trafficcone at (5.25,0.04). Future trajectory: [(5.25,0.04),(5.25,0.04),(5.25,0.04),(5.25,0.04),(5.25,0.04),(5.25,0.01)]
- movable_object.trafficcone at (-13.47,5.44). Future trajectory: [(-13.47,5.47),(-13.47,5.47),(-13.47,5.47),(-13.47,5.47),(-13.47,5.47),(-13.47,5.4)]
- movable_object.trafficcone at (-9.13,19.8). Future trajectory: [(-9.13,19.81),(-9.13,19.81),(-9.13,19.81),(89.87,-75.21000000000001),(-9.12,19.78),(-105.12,96.76)]
- movable_object.trafficcone at (-9.3,16.88). Future trajectory: [(-9.3,16.88),(-9.3,16.88),(-9.3,16.88),(-9.3,16.88),(-9.3,16.88),(-9.3,16.88)]
- movable_object.trafficcone at (-13.37,8.49). Future trajectory: [(-13.37,8.49),(-13.37,8.49),(-13.37,8.49),(-13.37,8.49),(-13.37,8.49),(-13.36,8.41)]
- movable_object.trafficcone at (-9.7,10.84). Future trajectory: [(-9.7,10.84),(-9.7,10.84),(-9.7,10.84),(-9.7,10.84),(-9.7,10.82),(-9.71,10.79)]
- vehicle.car at (1.08,15.43). Future trajectory: [(1.12,15.43),(1.12,15.43),(1.12,15.43),(1.08,15.59),(1.12,16.25),(1.16,16.9)]
- vehicle.car at (0.5,9.57). Future trajectory: [(0.51,9.57),(0.51,9.57),(0.51,9.57),(0.51,9.57),(0.52,9.59),(0.52,9.56)]
- movable_object.trafficcone at (-9.47,18.3). Future trajectory: [(-9.47,18.3),(-9.47,18.3),(-9.47,18.3),(-9.47,18.3),(-9.47,18.3),(-9.47,18.3)]
- movable_object.barrier at (-9.4,19.11). Future trajectory: [(-9.41,19.11),(-9.41,19.11),(-9.42,19.07),(-9.42,19.03),(-9.42,19.0),(-9.43,18.96)]
- movable_object.trafficcone at (5.47,3.14). Future trajectory: [(5.46,3.15),(5.46,3.16),(5.46,3.16),(5.45,3.17),(5.45,3.18),(5.46,3.14)]
- vehicle.construction at (-11.76,14.44). Future trajectory: [(-11.76,14.44),(-11.76,14.43),(-11.76,14.43),(-11.76,14.43),(-11.76,14.43),(-11.76,14.43)]
- movable_object.trafficcone at (-9.29,13.81). Future trajectory: [(-9.31,13.81),(-9.31,13.81),(-9.29,13.8),(-9.31,13.77),(-9.33,13.75),(-9.33,13.75)]
- vehicle.truck at (3.59,7.69). Future trajectory: [(3.59,7.71),(3.59,7.72),(3.59,7.74),(3.59,7.75),(3.59,7.77),(3.59,7.79)]
- movable_object.trafficcone at (-12.32,9.92). Future trajectory: [(-12.32,9.92),(-12.32,9.92),(-12.32,9.92),(-12.32,9.92),(-12.32,9.92),(-12.32,9.92)]

Ego-States:
 - Velocity (vx,vy): (-0.20,0.00)
 - Heading Angular Velocity (v_yaw): (-1.00)
 - Acceleration (ax,ay): (0.00,0.50)
 - Can Bus: (-0.03,0.06)
 - Heading Speed: (0.00)
 - Steering: (-0.28)
Historical Trajectory (last 2 seconds): [(0.10,0.00), (0.20,0.00), (0.30,0.00), (0.40,0.00)]
Mission Goal: FORWARD 
    """
    test2 = """
output: 
**Thought Process:**

Based on the input obstacles selected from multiple perspectives (distance, velocity, direction, and acceleration), I identify the most critical ones that pose a risk to the ego vehicle.

From Distance:
The top three obstacles are:

1. A vehicle.car at position (0.50, 9.57) with a score of 8.
2. A movable_object.trafficcone at position (5.25, 0.04) with a score of 7.
3. A movable_object.trafficcone at position (5.47, 3.14) with a score of 7.

From Velocity:
The top three obstacles are:

1. A vehicle.car at position (0.50, 9.57) with a score of 9.
2. A movable_object.trafficcone at position (5.47, 3.14) with a score of 6.
3. A movable_object.trafficcone at position (5.25, 0.04) with a score of 5.

From Direction:
The top three obstacles are:

1. A movable_object.trafficcone at position (-13.47, 5.44) with a score of 8.
2. A movable_object.trafficcone at position (-9.13, 19.80) with a score of 4.
3. A movable_object.trafficcone at position (-9.70, 10.84) with a score of 4.

From Acceleration:
The top three obstacles are:

1. A vehicle.car at position (0.50, 9.57) with a score of 9.
2. A movable_object.trafficcone at position (5.47, 3.14) with a score of 6.
3. A movable_object.trafficcone at position (5.25, 0.04) with a score of 5.

Considering the potential effects of these obstacles on the trajectory planning, I prioritize avoiding collisions and near-misses.

**Meta Action:**

To ensure the ego vehicle remains safe, I will:

1. Adjust speed to maintain a safe distance from the closest obstacle (vehicle.car at position (0.50, 9.57)).
2. Steer away from the traffic cones at positions (5.25, 0.04) and (5.47, 3.14).
3. Monitor the direction of the movable_object.trafficcones at positions (-13.47, 5.44), (-9.13, 19.80), and (-9.70, 10.84) to avoid any potential collisions.

**Trajectory Planning:**

Based on the analysis of obstacles and the ego vehicle's current state, I plan a safe and feasible 3-second route for the ego vehicle. The route consists of **6 waypoints**, one every 0.5 seconds:

1. (x: 0.10, y: 0.00)
2. (x: 0.20, y: 0.00)
3. (x: 0.30, y: 0.00)
4. (x: 0.40, y: 0.00)
5. (x: 0.50, y: 9.56)
6. (x: 0.52, y: 9.59)

This trajectory takes into account the selected obstacles and ensures the ego vehicle avoids any potential collisions or hazards while maintaining a safe distance from the closest obstacle.

**Output:**

Thoughts:

* Notable Objects: The top three obstacles from distance are a vehicle.car at position (0.50, 9.57), a movable_object.trafficcone at position (5.25, 0.04), and another movable_object.trafficcone at position (5.47, 3.14).
* Potential Effects: These obstacles pose a risk to the ego vehicle's trajectory planning, requiring adjustments in speed and steering.

Meta Action:

* Adjust speed to maintain a safe distance from the closest obstacle.
* Steer away from the traffic cones at positions (5.25, 0.04) and (5.47, 3.14).
* Monitor the direction of the movable_object.trafficcones at positions (-13.47, 5.44), (-9.13, 19.80), and (-9.70, 10.84) to avoid any potential collisions.

Trajectory:

[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
= [(0.10, 0.00), (0.20, 0.00), (0.30,0.00), (0.40,  0.00), (0.50, 9.56), (0.52, 9.59)]    
    
    """
    llm_driver = LLMMultiAgentDriver("llama3")
    data = load_pkl_file("data/our_dataset/basic_dataset/basic_dataset_default_middle.pkl")
    result_dict = dict()
    error_token_list = list()
    for key, val in tqdm(data.items(), total=len(data)):
        # print(val['ground_truth'])
        this_result = llm_driver.run(val["input"])
        if this_result is not None:
            result_dict[key] = this_result
        else:
            error_token_list.append(key)

    print("error tokens:")
    for x in error_token_list:
        print(x)
    save_to_pkl(result_dict, "outputs/pkl/basic_dataset_multi_agent_llama3_no_error_middle.pkl")
