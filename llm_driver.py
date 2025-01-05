from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import re
import sys
import ast
import math
import json
from typing import List, Tuple

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
            print("速度为零，方向不可定义。返回 0.0 度")
            return 0.0
        angle_rad = math.atan2(vy, vx)
        angle_deg = math.degrees(angle_rad)
        # 确保角度在 [0, 360) 范围内
        angle_deg = angle_deg % 360
        if DEBUG:
            print(f"行驶方向: {angle_deg} 度")
        return angle_deg

    def print_info(self):
        print(f"type: {self.type}\n position: {self.position}\n "
              f"trajectory: {self.trajectory}\n "
              f"velocity: {self.velocity()}\n "
              f"acceleration: {self.acceleration()}\n "
              f"direction: {self.direction()}\n "
              f"distance: {self.distance_to_hero()}\n")


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
        self.model = OllamaLLM(model=llm_name, temperature=temperature, num_ctx=8000, timeout=100, num_predict=8192)

    @staticmethod
    def distance_top_k(obstacles: List[Obstacle], car_status: CarInfo, k: int = K) -> str:
        sys_prompt = """
        
        """
        prompt_template = ChatPromptTemplate(sys_prompt)

        return

    @staticmethod
    def velocity_top_k(obstacles: List[Obstacle], car_status: CarInfo, k: int = K) -> str:
        pass

    @staticmethod
    def acceleration_top_k(obstacles: List[Obstacle], car_status: CarInfo, k: int = K) -> str:
        pass

    @staticmethod
    def direction_top_k(obstacles: List[Obstacle], car_status: CarInfo, k: int = K) -> str:
        pass

    def run(self, input_text: str) -> str:
        pass


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
    obs = Obstacles(test_text)
    car_info = CarInfo(test_text)
    LLMMultiAgentDriver.distance_top_k(obs.get_obstacles(), car_info)
