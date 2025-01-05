import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import pickle
import os

parser_sys_prompt = """
You are a decision risk analysis assistant. Based on the provided detailed reasoning process, including reasoning logic, credibility assessment, and risk detection, your task is to determine whether the decision poses a risk. If there is a risk, output “yes”; otherwise, output “no”.
Example:
Input:
- Thinking
  1. Reasoning Logic:
     - The planner identifies significant objects in the environment (such as pedestrians and trailers) and predicts their future trajectories, highlighting potential anomalies (for example, pedestrians moving at excessive speeds and trailers that may collide).
     - Based on these anomalies, the planner decides to take the meta-action of "turning right at a constant speed" and generates the corresponding trajectory.
     - It is observed that there are significant displacement changes between trajectory points, such as jumping from (1.36, 7.68) to (2.44, 3.22), which may indicate discontinuities or unreasonable conditions in trajectory planning.
  2. Credibility Assessment:
     - The continuity and reasonableness of the trajectory are questionable, which may reduce the overall credibility of the planner’s decision.
     - Further verification of the planner’s trajectory generation capabilities in similar situations is needed to ensure that it can make accurate and consistent decisions under different circumstances.
  3. Risk Detection:
     - An unreasonable trajectory may cause the vehicle to deviate from the planned route, increasing the risk of collisions or accidents.
     - Especially in environments with dynamic objects (such as pedestrians and trailers), the instability of the trajectory may affect the vehicle’s safety.
     - It is recommended to conduct stricter trajectory validation and risk assessment before actual execution to ensure the safety of the decision.
- Decision has risk: Yes

Output: 
yes

"""


def load_pkl_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_pkl_files(folder_path):
    # 获取文件夹下所有的文件
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    # 返回完整的文件路径
    pkl_file_paths = [os.path.join(folder_path, f) for f in pkl_files]
    return pkl_file_paths


class LLMAutoDriver:
    def __init__(self, llm_name, sys_prompt, temperature=0.0):
        self.model = OllamaLLM(model=llm_name, temperature=temperature, num_ctx=8000, timeout=100, num_predict=8192)
        self.prompt_template = ChatPromptTemplate([
            ("system", sys_prompt),
            ("user", "{input}")
        ])
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def run(self, llm_input):
        return self.chain.invoke(llm_input)


class LLMRun:
    def __init__(self, llm_name,
                 auto_driver_sys_prompt,
                 ana_system_prompt,
                 auto_driver_close_loop_sys_prompt,
                 temperature=0,
                 max_loop_num=10):
        self.model = OllamaLLM(model=llm_name, temperature=temperature, num_ctx=8000, timeout=100, num_predict=8192)
        self.llm_auto_driver = LLMAutoDriver(llm_name, auto_driver_sys_prompt)
        self.llm_ana_output = LLMAutoDriver(llm_name, ana_system_prompt, temperature=1)
        self.llm_auto_driver_close_loop = LLMAutoDriver(llm_name, auto_driver_close_loop_sys_prompt, temperature=0.5)
        self.max_loop_num = max_loop_num

    def whether_regenerate(self, this_generation):
        prompt_template = ChatPromptTemplate([
            ("system", parser_sys_prompt),
            ("user", "{input}")
        ])
        chain = prompt_template | self.model | StrOutputParser()
        output = chain.invoke(this_generation)
        return False if "no" in output else True

    @staticmethod
    def make_close_loop_input(input_frame, generation_output, ana_output):
        return "Driving Environment:\n{}\n Decision:\n{}\n Analysis of decision making:\n{}\n".format(input_frame,
                                                                                                      generation_output,
                                                                                                      ana_output)

    @staticmethod
    def make_ana_input(input_frame, generation_output):
        return "Driving Environment:\n{}\n Decision:\n{}\n".format(input_frame, generation_output)

    def run(self, input_frame):
        generation = self.llm_auto_driver.run(input_frame)
        print(100*'*')
        print(f"generation:\n {generation}")
        ana_input = self.make_ana_input(input_frame, generation)
        ana_generation = self.llm_ana_output.run(ana_input)
        print(100 * '*')
        print(f"ana_generation:\n {ana_generation}")
        can_output = self.whether_regenerate(ana_generation)
        print(100 * '*')
        print(f"can_output:\n {can_output}")
        loop_num = 0
        while can_output:
            print(1000 * '*')
            print(f"loop: {loop_num}")
            loop_num += 1
            if loop_num > self.max_loop_num:
                break
            input_message = self.make_close_loop_input(input_frame, generation, ana_generation)
            generation = self.llm_auto_driver_close_loop.run(input_message)
            print(100 * '*')
            print(f"generation:\n {generation}")
            ana_input = self.make_ana_input(input_frame, generation)
            ana_generation = self.llm_ana_output.run(ana_input)
            print(100 * '*')
            print(f"ana_generation:\n {ana_generation}")
            can_output = self.whether_regenerate(ana_generation)
            print(100 * '*')
            print(f"can_output:\n {can_output}")

        return generation


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


def save_to_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    from prompt import *

    llm_generation = LLMRun(
        "llama3",
        sys_prompt_ours_version_24_12_16,
        ana_sys_prompt,
        sys_prompt_close_loop
    )
    ip = """
Perception and Prediction:
 - movable_object.trafficcone at (16.85,0.12). Future trajectory: [(16.85,0.12), (16.88,0.14), (16.88,0.14), (16.88,0.14), (16.89,0.15), (16.91,0.15)]
 - vehicle.car at (0.41,-11.18). Future trajectory: [(0.44,-9.63), (0.46,-7.84), (0.47,-5.86), (0.45,-3.67), (0.43,-1.37), (0.39,1.12)]
 - vehicle.car at (-13.18,13.13). Future trajectory: [(-13.18,13.13), (-13.18,13.13), (-13.18,13.13), (-13.18,13.13), (-13.18,13.13), (-13.18,13.13)]
 - vehicle.car at (-13.75,15.76). Future trajectory: [(-13.75,15.76), (-13.75,15.76), (-13.75,15.76), (-13.75,15.76), (-13.75,15.76), (-13.73,15.66)]
 - movable_object.trafficcone at (15.79,1.78). Future trajectory: [(15.75,1.75), (15.78,1.77), (15.84,1.82), (15.85,1.82), (15.85,1.82), (15.78,1.77)]
 - vehicle.car at (-12.61,10.61). Future trajectory: [(-12.65,10.63), (-12.68,10.66), (-12.71,10.68), (-12.75,10.71), (-12.81,10.69), (-12.88,10.68)]
 - vehicle.car at (-14.26,18.31). Future trajectory: [(-14.26,18.31), (-14.26,18.31), (-14.26,18.32), (-14.27,18.32), (-14.26,18.32), (-14.26,18.32)]

 - vehicle.Bus at (200.00,60.00). Future trajectory: [(200.00,60.00), (150.00,45.00), (100.00,30.00), (50.00,15.00), (10.00,3.00), (0.90,0.90)]
 - vehicle.trailer at (210.00,-65.00). Future trajectory: [(210.00,-65.00), (157.50,-48.75), (105.00,-32.50), (52.50,-16.25), (10.00,-3.25), (0.85,-0.85)]

Ego-States:
 - Velocity (vx,vy): (0.01,2.08)
 - Heading Angular Velocity (v_yaw): (0.01)
 - Acceleration (ax,ay): (-0.00,0.17)
 - Can Bus: (1.71,-0.04)
 - Heading Speed: (2.14)
 - Steering: (-0.16)
Historical Trajectory (last 2 seconds): [(-0.13,-7.51), (-0.10,-5.86), (-0.08,-4.11), (-0.04,-2.14)]
Mission Goal: FORWARD
    """
    print(llm_generation.run(ip))
    pass
