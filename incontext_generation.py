import os.path
import re
import pickle
import json
import ast
import numpy as np
import argparse
from prompt_message import (
    system_message,
    example_message
)
from util import LLMAutoDriver, LLMRun
from tqdm import tqdm
from prompt import *

prompt_mapping_dict = {
    "gpt-driver": f"{system_message} \n{example_message}",
    "ours": sys_prompt_ours_version_24_12_16
}


class TrajectoryGenerator:
    def __init__(self, args):
        """
        Initialize the TrajectoryGenerator with command-line arguments.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.data_path = args.data_path
        self.output_name = args.name
        self.model_name = args.model

        if args.parser_type == "without_output_ana":
            self.llm_generate = LLMAutoDriver(self.model_name, prompt_mapping_dict[args.method_type])
        elif args.parser_type == "with_output_ana":
            self.llm_generate = LLMRun(
                self.model_name,
                prompt_mapping_dict[args.method_type],
                ana_sys_prompt,
                sys_prompt_close_loop
            )

        # Define file names based on output_name
        self.saved_traj_name = f"outputs/pkl/{self.output_name}.pkl"
        self.saved_text_name = f"outputs/pkl/{self.output_name}_text.pkl"
        self.temp_text_name = f"outputs/pkl/{self.output_name}_temp.jsonl"

        # Load data and split information
        self.dataset = self.load_pickle(os.path.join("data/our_dataset/basic_dataset", self.data_path))

        # Initialize dictionaries to store results
        self.text_dict = {}
        self.traj_dict = {}

        # List to store invalid tokens
        self.invalid_tokens = []

        # Tokens to exclude from testing (if any)
        self.untest_tokens = []

        # Mapping for data preprocessing functions
        self.base_dataset_generate = dict()

    @staticmethod
    def load_pickle(file_path):
        """Load a pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_json(file_path):
        """Load a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_pickle(self, data, file_path):
        """Save data to a pickle file."""
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def append_jsonl(self, data, file_path):
        """Append a dictionary as a JSON line to a file."""
        with open(file_path, "a+") as f:
            f.write(json.dumps(data) + '\n')

    def process_tokens(self):
        """
        Process each token in the test set based on the provided arguments.
        """
        for token, data_info in tqdm(self.dataset.items(), total=len(self.dataset)):
            # Skip tokens not in untest_tokens if untest_tokens is specified
            if self.untest_tokens and token not in self.untest_tokens:
                continue

            user_message = data_info['input']
            assistant_message = data_info['ground_truth']

            # Generate completion using LLMAutoDriver
            completion = self.llm_generate.run(user_message)
            result = completion

            output_dict = {
                "token": token,
                "GPT": result,
                "GT": assistant_message,
            }

            # Store GPT result
            self.text_dict[token] = result

            # Regex pattern to extract trajectory
            pattern = r"\[\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)(?:,\s*\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)){5}\s*\]"
            match_result = re.search(pattern, result, re.DOTALL)
            match_gt = re.search(pattern, assistant_message, re.DOTALL)

            # Append the result to the temporary JSONL file
            self.append_jsonl(output_dict, self.temp_text_name)
            # print(100 * "*")

            if match_result:
                traj = match_result.group()
                traj = ast.literal_eval(traj)
                traj = np.array(traj)
                self.traj_dict[token] = traj
                # print(f"GT:         {match_gt.group()}")
                # print(f"generation: {match_result.group()}")
            else:
                # print(f"Invalid token format: {token}")
                self.invalid_tokens.append(token)
                continue

            # If untest_tokens is specified, update the saved trajectory dictionary
            if self.untest_tokens:
                exist_dict = self.load_pickle(self.saved_traj_name)
                exist_dict.update(self.traj_dict)
                self.save_pickle(exist_dict, self.saved_traj_name)

    def save_results(self):
        """
        Save the collected text and trajectory dictionaries to pickle files.
        """
        if not self.untest_tokens:
            self.save_pickle(self.text_dict, self.saved_text_name)
            self.save_pickle(self.traj_dict, self.saved_traj_name)

    def display_invalid_tokens(self):
        """Print all invalid tokens encountered during processing."""
        if self.invalid_tokens:
            print("#### Invalid Tokens ####")
            for token in self.invalid_tokens:
                print(token)

    def run(self):
        """Execute the token processing workflow."""
        self.process_tokens()
        self.display_invalid_tokens()
        self.save_results()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Trajectory Generator")
    parser.add_argument('-d', '--data_path', type=str, required=True, help='the path of data')
    parser.add_argument('-n', '--name', type=str, required=True, help='Output name for saving results')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    parser.add_argument('-t', '--method_type', type=str, default="gpt-driver", help='method type')
    parser.add_argument('-p', '--parser_type', type=str, default="with_output_ana", help='whether have parser')
    return parser.parse_args()


def main():
    """Main function to execute the TrajectoryGenerator."""
    args = parse_arguments()
    print(f"Arguments: {args}")

    generator = TrajectoryGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
