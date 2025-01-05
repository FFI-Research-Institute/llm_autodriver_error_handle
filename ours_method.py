from util import LLMAutoDriver, load_pkl_file
import argparse
from incontext_generation import parse_arguments


class OurTrajectoryGenerator:
    def __init__(self, args):
        self.llm_name = args.model
        self.data_path = args.data_path
        self.output_name = args.name
        self.llm_generator = LLMAutoDriver()


def main():
    """Main function to execute the TrajectoryGenerator."""
    args = parse_arguments()
    print(f"Arguments: {args}")

    generator = OurTrajectoryGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
