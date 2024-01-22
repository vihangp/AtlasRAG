import json
from src.util import combine_jsonl_files
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input directory")
    args = parser.parse_args()

    input_dir = args.input_dir
    files = [f"{input_dir}/train-step-0-step-0-eval.jsonl", f"{input_dir}/trivia_train-step-0-step-0-eval.jsonl"]
    output_file = f"{input_dir}/train_nq_trivia.jsonl"
    combine_jsonl_files(files, output_file)

    files = [f"{input_dir}/nq_dev-step-0-step-0-eval.jsonl", f"{input_dir}/triviaqa_dev-step-0-step-0-eval.jsonl"]
    output_file = f"{input_dir}/dev_nq_trivia.jsonl"
    combine_jsonl_files(files, output_file)

    files = [f"{input_dir}/nq_test-step-0-step-0-eval.jsonl", f"{input_dir}/triviaqa_test-step-0-step-0-eval.jsonl"]
    output_file = f"{input_dir}/test_nq_trivia.jsonl"
    combine_jsonl_files(files, output_file)
 