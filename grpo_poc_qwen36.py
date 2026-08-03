"""B200 launcher for Qwen3.6-27B with thinking retained in completions."""

import argparse
import os

os.environ["GRPO_POC_USE_GPU"] = "1"
os.environ["GRPO_POC_BETA"] = "0.0"

import grpo_poc

grpo_poc.MODEL_ID = "Qwen/Qwen3.6-27B"
grpo_poc.SFT_OUTPUT_DIR = "./qwen36_structured_output_sft"
grpo_poc.GRPO_OUTPUT_DIR = "./qwen36_structured_output_grpo"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "sft", "grpo"), default="all")
    phase = parser.parse_args().phase
    if phase in ("all", "sft"):
        grpo_poc.run_sft()
    if phase in ("all", "grpo"):
        grpo_poc.run_grpo()
