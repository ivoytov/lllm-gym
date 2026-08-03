"""GPU launcher for the SFT → GRPO proof of concept."""

import argparse
import os

os.environ["GRPO_POC_USE_GPU"] = "1"

from grpo_poc import run_grpo, run_sft


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "sft", "grpo"), default="all")
    phase = parser.parse_args().phase
    if phase in ("all", "sft"):
        run_sft()
    if phase in ("all", "grpo"):
        run_grpo()
