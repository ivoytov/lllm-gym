"""Create the small SFT and GRPO datasets used by the VERL LoRA PoC."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


TARGET = 111.2
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PROMPT = """What is Apple's (AAPL US) quarterly revenue in the quarter ended March 28, 2026, in billions of USD?
Return JSON only, with exactly this structure:
{"revenue":{"prediction":<number>}}
`prediction` must be a bare JSON number: do not include units or explanatory text.
"""


def json_answer(prediction: float) -> str:
    payload = json.dumps({"revenue": {"prediction": prediction}}, separators=(",", ":"))
    # GPT-OSS gets its reasoning/final channels from the Harmony chat template.
    # Keep the supervised assistant content as the final JSON payload itself.
    return payload


def make_sft_rows() -> list[dict]:
    rng = random.Random(42)
    predictions: list[float] = []
    while len(predictions) < 192:
        prediction = rng.randrange(-1000, 3001) / 10
        if prediction != TARGET:
            predictions.append(prediction)

    rows = []
    for index, prediction in enumerate(predictions):
        if index % 2:
            prompt = (
                "Format the supplied revenue prediction as JSON only, with exactly this structure:\n"
                '{"revenue":{"prediction":<number>}}\n'
                "`prediction` must be a bare JSON number: do not include units or explanatory text.\n"
                f"Prediction: {prediction}\n"
            )
        else:
            prompt = PROMPT
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json_answer(prediction)},
                ],
                "enable_thinking": True,
            }
        )
    return rows


def make_rl_rows(count: int = 200) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": PROMPT}],
            "reward_model": {"ground_truth": str(TARGET)},
            "data_source": "revenue_json",
        }
        for _ in range(count)
    ]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(make_sft_rows()).to_parquet(DATA_DIR / "sft.parquet", index=False)
    rl_rows = make_rl_rows()
    pd.DataFrame(rl_rows).to_parquet(DATA_DIR / "grpo_train.parquet", index=False)
    pd.DataFrame(rl_rows[:16]).to_parquet(DATA_DIR / "grpo_val.parquet", index=False)
    print(f"Wrote {DATA_DIR / 'sft.parquet'} and GRPO train/validation parquet files")


if __name__ == "__main__":
    main()
