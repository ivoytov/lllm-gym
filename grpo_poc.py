"""Two-stage proof of concept: format SFT followed by numeric GRPO."""

import json
import math
import argparse
import os
import random

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-1.7B"
TARGET = 111.2
USE_GPU = os.environ.get("GRPO_POC_USE_GPU") == "1"
GRPO_BETA = float(os.environ.get("GRPO_POC_BETA", "0.0"))
# Transformers v5 uses `dtype`; without it TRL deliberately defaults to float32.
MODEL_INIT_KWARGS = {"dtype": "bfloat16"} if USE_GPU else None
SFT_OUTPUT_DIR = "./structured_output_sft"
GRPO_OUTPUT_DIR = "./structured_output_grpo"

PROMPT = """What is Apple's (AAPL US) quarterly revenue in the quarter ended March 28, 2026, in billions of USD?
Return JSON only, with exactly this structure:
{"revenue":{"prediction":<number>}}
`prediction` must be a bare JSON number: do not include units or explanatory text.
"""


def final_answer(text):
    """Discard Qwen thinking; score only the answer emitted after its closing tag."""
    if "<think>" in text:
        before, closing_tag, answer = text.rpartition("</think>")
        return answer.strip() if closing_tag else ""
    return text.strip()


def is_structured_prediction(text):
    """Return the prediction only for the exact requested JSON schema."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict) or set(payload) != {"revenue"}:
        return None
    revenue = payload["revenue"]
    if not isinstance(revenue, dict) or set(revenue) != {"prediction"}:
        return None
    prediction = revenue["prediction"]
    if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
        return None
    return float(prediction) if math.isfinite(prediction) else None


def revenue_reward(completions, **kwargs):
    """Penalize valid answers by distance (floored at -100); malformed output is -200."""
    rewards = []
    raw_texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    texts = [final_answer(text) for text in raw_texts]
    for text in texts:
        prediction = is_structured_prediction(text)
        if prediction is None:
            reward = -200.0
        else:
            # Exact target: 0. A prediction of 100 receives -11.2.
            reward = -min(abs(prediction - TARGET), 100.0)
        rewards.append(reward)
    print("SAMPLES:", [(text[:70], round(reward, 1)) for text, reward in zip(texts, rewards)])
    return rewards


def format_sft_dataset():
    """Teach JSON formatting by copying varied decimals, without training the finance answer."""
    rng = random.Random(42)
    # Tenths make the decimal-token pattern familiar, while omitting the GRPO target.
    format_only_predictions = []
    while len(format_only_predictions) < 192:
        prediction = rng.randrange(-1000, 3001) / 10
        if prediction != TARGET:
            format_only_predictions.append(prediction)
    completions = [
        json.dumps({"revenue": {"prediction": prediction}}, separators=(",", ":"))
        for prediction in format_only_predictions
    ]
    copy_prompts = [
        "Format the supplied revenue prediction as JSON only, with exactly this structure:\n"
        '{"revenue":{"prediction":<number>}}\n'
        "`prediction` must be a bare JSON number: do not include units or explanatory text.\n"
        f"Prediction: {prediction}\n"
        for prediction in format_only_predictions
    ]
    # Include the real GRPO prompt so financial wording such as "billions of USD"
    # does not pull the model back toward prose/unit-bearing answers.
    prompts = [PROMPT if index % 2 == 0 else copy_prompts[index]
               for index in range(len(format_only_predictions))]
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def run_sft():
    config = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        max_length=256,
        completion_only_loss=True,
        logging_steps=5,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="adafactor",
        report_to=[],
        seed=42,
        use_cpu=not USE_GPU,
        bf16=USE_GPU,
        model_init_kwargs=MODEL_INIT_KWARGS,
    )
    trainer = SFTTrainer(model=MODEL_ID, args=config, train_dataset=format_sft_dataset())
    trainer.train()
    trainer.save_model(SFT_OUTPUT_DIR)


def run_grpo():
    dataset = Dataset.from_dict({"prompt": [PROMPT] * 200})
    config = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        num_generations=4,
        gradient_accumulation_steps=8,
        max_completion_length=32,
        # Keep a small model's JSON generations coherent; the batch supplies exploration.
        temperature=1.0,
        # A B200 full-model run cannot afford a second, frozen reference-model copy.
        beta=GRPO_BETA,
        max_steps=100,
        logging_steps=1,
        gradient_checkpointing=True,
        optim="adafactor",
        report_to=[],
        seed=42,
        use_cpu=not USE_GPU,
        bf16=USE_GPU,
        model_init_kwargs=MODEL_INIT_KWARGS,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=1,
    )
    trainer = GRPOTrainer(
        model=SFT_OUTPUT_DIR,
        args=config,
        reward_funcs=revenue_reward,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "sft", "grpo"), default="all")
    phase = parser.parse_args().phase
    if phase in ("all", "sft"):
        run_sft()
    if phase in ("all", "grpo"):
        run_grpo()
