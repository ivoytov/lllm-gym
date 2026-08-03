"""VERL reward for the structured revenue prediction experiment."""

from __future__ import annotations

import json
import math


def final_answer(text: str) -> str:
    """Score only text after Qwen's final closing thinking tag."""
    if "<think>" in text:
        _, closing_tag, answer = text.rpartition("</think>")
        return answer.strip() if closing_tag else ""
    return text.strip()


def structured_prediction(text: str) -> float | None:
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
    value = float(prediction)
    return value if math.isfinite(value) else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return [-100, 0] for valid JSON and -200 for invalid syntax/schema."""
    answer = final_answer(solution_str)
    prediction = structured_prediction(answer)
    if prediction is None:
        reward = -200.0
    else:
        reward = -min(abs(prediction - float(ground_truth)), 100.0)
    print(f"[revenue_reward] answer={answer[:120]!r} reward={reward:.1f}")
    return reward
