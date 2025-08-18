import re
import numpy as np
from typing import List, Tuple


def compute_confidence(token_confidences: List[float]) -> float:
    """Calculates the average confidence from a list of token probabilities."""
    return np.exp(np.mean(token_confidences)) if token_confidences else 0.0

def match_confidence(
    matches: List[re.Match],
    logprobs: object,
) -> List[Tuple[str, float]]:
    """Finds regex pattern occurrences and computes confidence scores."""
    if not logprobs or not logprobs.content or not matches:
        return []

    confidences_by_match = [[] for _ in matches]
    match_idx = 0
    current_pos = 0

    for item in logprobs.content:
        token_start = current_pos
        token_end = token_start + len(item.token)
        current_pos = token_end

        while match_idx < len(matches) and matches[match_idx].end() <= token_start:
            match_idx += 1

        # early stop
        if match_idx == len(matches):
            break

        current_match = matches[match_idx]
        if token_end > current_match.start():
            confidences_by_match[match_idx].append(item.logprob)

    return [
        (match.group(0), compute_confidence(scores))
        for match, scores in zip(matches, confidences_by_match)
        if scores
    ]
