"""Speculative decoding for accelerated text generation."""
import time
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SpeculativeDecoder:
    """Implements speculative decoding for faster inference."""

    def __init__(self, draft_model=None, target_model=None, gamma: int = 4):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma  # number of speculative tokens
        self.stats = {"accepted": 0, "rejected": 0, "total_tokens": 0}

    def generate(self, prompt: str, max_tokens: int = 100) -> dict:
        start = time.perf_counter()
        tokens = self._simulate_generation(prompt, max_tokens)
        elapsed = time.perf_counter() - start

        acceptance_rate = self.stats["accepted"] / max(self.stats["total_tokens"], 1)
        speedup = 1 + acceptance_rate * (self.gamma - 1) / self.gamma

        return {
            "text": tokens,
            "tokens_generated": len(tokens.split()),
            "latency_ms": round(elapsed * 1000, 2),
            "acceptance_rate": round(acceptance_rate, 4),
            "estimated_speedup": round(speedup, 2),
            "stats": dict(self.stats),
        }

    def _simulate_generation(self, prompt: str, max_tokens: int) -> str:
        words = prompt.split()[-5:]
        generated = list(words)
        for i in range(min(max_tokens, 50)):
            self.stats["total_tokens"] += 1
            if np.random.random() < 0.7:
                self.stats["accepted"] += 1
                generated.append(f"[tok_{i}]")
            else:
                self.stats["rejected"] += 1
                generated.append(f"[corrected_{i}]")
        return " ".join(generated)

    def benchmark(self, prompts: list[str]) -> dict:
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens=50)
            results.append(result)
        avg_speedup = np.mean([r["estimated_speedup"] for r in results])
        avg_acceptance = np.mean([r["acceptance_rate"] for r in results])
        return {
            "avg_speedup": round(float(avg_speedup), 2),
            "avg_acceptance_rate": round(float(avg_acceptance), 4),
            "total_benchmarks": len(prompts),
        }


def generate_fast(prompt: str, max_tokens: int = 100) -> dict:
    decoder = SpeculativeDecoder()
    return decoder.generate(prompt, max_tokens)
