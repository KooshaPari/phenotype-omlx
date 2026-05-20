# SPDX-License-Identifier: Apache-2.0
"""Accuracy evaluation benchmarks for LLMs.

Provides MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluators with deterministic sampling for fair model comparison.
"""

from .arc import ARCChallengeBenchmark
from .base import BaseBenchmark, BenchmarkResult, QuestionResult
from .cmmlu import CMMLUBenchmark
from .gsm8k import GSM8KBenchmark
from .hellaswag import HellaSwagBenchmark
from .humaneval import HumanEvalBenchmark
from .jmmlu import JMMLUBenchmark
from .kmmlu import KMMLUBenchmark
from .livecodebench import LiveCodeBenchBenchmark
from .mbpp import MBPPBenchmark
from .terminal_bench import TerminalBenchBenchmark
from .swe_bench import SWEBenchBenchmark
from .registry import BenchmarkPackage, BenchmarkRegistry
from .inspect_bridge import InspectBridge, InspectSample
from .dsl import BenchmarkDSL, BenchmarkStep, BenchmarkSuite, STANDARD_SUITES, standard_suite
from .mmlu import MMLUBenchmark
from .truthfulqa import TruthfulQABenchmark
from .winogrande import WinograndeBenchmark

BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    "mmlu": MMLUBenchmark,
    "kmmlu": KMMLUBenchmark,
    "cmmlu": CMMLUBenchmark,
    "jmmlu": JMMLUBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "arc_challenge": ARCChallengeBenchmark,
    "winogrande": WinograndeBenchmark,
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
    "terminal_bench": TerminalBenchBenchmark,
    "swe_bench": SWEBenchBenchmark,
}

__all__ = [
    "BENCHMARKS",
    "BaseBenchmark",
    "BenchmarkResult",
    "QuestionResult",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "TruthfulQABenchmark",
    "ARCChallengeBenchmark",
    "WinograndeBenchmark",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "LiveCodeBenchBenchmark",
    "BenchmarkRegistry",
    "BenchmarkPackage",
    "InspectSample",
    "InspectBridge",
    "standard_suite",
    "STANDARD_SUITES",
    "BenchmarkSuite",
    "BenchmarkStep",
    "BenchmarkDSL",
    "TerminalBenchBenchmark",
    "SWEBenchBenchmark",
]
