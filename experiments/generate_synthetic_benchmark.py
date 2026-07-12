#!/usr/bin/env python3
"""Generate controlled benchmark-style prompts for UCKV pilots."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


FILLER_SENTENCES = [
    "The maintenance note describes routine checks and ordinary status updates.",
    "The archive paragraph contains background information without the requested value.",
    "The operator log mentions queue lengths, weather, and unrelated scheduling details.",
    "The project memo records stable conditions and no urgent changes.",
    "The report section discusses labels, folders, and common administrative actions.",
    "The appendix repeats harmless context to increase the prompt length.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/prompts/synthetic_benchmark.jsonl")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mode", default="mixed", choices=["mixed", "retrieval"])
    parser.add_argument("--lengths", default="48,96,160")
    parser.add_argument("--retrieval-repeats", type=int, default=1)
    parser.add_argument(
        "--retrieval-answer-style",
        default="code",
        choices=["code", "phrase"],
        help="Use legacy short codes or multi-token phrases as retrieval values.",
    )
    return parser.parse_args()


def filler(words: int, rng: random.Random) -> str:
    chunks: List[str] = []
    while len(" ".join(chunks).split()) < words:
        chunks.append(rng.choice(FILLER_SENTENCES))
    return " ".join(" ".join(chunks).split()[:words]) + "."


def make_retrieval_prompt(
    prompt_id: str,
    task_type: str,
    key: str,
    answer: str,
    answer_position: str,
    context_words: int,
    rng: random.Random,
) -> Dict[str, object]:
    before_words = context_words // 2
    after_words = context_words - before_words
    fact = f"The requested key is {key}. The value for {key} is {answer}."
    if answer_position == "early":
        context = f"{fact} {filler(context_words, rng)}"
    elif answer_position == "middle":
        context = f"{filler(before_words, rng)} {fact} {filler(after_words, rng)}"
    elif answer_position == "late":
        context = f"{filler(context_words, rng)} {fact}"
    else:
        raise ValueError(answer_position)
    text = (
        "Read the document and answer with only the requested value.\n"
        f"Document: {context}\n"
        f"Question: What is the value for {key}?\n"
        "Answer:"
    )
    return {
        "id": prompt_id,
        "task_type": task_type,
        "answer_position": answer_position,
        "context_words": context_words,
        "answer": answer,
        "text": text,
    }


def make_arithmetic_prompt(prompt_id: str, a: int, b: int, c: int) -> Dict[str, object]:
    answer = str(a - b + c)
    text = (
        "Solve the arithmetic problem and answer with only the number.\n"
        f"Alice has {a} tokens. She spends {b} tokens and then receives {c} tokens. "
        "How many tokens does Alice have now?\n"
        "Answer:"
    )
    return {
        "id": prompt_id,
        "task_type": "arithmetic",
        "answer_position": "none",
        "context_words": 0,
        "answer": answer,
        "text": text,
    }


def make_summary_prompt(prompt_id: str, topic: str, answer: str, rng: random.Random) -> Dict[str, object]:
    text = (
        "Summarize the document in exactly three words.\n"
        f"Document: {filler(30, rng)} The central theme is {topic}. {filler(30, rng)}\n"
        "Summary:"
    )
    return {
        "id": prompt_id,
        "task_type": "summary_keyword",
        "answer_position": "middle",
        "context_words": 60,
        "answer": answer,
        "text": text,
    }


def parse_lengths(value: str) -> List[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive integers.")
    return lengths


def retrieval_key_answer(index: int, answer_style: str = "code") -> tuple[str, str]:
    prefixes = [
        "aurora",
        "borealis",
        "cygnus",
        "draco",
        "ember",
        "fable",
        "glacier",
        "harbor",
        "ion",
        "juniper",
        "kepler",
        "lumen",
    ]
    prefix = prefixes[index % len(prefixes)]
    key = f"{prefix} code {index:02d}"
    if answer_style == "phrase":
        colors = ["saffron", "indigo", "silver", "crimson", "emerald", "violet"]
        objects = ["bridge", "harbor", "compass", "lantern", "meadow", "summit"]
        answer = (
            f"{colors[index % len(colors)]} "
            f"{objects[(index * 5 + 1) % len(objects)]} "
            f"{(index * 37 + 1017) % 10000:04d}"
        )
    else:
        answer = f"{prefix[:1].upper()}{(index * 37 + 17) % 1000:03d}"
    return key, answer


def build_retrieval_examples(
    seed: int,
    lengths: List[int],
    repeats: int,
    answer_style: str = "code",
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    examples: List[Dict[str, object]] = []
    positions = ["early", "middle", "late"]
    idx = 0
    for context_words in lengths:
        for answer_position in positions:
            for repeat in range(repeats):
                key, answer = retrieval_key_answer(idx, answer_style)
                examples.append(
                    make_retrieval_prompt(
                        prompt_id=f"retrieval_{context_words}_{answer_position}_{repeat:02d}",
                        task_type="kv_retrieval",
                        key=key,
                        answer=answer,
                        answer_position=answer_position,
                        context_words=context_words,
                        rng=rng,
                    )
                )
                idx += 1
    return examples


def build_examples(
    seed: int,
    lengths: List[int],
    retrieval_repeats: int,
    mode: str,
    answer_style: str = "code",
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    examples = build_retrieval_examples(seed, lengths, retrieval_repeats, answer_style)
    if mode == "retrieval":
        return examples

    for i, (a, b, c) in enumerate([(12, 5, 3), (31, 9, 4), (44, 18, 7), (25, 6, 11)]):
        examples.append(make_arithmetic_prompt(f"arithmetic_{i:02d}", a, b, c))

    for i, (topic, answer) in enumerate(
        [
            ("calibrated cache compression", "calibrated cache compression"),
            ("memory efficient inference", "memory efficient inference"),
            ("risk aware decoding", "risk aware decoding"),
        ]
    ):
        examples.append(make_summary_prompt(f"summary_{i:02d}", topic, answer, rng))

    return examples


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_examples(
        args.seed,
        parse_lengths(args.lengths),
        args.retrieval_repeats,
        args.mode,
        args.retrieval_answer_style,
    )
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
