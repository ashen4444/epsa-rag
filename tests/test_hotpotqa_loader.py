from __future__ import annotations

import json
from pathlib import Path

import pytest

from epsa_rag.datasets.hotpotqa_loader import (
    HotPotQAFormatError,
    load_hotpotqa_examples,
)


def test_hotpotqa_loader_loads_examples_from_json_file(tmp_path: Path) -> None:
    input_path = tmp_path / "hotpot_sample.json"

    raw_data = [
        {
            "_id": "hotpot_train_000001",
            "question": "Where was Marie Curie born?",
            "answer": "Warsaw",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": [["Marie Curie", 0]],
            "context": [
                [
                    "Marie Curie",
                    [
                        "Marie Curie was born in Warsaw.",
                        "She later moved to France.",
                    ],
                ]
            ],
        }
    ]

    input_path.write_text(json.dumps(raw_data), encoding="utf-8")

    examples = load_hotpotqa_examples(input_path)

    assert len(examples) == 1
    assert examples[0].source_question_id == "hotpot_train_000001"
    assert examples[0].question == "Where was Marie Curie born?"
    assert examples[0].answer == "Warsaw"
    assert examples[0].question_type == "bridge"
    assert examples[0].level == "easy"
    assert examples[0].supporting_facts == [("Marie Curie", 0)]
    assert examples[0].context == [
        (
            "Marie Curie",
            [
                "Marie Curie was born in Warsaw.",
                "She later moved to France.",
            ],
        )
    ]


def test_hotpotqa_loader_respects_sample_size(tmp_path: Path) -> None:
    input_path = tmp_path / "hotpot_sample.json"

    raw_data = [
        {
            "_id": f"hotpot_train_{index:06d}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": [["Document", 0]],
            "context": [["Document", ["Sentence."]]],
        }
        for index in range(3)
    ]

    input_path.write_text(json.dumps(raw_data), encoding="utf-8")

    examples = load_hotpotqa_examples(input_path, sample_size=2)

    assert len(examples) == 2
    assert examples[0].source_question_id == "hotpot_train_000000"
    assert examples[1].source_question_id == "hotpot_train_000001"


def test_hotpotqa_loader_rejects_missing_required_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid_hotpot.json"

    raw_data = [
        {
            "_id": "hotpot_train_000001",
            "question": "Where was Marie Curie born?",
            "answer": "Warsaw",
            "context": [["Marie Curie", ["Marie Curie was born in Warsaw."]]],
        }
    ]

    input_path.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(HotPotQAFormatError, match="Missing required HotPotQA fields"):
        load_hotpotqa_examples(input_path)


def test_hotpotqa_loader_rejects_invalid_context_shape(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid_context.json"

    raw_data = [
        {
            "_id": "hotpot_train_000001",
            "question": "Where was Marie Curie born?",
            "answer": "Warsaw",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": [["Marie Curie", 0]],
            "context": [["Marie Curie"]],
        }
    ]

    input_path.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(HotPotQAFormatError, match="context\\[0\\] must be"):
        load_hotpotqa_examples(input_path)


def test_hotpotqa_loader_rejects_non_list_root(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid_root.json"

    input_path.write_text(json.dumps({"_id": "not_a_list"}), encoding="utf-8")

    with pytest.raises(HotPotQAFormatError, match="JSON list"):
        load_hotpotqa_examples(input_path)