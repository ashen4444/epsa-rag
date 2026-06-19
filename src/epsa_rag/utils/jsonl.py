from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
            written_count += 1

    return written_count


def read_jsonl(
    input_path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number} in {path}"
                ) from exc

    return records