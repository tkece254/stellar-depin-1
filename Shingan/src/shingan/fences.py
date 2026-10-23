from __future__ import annotations

from dataclasses import dataclass

FENCE = "```"


@dataclass(frozen=True)
class Block:
    language: str
    body: str


def fenced_blocks(text: str) -> list[Block]:
    blocks = []
    language: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if language is None:
            if stripped.startswith(FENCE):
                language = stripped[len(FENCE):].strip().lower()
                lines = []
        elif stripped == FENCE:
            blocks.append(Block(language, "\n".join(lines) + "\n"))
            language = None
        else:
            lines.append(line)
    return blocks


def last_block(text: str, languages: frozenset[str]) -> str | None:
    matches = [b.body for b in fenced_blocks(text) if b.language in languages]
    return matches[-1] if matches else None
