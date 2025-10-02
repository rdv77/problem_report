from __future__ import annotations

import re
from typing import List, Dict


HEADING_PATTERNS = [
    re.compile(r"^Описание проблемы\s*:?$", re.IGNORECASE),
    re.compile(r"^Пример\s+\d+\s*:?$", re.IGNORECASE),
    re.compile(r"^Цитаты\s*:?$", re.IGNORECASE),
    re.compile(r"^Источники\s*:?$", re.IGNORECASE),
]


def _is_heading(line: str) -> bool:
    s = line.strip()
    return any(p.match(s) for p in HEADING_PATTERNS)


def _is_link(line: str) -> bool:
    s = line.strip()
    return s.startswith("http://") or s.startswith("https://")


def _is_quote(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # bullets as quotes
    if s.startswith("- ") or s.startswith("— ") or s.startswith("• "):
        return True
    # text enclosed in quotes
    if s[0] in {'"', '«', '“', '„', "'", '‚'}:
        return True
    # heuristic: contains translation in parentheses at end
    if re.search(r"\)\s*$", s) and ("(" in s):
        return True
    return False


def parse_editor_output(text: str) -> List[Dict[str, str | list]]:
    """Parse freeform edited section text into structured blocks for DOCX builder.

    Returns a list of blocks with types: subheading | paragraph | quote | sources.
    """
    blocks: List[Dict[str, str | list]] = []
    if not text or not text.strip():
        return blocks

    lines = [ln.rstrip() for ln in text.splitlines()]
    current_section = None  # quotes | sources | other

    sources_acc: list[str] = []

    def flush_sources():
        nonlocal sources_acc
        if sources_acc:
            blocks.append({"type": "sources", "items": sources_acc})
            sources_acc = []

    for line in lines:
        if not line.strip():
            continue
        if _is_heading(line):
            flush_sources()
            blocks.append({"type": "subheading", "text": line.strip()})
            if line.strip().lower().startswith("цитаты"):
                current_section = "quotes"
            elif line.strip().lower().startswith("источники"):
                current_section = "sources"
            else:
                current_section = None
            continue

        if current_section == "sources":
            if _is_link(line):
                sources_acc.append(line.strip())
            else:
                # If non-link text appears under Sources, drop it silently
                # and end the Sources section without adding explanations.
                flush_sources()
                current_section = None
            continue

        if current_section == "quotes" or _is_quote(line):
            flush_sources()
            # Normalize bullet prefix
            s = line.strip()
            if s.startswith("- ") or s.startswith("— ") or s.startswith("• "):
                s = s[2:].strip()
            blocks.append({"type": "quote", "text": s})
            continue

        # Fallback paragraph
        flush_sources()
        blocks.append({"type": "paragraph", "text": line})

    # End cleanup
    flush_sources()
    return blocks


