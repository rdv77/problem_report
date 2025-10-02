from __future__ import annotations

import json
from typing import Dict, List

from src.models.openai_client import chat_complete_ex


def _safe_json_loads(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        # strip markdown fences if present
        s = s.strip("`")
        # try to remove possible language hint like json
        parts = s.split("\n", 1)
        if len(parts) == 2 and parts[0].lower() in {"json", "javascript", "js"}:
            s = parts[1]
    return json.loads(s)


def translate_problems(
    cfg,
    problems_ru: List[str],
    langs: List[str],
) -> tuple[Dict[str, Dict[str, str]], dict]:
    """Translate a list of RU problems into specified languages.

    Returns mapping: ru_label -> { lang: translation }
    """

    # Build a single prompt to translate the entire list per language
    numbered = "\n".join([f"{i}. {name}" for i, name in enumerate(problems_ru, 1)])
    languages_str = ", ".join(langs)
    sys = (
        "Ты профессиональный переводчик. Переводи краткие словосочетания проблем строго по смыслу, без добавления новых слов. "
        "Верни только валидный JSON без комментариев."
    )
    usr = f"""
Исходные формулировки (RU), по одной на строку, с номерами:
{numbered}

Языки перевода (ISO 639-1): {languages_str}

Требования:
- Для каждого языка верни список переводов в том же порядке, что и входной список.
- Формат ответа (JSON):
{{
  "translations": {{
    "<lang1>": ["...", "..."],
    "<lang2>": ["...", "..."]
  }}
}}
"""

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr.strip()}]
    resp_text, usage = chat_complete_ex(cfg.models.classify, messages, temperature=0.0, max_tokens=2000)
    data = _safe_json_loads(resp_text)
    translations = data.get("translations", {})

    # Build mapping ru -> {lang: translation}
    result: Dict[str, Dict[str, str]] = {ru: {} for ru in problems_ru}
    for lang in langs:
        arr = translations.get(lang, [])
        for idx, ru in enumerate(problems_ru):
            if idx < len(arr):
                result[ru][lang] = str(arr[idx]).strip()
    return result, usage


