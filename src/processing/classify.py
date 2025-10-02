from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.models.openai_client import chat_complete_ex


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _safe_json_loads(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        parts = s.split("\n", 1)
        if len(parts) == 2 and parts[0].lower() in {"json", "javascript", "js"}:
            s = parts[1]
    return json.loads(s)


@dataclass
class ClassificationParams:
    max_labels_per_text: int = 3


def build_classify_prompt(
    text: str,
    problems_ru: List[str],
    translations: Dict[str, Dict[str, str]] | None,
    langs: List[str],
    max_labels: int,
) -> List[dict]:
    problems_block = "\n".join([f"{i}. {name}" for i, name in enumerate(problems_ru, 1)])
    translations_block = ""
    if translations and langs:
        # Provide a compact per-lang list for robustness if text language differs
        for lang in langs:
            per_lang = [translations.get(ru, {}).get(lang, "") for ru in problems_ru]
            translations_block += f"\n[{lang}]\n" + "\n".join([f"{i}. {t}" for i, t in enumerate(per_lang, 1)])

    sys = (
        "Ты опытный аналитик. Твоя задача — присвоить тексту одну или несколько проблем из заданного списка. "
        "Выбирать можно ТОЛЬКО из списка (по номерам). Не выдумывай новых меток. Верни только валидный JSON."
    )
    usr = f"""
Текст:
{text}

Справочник проблем (RU):
{problems_block}
{translations_block}

Требования:
- Верни до {max_labels} меток (может быть 0) из списка, по убыванию релевантности.
- Строго JSON без комментариев, формат:
{{
  "problem_ids": [1, 3, 5]
}}
"""
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr.strip()}]


def classify_dataframe(
    cfg,
    df: pd.DataFrame,
    text_col: str,
    problems_ru: List[str],
    translations: Dict[str, Dict[str, str]] | None,
    langs: List[str],
    params: Optional[ClassificationParams] = None,
) -> tuple[pd.DataFrame, dict]:
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if params is None:
        params = ClassificationParams()

    candidates_ru: List[str] = []
    first_ru: List[str] = []

    # Per-language candidate and first-choice columns
    per_lang_candidates: Dict[str, List[str]] = {lang: [] for lang in langs}
    per_lang_first: Dict[str, List[str]] = {lang: [] for lang in langs}

    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "classify_cache.json"
    try:
        cache = json.loads(cache_file.read_text("utf-8")) if cache_file.exists() else {}
    except Exception:
        cache = {}

    problems_index_to_ru = {i + 1: ru for i, ru in enumerate(problems_ru)}

    for _, row in df.iterrows():
        text_raw = str(row[text_col]) if pd.notna(row[text_col]) else ""
        t = text_raw.strip()
        if not t:
            candidates_ru.append("")
            first_ru.append("")
            for lang in langs:
                per_lang_candidates[lang].append("")
                per_lang_first[lang].append("")
            continue

        key = _hash_text(t + "|" + "|".join(problems_ru))
        if key in cache:
            ids = cache[key]
        else:
            msgs = build_classify_prompt(t, problems_ru, translations, langs, params.max_labels_per_text)
            resp_text, usage = chat_complete_ex(cfg.models.classify, msgs, temperature=0.0, max_tokens=800)
            data = _safe_json_loads(resp_text)
            ids = data.get("problem_ids", [])
            # accumulate tokens
            usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
            usage_total["total_tokens"] += usage.get("total_tokens", 0)
            if not isinstance(ids, list):
                ids = []
            ids = [int(i) for i in ids if isinstance(i, (int, float))]
            ids = ids[: params.max_labels_per_text]
            cache[key] = ids

        ru_labels = [problems_index_to_ru.get(i, "") for i in ids if i in problems_index_to_ru]
        ru_labels = [s for s in ru_labels if s]

        candidates_ru.append("|".join(ru_labels))
        first_ru.append(ru_labels[0] if ru_labels else "")

        for lang in langs:
            lang_labels = [translations.get(ru, {}).get(lang, "") for ru in ru_labels]
            lang_labels = [s for s in lang_labels if s]
            per_lang_candidates[lang].append("|".join(lang_labels))
            per_lang_first[lang].append(lang_labels[0] if lang_labels else "")

    # Save cache
    try:
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass

    df_out = df.copy()
    df_out["problem_russ_candidates"] = candidates_ru
    df_out["problem_russ"] = first_ru
    for lang in langs:
        df_out[f"problem_{lang}_candidates"] = per_lang_candidates[lang]
        df_out[f"problem_{lang}"] = per_lang_first[lang]

    return df_out, usage_total


