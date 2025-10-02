from __future__ import annotations

import pandas as pd
import re


def _normalize_label(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower()


def _merge_key_for_row(base_label: str, ru_label: str | None) -> str:
    """Map raw labels into merged categories per user rules.

    Priority of matching:
    1) RU label synonyms
    2) Base label (could be RU/EN depending on dataset)
    """
    ru_norm = _normalize_label(ru_label)
    base_norm = _normalize_label(base_label)

    # Group 7.1: ecology/climate/pollution/droughts
    ecology_synonyms = {
        "экологические проблемы",
        "экология",
        "загрязнение воздуха и воды",
        "загрязнение воздуха",
        "загрязнение воды",
        "изменение климата",
        "климатические изменения",
        "засуха",
        "засухи",
    }
    if ru_norm in ecology_synonyms or base_norm in ecology_synonyms:
        return "Экология, климат и загрязнение"

    # Group 7.2: power outages & electricity cost
    power_synonyms = {
        "сбои электроснабжения",
        "перебои с электричеством",
        "стоимость электроэнергии",
        "проблемы с электроэнергией (сбои, высокая стоимость, дефицит)",
        "электроэнергия",
    }
    if ru_norm in power_synonyms or base_norm in power_synonyms:
        return "Электроэнергия: сбои и высокая стоимость"

    # Group 7.3: housing conditions & high cost/rent
    housing_synonyms = {
        "жилищные условия",
        "высокие цены на жильё и аренду",
        "плохие жилищные условия и высокая стоимость жилья",
    }
    if ru_norm in housing_synonyms or base_norm in housing_synonyms:
        return "Жильё: условия и высокая стоимость"

    # Group 7.4: inflation & high prices for food/fuel
    inflation_synonyms = {
        "инфляция",
        "инфляция и рост цен",
        "высокие цены на продукты",
        "высокие цены на топливо",
    }
    if ru_norm in inflation_synonyms or base_norm in inflation_synonyms:
        return "Инфляция и рост цен (продукты, топливо)"

    # Fallback: original base label
    return base_label


def assemble_problems(df: pd.DataFrame, cols: dict, max_messages: int, max_chars: int) -> tuple[list[str], dict]:
    problem_col = cols["problem"]
    message_col = cols["message"]
    problem_ru_col = cols.get("problem_ru")
    link_col = cols.get("link")

    # Build merged label per row
    ru_series = df[problem_ru_col] if (problem_ru_col and problem_ru_col in df.columns) else None
    merged_labels = df[problem_col].astype(str).copy()
    if ru_series is not None:
        merged_labels = [
            _merge_key_for_row(base_label=str(b), ru_label=str(r))
            for b, r in zip(df[problem_col].astype(str).tolist(), ru_series.astype(str).tolist())
        ]
    else:
        merged_labels = [
            _merge_key_for_row(base_label=str(b), ru_label=None)
            for b in df[problem_col].astype(str).tolist()
        ]

    counts = pd.Series(merged_labels).value_counts(dropna=True)
    # Exclude pseudo-label 'нет'
    if 'нет' in counts.index:
        counts = counts[counts.index != 'нет']
    problems_ordered = list(counts.index)

    def normalize(msg: object) -> str | None:
        if pd.isna(msg):
            return None
        s = str(msg).strip()
        if len(s) > max_chars:
            s = s[:max_chars] + "..."
        return s if s else None

    groups: dict = {}
    for p in problems_ordered:
        # Rebuild subset based on merged labels
        subset_mask = [ml == p for ml in merged_labels]
        subset = df[subset_mask].copy()
        subset = subset[~subset[message_col].isna()]
        if subset.empty:
            continue
        messages = (
            subset[message_col].map(normalize).dropna().tolist()[:max_messages]
        )
        if not messages:
            continue
        # Pick the first non-empty RU translation within the group (not just the first row)
        if problem_ru_col and problem_ru_col in subset.columns:
            ru_series = subset[problem_ru_col].dropna()
            problem_ru = str(ru_series.iloc[0]) if not ru_series.empty else None
        else:
            problem_ru = None
        links = []
        if link_col:
            links = (
                subset[link_col].dropna().astype(str).drop_duplicates().head(3).tolist()
            )
        # collect labels by language if present in dataset (e.g., problem_en)
        labels_by_lang: dict = {}
        for col in df.columns:
            if col.startswith("problem_") and col.endswith("_candidates") is False:
                # pick first non-null value for this label within group
                if col in subset.columns:
                    val = subset[col].dropna().astype(str)
                    labels_by_lang[col.split("_", 1)[1]] = str(val.iloc[0]) if not val.empty else None

        groups[p] = {
            "problem_ru": problem_ru,
            "messages": messages,
            "links": links,
            "count": int(counts[p]),
            "labels_by_lang": labels_by_lang,
        }

    return problems_ordered, groups


def build_distribution(df: pd.DataFrame, cols: dict) -> list[tuple[str, int]]:
    problem_col = cols["problem"]
    problem_ru_col = cols.get("problem_ru")
    # Compute merged labels for distribution as well
    ru_series_full = df[problem_ru_col] if (problem_ru_col and problem_ru_col in df.columns) else None
    merged_labels = df[problem_col].astype(str).copy()
    if ru_series_full is not None:
        merged_labels = [
            _merge_key_for_row(base_label=str(b), ru_label=str(r))
            for b, r in zip(df[problem_col].astype(str).tolist(), ru_series_full.astype(str).tolist())
        ]
    else:
        merged_labels = [
            _merge_key_for_row(base_label=str(b), ru_label=None)
            for b in df[problem_col].astype(str).tolist()
        ]

    series = pd.Series(merged_labels).value_counts(dropna=True)
    if 'нет' in series.index:
        series = series[series.index != 'нет']
    rows: list[tuple[str, int]] = []
    for merged_label, count in series.items():
        display = str(merged_label)
        rows.append((display, int(count)))
    return rows



