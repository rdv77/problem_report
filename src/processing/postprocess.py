from __future__ import annotations

from typing import List

import pandas as pd


def add_wide_columns(
    df: pd.DataFrame,
    problems_ru: List[str],
    langs: List[str],
    ru_candidates_col: str = "problem_russ_candidates",
) -> pd.DataFrame:
    df_out = df.copy()

    # RU wide
    for ru in problems_ru:
        col = f"ru_{ru}"
        df_out[col] = df_out[ru_candidates_col].astype(str).apply(
            lambda s: str(ru) in s.split("|") if s else False
        )

    # per language wide columns (based on candidates too)
    for lang in langs:
        cand_col = f"problem_{lang}_candidates"
        # Gather known labels for this lang from existing translations in the dataset
        # We'll derive from translated first/unique values to avoid redundant columns
        unique_labels = sorted(
            set(
                l
                for v in df_out[cand_col].dropna().astype(str).tolist()
                for l in v.split("|")
                if l
            )
        )
        for lbl in unique_labels:
            col = f"{lang}_{lbl}"
            df_out[col] = df_out[cand_col].astype(str).apply(
                lambda s, label=lbl: label in s.split("|") if s else False
            )

    return df_out


