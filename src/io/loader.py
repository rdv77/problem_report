from __future__ import annotations

from pathlib import Path
import pandas as pd


PROBLEM_COL_CANDIDATES = ["проблема", "problem", "Проблема"]
PROBLEM_RU_COL_CANDIDATES = [
    "problem_russ",
    "problem_rus",  # common variant with single 's'
    "problem_ru",
    "проблема_ru",
    "проблема_rus",
    "проблема_рус",
    "проблема_перевод",
    "проблема (ru)",
    "перевод_проблемы",
]
MESSAGE_COL_CANDIDATES = ["Сообщение", "Сообщения", "сообщение", "text", "Текст"]


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_dataset(input_path: str | Path, require_problem: bool = True) -> tuple[pd.DataFrame, dict]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    # Try ; then ,
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, sep=",", encoding="utf-8-sig", on_bad_lines="skip")

    if "Заголовок" in df.columns:
        df = df.drop_duplicates(subset=["Заголовок"]).reset_index(drop=True)

    problem_ru_col = _detect_column(df, PROBLEM_RU_COL_CANDIDATES)
    problem_col = _detect_column(df, PROBLEM_COL_CANDIDATES)
    # Fallback: if base problem column is absent, but RU translation exists, use it as the problem column
    if problem_col is None and problem_ru_col is not None:
        problem_col = problem_ru_col
    if problem_col is None and require_problem:
        raise ValueError("Не найдена колонка с типом проблемы ('проблема'/'problem').")
    message_col = _detect_column(df, MESSAGE_COL_CANDIDATES)
    if message_col is None:
        raise ValueError("Не найдена колонка с текстами сообщений ('Сообщение'/'Сообщения').")

    link_col = "Ссылка" if "Ссылка" in df.columns else None

    cols = {
        "problem": problem_col,
        "problem_ru": problem_ru_col,
        "message": message_col,
        "link": link_col,
    }
    return df, cols


