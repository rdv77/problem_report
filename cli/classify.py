import argparse
from pathlib import Path
import os
import time

import pandas as pd
from dotenv import load_dotenv

from src.config.settings import load_config_from_env
from src.io.loader import load_dataset
from src.processing.translate import translate_problems
from src.processing.classify import classify_dataframe, ClassificationParams
from src.processing.postprocess import add_wide_columns


def main():
    parser = argparse.ArgumentParser(description="Classify problems in CSV and save *_classified.csv")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--problems-file", required=True, help="Path to RU problems list (one per line)")
    parser.add_argument("--langs", required=True, help="Comma-separated ISO 639-1 codes (1-2 languages). Example: en,sw")
    parser.add_argument("--output", required=False, help="Output CSV path (default: *_classified.csv next to input)")
    parser.add_argument("--wide", action="store_true", help="Add one-hot wide columns per problem/language")
    parser.add_argument("--max-labels", type=int, default=None, help="Max labels per text (default from config or 3)")

    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required. Set it in .env, ENV, or pass via environment.")

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs or len(langs) > 2:
        raise SystemExit("--langs must specify 1 or 2 ISO codes, e.g.: --langs en,sw")

    cfg = load_config_from_env()

    # Load dataset using existing detection logic
    # For classification we don't require an existing problem column
    df, cols = load_dataset(args.input, require_problem=False)
    text_col = cols["message"]

    # Load RU problems list
    ru_list = [line.strip() for line in Path(args.problems_file).read_text("utf-8").splitlines() if line.strip()]
    if not ru_list:
        raise SystemExit("Problems list is empty. Provide RU problems in a file, one per line.")

    # Translate problems
    print("[1/4] Перевод списка проблем...")
    t0 = time.time()
    translations, usage_translate = translate_problems(cfg, ru_list, langs)
    t1 = time.time()
    print(f"[1/4] Готово за {t1 - t0:.2f} c | токены: {usage_translate}")

    # Classify
    params = ClassificationParams(max_labels_per_text=args.max_labels or 3)
    print("[2/4] Классификация текстов...")
    t2 = time.time()
    df_cls, usage_classify = classify_dataframe(cfg, df, text_col, ru_list, translations, langs, params)
    t3 = time.time()
    print(f"[2/4] Готово за {t3 - t2:.2f} c | токены: {usage_classify}")

    # Optionally add wide columns
    if args.wide:
        print("[3/4] Формирование широкого вида (one-hot)...")
        t4 = time.time()
        df_cls = add_wide_columns(df_cls, ru_list, langs)
        t5 = time.time()
        print(f"[3/4] Готово за {t5 - t4:.2f} c")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        in_path = Path(args.input)
        out_path = in_path.with_name(in_path.stem + "_classified.csv")
    print("[4/4] Сохранение результата...")
    t6 = time.time()
    df_cls.to_csv(out_path, index=False, encoding="utf-8-sig", sep=";")
    t7 = time.time()
    print(f"[4/4] Готово за {t7 - t6:.2f} c | Файл: {out_path}")

    total_tokens = {
        "prompt_tokens": usage_translate.get("prompt_tokens", 0) + usage_classify.get("prompt_tokens", 0),
        "completion_tokens": usage_translate.get("completion_tokens", 0) + usage_classify.get("completion_tokens", 0),
        "total_tokens": usage_translate.get("total_tokens", 0) + usage_classify.get("total_tokens", 0),
    }
    print(f"ИТОГО токены: {total_tokens}")


if __name__ == "__main__":
    main()


