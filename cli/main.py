import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.settings import load_config_from_env
from src.io.loader import load_dataset
from src.processing.assemble_data import assemble_problems
from src.processing.generate import generate_section
from src.processing.edit import edit_section
from src.processing.segment import parse_editor_output
from src.report.docx_builder import build_document


def main():
    parser = argparse.ArgumentParser(description="Generate country-focused problems report (DOCX)")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--country", required=False, help="Country focus (overrides config/env)")
    parser.add_argument("--output-dir", required=False, help="Output directory (default=dir of input)")
    parser.add_argument("--on-missing-facts", required=False, choices=["skip", "search"], help="If no facts: skip or search")
    parser.add_argument("--auto-classify", action="store_true", help="Run classification before report generation if problem column missing")
    parser.add_argument("--langs", required=False, help="Languages for classification translations (1-2 ISO codes, e.g. en,sw)")
    parser.add_argument("--classify-max-labels", type=int, default=None, help="Max labels per text during auto-classify (e.g., 1)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel sections to generate/edit (default=1)")
    parser.add_argument("--api-key", required=False, help="OpenAI API key (overrides .env/ENV)")

    args = parser.parse_args()

    # Load .env first so OPENAI_API_KEY and others are available
    load_dotenv()
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    cfg = load_config_from_env(country=args.country)
    if args.on_missing_facts:
        cfg.on_missing_facts = args.on_missing_facts

    input_path = Path(args.input).resolve()
    assert input_path.exists(), f"Input file not found: {input_path}"

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    os.makedirs(output_dir, exist_ok=True)

    # Validate API key early
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required. Set it in .env, ENV, or pass --api-key")

    # Load dataset and detect columns
    print("[1/6] Загрузка датасета и определение колонок...")
    import time
    t0 = time.time()
    df, cols = load_dataset(input_path)
    t1 = time.time()
    print(f"[1/6] Готово за {t1 - t0:.2f} c | найдено: problem={cols.get('problem')}, problem_ru={cols.get('problem_ru')}, message={cols.get('message')}")

    # Optional pre-classify step to ensure RU problem column is present
    if args.auto_classify and (cols.get("problem_ru") is None):
        print("[2/6] Автоклассификация отсутствующих меток (RU)...")
        t2 = time.time()
        if not args.langs:
            raise SystemExit("--auto-classify requires --langs (1-2 ISO codes), e.g.: --langs en,sw")
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        if not langs or len(langs) > 2:
            raise SystemExit("--langs must specify 1 or 2 ISO codes, e.g.: --langs en,sw")

        # Expect problems list file via env var
        problems_file = os.getenv("PROBLEMS_RU_FILE")
        if not problems_file or not Path(problems_file).exists():
            raise SystemExit("Set PROBLEMS_RU_FILE env var pointing to data/problems_ru.txt with RU problems (one per line)")

        # Inline call of classification pipeline
        from src.processing.translate import translate_problems
        from src.processing.classify import classify_dataframe, ClassificationParams
        from src.processing.postprocess import add_wide_columns

        ru_list = [line.strip() for line in Path(problems_file).read_text("utf-8").splitlines() if line.strip()]
        translations, usage_tr = translate_problems(cfg, ru_list, langs)
        params = ClassificationParams(max_labels_per_text=(args.classify_max_labels or 3))
        df, usage_cl = classify_dataframe(cfg, df, cols["message"], ru_list, translations, langs, params)
        # refresh detected columns after classification
        from src.io.loader import _detect_column
        cols["problem_ru"] = _detect_column(df, ["problem_russ", "problem_ru"]) or "problem_russ"
        t3 = time.time()
        print(f"[2/6] Готово за {t3 - t2:.2f} c | токены перевод={usage_tr}, классификация={usage_cl}")

    # Assemble messages per problem
    problems_ordered, groups = assemble_problems(
        df,
        cols,
        cfg.max_messages_per_problem,
        cfg.max_chars_per_message,
    )
    print(f"Обнаружено разделов к генерации: {len(problems_ordered)}")

    # Generate sections (minimal path; no editor yet)
    sections = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def process_section(problem_key: str, group: dict):
        try:
            print(f"[3/6] Генерация раздела: {problem_key}...")
            tg0 = time.time()
            raw_text, u_gen = generate_section(cfg, cfg.country, problem_key, group)
            tg1 = time.time()
            print(f"[3/6] Генерация ок за {tg1 - tg0:.2f} c | токены {u_gen}")

            print(f"[4/6] Редактирование раздела: {problem_key}...")
            te0 = time.time()
            text, u_edit = edit_section(cfg, cfg.country, raw_text, group.get("links", []))
            te1 = time.time()
            print(f"[4/6] Редактирование ок за {te1 - te0:.2f} c | токены {u_edit}")

            ru_title = group.get('problem_ru')
            if ru_title and ru_title != problem_key:
                section_title = f"{ru_title} ({problem_key})"
            else:
                section_title = f"{problem_key}"
            blocks = parse_editor_output(text)
            if not blocks:
                blocks = [{"type": "paragraph", "text": text}]
            section = {"title": section_title, "blocks": blocks}
            usage = {
                "prompt_tokens": u_gen.get("prompt_tokens", 0) + u_edit.get("prompt_tokens", 0),
                "completion_tokens": u_gen.get("completion_tokens", 0) + u_edit.get("completion_tokens", 0),
                "total_tokens": u_gen.get("total_tokens", 0) + u_edit.get("total_tokens", 0),
            }
            return section, usage
        except Exception as e:
            section = {
                "title": group.get('problem_ru') or problem_key,
                "blocks": [{"type": "paragraph", "text": f"Не удалось получить анализ. Ошибка: {e}"}],
            }
            return section, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if args.concurrency and args.concurrency > 1:
        print(f"Параллельная генерация с concurrency={args.concurrency} ...")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            future_to_key = {ex.submit(process_section, p, groups.get(p)): p for p in problems_ordered if groups.get(p)}
            for fut in as_completed(future_to_key):
                section, usage = fut.result()
                sections.append(section)
                total_usage["prompt_tokens"] += usage["prompt_tokens"]
                total_usage["completion_tokens"] += usage["completion_tokens"]
                total_usage["total_tokens"] += usage["total_tokens"]
    else:
        for p in problems_ordered:
            g = groups.get(p)
            if not g:
                continue
            section, usage = process_section(p, g)
            sections.append(section)
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            total_usage["total_tokens"] += usage["total_tokens"]

    # Build distribution table (RU (EN)) excluding 'нет'
    from src.processing.assemble_data import build_distribution
    print("[5/6] Подготовка таблицы распределения проблем...")
    td0 = time.time()
    distribution = build_distribution(df, cols)
    td1 = time.time()
    print(f"[5/6] Готово за {td1 - td0:.2f} c")

    print("[6/6] Сборка DOCX...")
    tb0 = time.time()
    doc = build_document(cfg.country, f"Проблемы жителей {cfg.country}: тематический анализ", sections, distribution)
    output_path = output_dir / f"{cfg.output_basename}.docx"
    doc.save(str(output_path))
    tb1 = time.time()
    print(f"[6/6] Готово за {tb1 - tb0:.2f} c | Отчёт сохранён: {output_path}")
    print(f"ИТОГО токены (генерация+редактирование): {total_usage}")


if __name__ == "__main__":
    main()


