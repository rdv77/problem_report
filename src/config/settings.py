import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelsConfig:
    classify: str
    generate: str
    editor: str


@dataclass
class AppConfig:
    country: str
    language: str
    max_messages_per_problem: int
    max_chars_per_message: int
    temperature: float
    max_tokens: int
    on_missing_facts: str  # "skip" | "search"
    output_basename: str
    models: ModelsConfig


def load_config_from_env(country: str | None = None) -> AppConfig:
    """Build configuration using environment variables with sane defaults.
    CLI can pass `country` to override.
    """
    models = ModelsConfig(
        classify=os.getenv("OPENAI_MODEL_CLASSIFY", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        generate=os.getenv("OPENAI_MODEL_GENERATE", os.getenv("OPENAI_MODEL", "gpt-4.1")),
        editor=os.getenv("OPENAI_MODEL_EDITOR", os.getenv("OPENAI_MODEL", "gpt-4.1")),
    )

    return AppConfig(
        country=country or os.getenv("REPORT_COUNTRY", "Tanzania"),
        language=os.getenv("REPORT_LANG", "ru"),
        max_messages_per_problem=int(os.getenv("MAX_MESSAGES_PER_PROBLEM", "30")),
        max_chars_per_message=int(os.getenv("MAX_CHARS_PER_MESSAGE", "2000")),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "5000")),
        on_missing_facts=os.getenv("ON_MISSING_FACTS", "skip"),
        output_basename=os.getenv(
            "OUTPUT_BASENAME",
            f"problems_report_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ),
        models=models,
    )


