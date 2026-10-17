"""Runtime settings, read from the environment (and a local .env if present)."""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://arl:arl@localhost:5433/arl"
    )
    # Default is a free-tier provider so the project costs $0 to run. `anthropic` also works.
    llm_provider: str = os.environ.get("LLM_PROVIDER", "groq")
    # Model names change: list what your key can use and set LLM_MODEL in .env.
    llm_model: str = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    llm_max_retries: int = int(os.environ.get("LLM_MAX_RETRIES", "6"))   # backoff on rate limits
    max_spend_usd: float = float(os.environ.get("MAX_SPEND_USD", "0"))
    # USD per million tokens, for the "cost per successful task" metric. Free tiers charge nothing,
    # so by default cost is reported as token counts only. Set these to price tokens at a
    # reference rate (e.g. a paid equivalent) if you want a dollar figure; label it as such.
    price_in_per_mtok: float = float(os.environ.get("PRICE_IN_PER_MTOK", "0"))
    price_out_per_mtok: float = float(os.environ.get("PRICE_OUT_PER_MTOK", "0"))

    @property
    def psycopg_url(self) -> str:
        """Plain libpq URL for psycopg / LangGraph's Postgres checkpointer."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


settings = Settings()
