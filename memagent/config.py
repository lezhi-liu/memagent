"""Application settings, read from environment variables (and `.env` when present).

Every setting has a safe default except the ones that cost money or name a paid model:
`OPENAI_API_KEY`, `AGENT_MODEL` and `SEARCH_MODEL` are optional here and only checked
when the real OpenAI providers are built. That is what lets the tests run with no `.env`.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATABASE_URL = "postgresql+asyncpg://memagent:memagent@localhost:5433/memagent"


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    agent_model: str | None = None
    search_model: str | None = None
    database_url: str = DEFAULT_DATABASE_URL
    max_steps: int = 10
    workspace_root: Path = Path("data/workspaces")
    prices_path: Path = Path("config/prices.toml")

    def require_openai(self) -> tuple[str, str, str]:
        """Return (api_key, agent_model, search_model), or fail before any money is spent."""
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                ("AGENT_MODEL", self.agent_model),
                ("SEARCH_MODEL", self.search_model),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"missing required environment variables: {', '.join(missing)}")
        assert self.openai_api_key is not None
        assert self.agent_model is not None
        assert self.search_model is not None
        return self.openai_api_key.get_secret_value(), self.agent_model, self.search_model
