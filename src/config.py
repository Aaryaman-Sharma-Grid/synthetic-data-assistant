"""Typed application configuration loaded from environment variables."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_genai_use_vertexai: bool = True
    google_cloud_project: str = Field(default="gd-gcp-gridu-genai", validation_alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", validation_alias="GOOGLE_CLOUD_LOCATION")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")

    database_url: str = Field(
        default="postgresql+psycopg://app_user:change-me@localhost:5432/synthetic_data",
        validation_alias="DATABASE_URL",
    )

    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL"
    )
    langfuse_tracing_environment: str = Field(
        default="local", validation_alias="LANGFUSE_TRACING_ENVIRONMENT"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
