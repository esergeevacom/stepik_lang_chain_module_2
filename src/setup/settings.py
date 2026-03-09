from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class OpenAiSetting(BaseSettings):
    key: str
    model: str
    base_url: str | None = None
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=2)
    temperature: float = Field(default=0.0)

    model_config = SettingsConfigDict(env_prefix="OPENAI_API_")


openai_settings = OpenAiSetting()
