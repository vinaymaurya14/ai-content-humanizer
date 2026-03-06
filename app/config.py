"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    nltk_data_path: str = "./nltk_data"
    min_sentence_length: int = 5
    max_batch_size: int = 50
    default_humanization_strength: float = 0.7
    preserve_keywords: bool = True
    max_content_length: int = 50000
    target_keyword_density_min: float = 0.5
    target_keyword_density_max: float = 2.5

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


settings = Settings()
