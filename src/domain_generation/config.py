"""Configuration parameters for domain content generation."""

from pydantic_settings import BaseSettings


class GenerationConfig(BaseSettings):
    """Configuration for content generation parameters."""

    # Domain and Topic Configuration
    topics_min_items: int = 3
    topics_max_items: int = 5
    articles_min_per_topic: int = 3

    # Book Configuration
    toc_min_items: int = 3
    toc_max_items: int = 5
    sections_min_per_chapter: int = 2
    sections_max_per_chapter: int = 4

    # Article Configuration
    sections_min_per_article: int = 3
    sections_max_per_article: int = 5

    # QA Configuration
    grounded_qa_min_items: int = 10
    grounded_qa_max_items: int = 15
    ungrounded_qa_min_items: int = 5
    ungrounded_qa_max_items: int = 8

    # LLM Configuration
    model_name: str = "gpt-5-mini"
    fallback_model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_retries: int = 5
    max_tokens: int = 16000  # Maximum tokens for LLM responses

    class Config:
        env_file = ".env"
        env_prefix = "GEN_"
        extra = "ignore"  # Ignore extra fields from .env file


# Global config instance
config = GenerationConfig()
