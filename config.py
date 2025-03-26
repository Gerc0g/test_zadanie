from pydantic_settings import BaseSettings
from typing import Optional
class Settings(BaseSettings):

    LOG_LEVEL: str
    OPENAI_API_KEY: str
    OPENAI_PROXY: Optional[str] = None
    MODEL_NAME: str
    MODEL_EMBEDDING_NAME: str
    PATH_TO_VECTOR_STORE: str
    GENERATION_TEMPERATURE: float
    GENERATION_TOKENS: int

    class Config:
        env_file = ".env"


settings = Settings()
