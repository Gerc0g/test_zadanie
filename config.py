from pydantic_settings import BaseSettings
from pydantic import Field
import json

class Settings(BaseSettings):

    LOG_LEVEL: str


    class Config:
        env_file = ".env"


settings = Settings()
