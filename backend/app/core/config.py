from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    youtube_channel_url: str = "https://www.youtube.com/@inspiremydaytiffany"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    database_url: str = "postgresql+asyncpg://tiiffbot:tiiffbot@localhost:5432/tiiffbot"

    jwt_secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440

    backend_cors_origins: str = "http://localhost:3000"

    chroma_host: str = "localhost"
    chroma_port: int = 8100

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
