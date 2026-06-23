from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Proptyze API"
    debug: bool = True
    database_url: str = "sqlite:///./proptyze_dev.db"
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 500
    allowed_video_formats: list[str] = ["mp4", "mov", "avi"]
    roboflow_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
