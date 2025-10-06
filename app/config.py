from pydantic_settings import BaseSettings
from typing import Optional
import torch

class Settings(BaseSettings):
    # App settings
    app_name: str = "Speech to Text API"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8001
    
    # Whisper model settings
    model_name: str = "vhdm/whisper-large-fa-v1"  # Good balance for 6GB GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"
    
    # File upload settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "uploads"
    allowed_extensions: list = [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"]
    
    # Audio processing settings
    sample_rate: int = 16000
    
    class Config:
        env_file = ".env"

settings = Settings()