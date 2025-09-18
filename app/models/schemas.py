from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(None, description="Language code (e.g., 'fa' for Persian, 'en' for English)")
    
class TranscriptionResponse(BaseModel):
    transcription: str = Field(..., description="The transcribed text")
    language: str = Field(..., description="Detected or specified language")
    confidence: Optional[float] = Field(None, description="Confidence score (if available)")
    processing_time: float = Field(..., description="Processing time in seconds")
    file_size: int = Field(..., description="File size in bytes")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device being used (cpu/cuda)")
    supported_languages: List[str] = Field(..., description="List of supported languages")

class FileInfo(BaseModel):
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the file")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")