from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List
import logging
from ..controllers.speech_controller import speech_controller
from ..models.schemas import (
    TranscriptionResponse, 
    ErrorResponse, 
    HealthResponse, 
    FileInfo
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/speech", tags=["Speech to Text"])

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        413: {"model": ErrorResponse, "description": "File Too Large"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Transcribe audio to text",
    description="Upload an audio file and get the transcribed text. Supports multiple languages including Persian."
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'fa' for Persian, 'en' for English)")
):
    """
    Transcribe an audio file to text using Whisper model.
    
    - **file**: Audio file (supported formats: wav, mp3, m4a, flac, aac, ogg, wma)
    - **language**: Optional language code to force specific language detection
    """
    try:
        return await speech_controller.transcribe_audio(file, language)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in transcribe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post(
    "/file-info",
    response_model=FileInfo,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        413: {"model": ErrorResponse, "description": "File Too Large"}
    },
    summary="Get audio file information",
    description="Get information about an uploaded audio file including duration and format."
)
async def get_file_info(
    file: UploadFile = File(..., description="Audio file to analyze")
):
    """
    Get information about an audio file.
    
    - **file**: Audio file to analyze
    """
    try:
        return await speech_controller.get_file_info(file)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file-info endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/languages",
    response_model=List[str],
    summary="Get supported languages",
    description="Get list of all languages supported by the Whisper model."
)
async def get_supported_languages():
    """
    Get list of supported languages.
    """
    try:
        return speech_controller.get_supported_languages()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in languages endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/model-info",
    response_model=dict,
    summary="Get model information",
    description="Get information about the loaded Whisper model."
)
async def get_model_info():
    """
    Get information about the loaded model.
    """
    try:
        return speech_controller.get_model_info()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in model-info endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the speech-to-text service."
)
async def health_check():
    """
    Health check endpoint.
    """
    try:
        supported_languages = speech_controller.get_supported_languages()
        model_loaded = len(supported_languages) > 0
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            version=settings.version,
            model_loaded=model_loaded,
            device=settings.device,
            supported_languages=supported_languages[:10]  # Limit to first 10 for brevity
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.version,
            model_loaded=False,
            device=settings.device,
            supported_languages=[]
        )