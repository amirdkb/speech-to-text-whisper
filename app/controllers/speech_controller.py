import os
import time
import librosa
from fastapi import HTTPException, UploadFile
from typing import Optional
import logging
from ..models.speech_model import whisper_model
from ..models.schemas import TranscriptionResponse, FileInfo
from ..services.file_service import FileService
from ..config import settings

logger = logging.getLogger(__name__)

class SpeechController:
    def __init__(self):
        self.file_service = FileService()
    
    async def transcribe_audio(
        self, 
        file: UploadFile, 
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """Transcribe uploaded audio file"""
        start_time = time.time()
        temp_file_path = None
        
        try:
            # Validate file
            await self.file_service.validate_file(file)
            
            # Save uploaded file temporarily
            temp_file_path = await self.file_service.save_temp_file(file)
            
            # Get file info
            file_info = await self.file_service.get_file_info(file, temp_file_path)
            
            # Transcribe audio
            logger.info(f"Starting transcription for file: {file.filename}")
            result = whisper_model.transcribe(temp_file_path, language)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            return TranscriptionResponse(
                transcription=result["transcription"],
                language=result["language"],
                confidence=result["confidence"],
                processing_time=processing_time,
                file_size=file_info.size,
                duration=file_info.duration
            )
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
        finally:
            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file: {str(e)}")
    
    async def get_file_info(self, file: UploadFile) -> FileInfo:
        """Get information about uploaded audio file"""
        try:
            await self.file_service.validate_file(file)
            temp_file_path = await self.file_service.save_temp_file(file)
            
            try:
                file_info = await self.file_service.get_file_info(file, temp_file_path)
                return file_info
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        try:
            return whisper_model.get_supported_languages()
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get supported languages")
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": settings.model_name,
            "device": settings.device,
            "compute_type": settings.compute_type,
            "supported_languages": self.get_supported_languages()
        }

# Global controller instance
speech_controller = SpeechController()