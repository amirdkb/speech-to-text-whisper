import os
import time
import librosa
from fastapi import HTTPException, UploadFile
from typing import Optional
import logging
from ..models.speech_model import whisper_model
from ..models.schemas import TranscriptionResponse, FileInfo
from ..services.file_service import FileService
from ..services.audio_processor import AudioProcessor  
from ..config import settings

logger = logging.getLogger(__name__)

def safe_log_transcription(logger, result_text):
    """
    Safely log transcription results that might contain Persian/Arabic text
    """
    try:
        preview = result_text[:100] if len(result_text) > 100 else result_text
        # Try to log normally first
        logger.info(f"Transcription result: '{preview}{'...' if len(result_text) > 100 else ''}'")
    except UnicodeEncodeError:
        # Fallback for Unicode issues
        char_count = len(result_text)
        logger.info(f"Transcription completed with {char_count} characters (contains Unicode text)")
    except Exception as e:
        logger.info(f"Transcription completed successfully (logging error: {str(e)})")

class SpeechController:
    def __init__(self):
        self.file_service = FileService()
        self.audio_processor = AudioProcessor(chunk_duration=25)  
   
    async def transcribe_audio(
        self,
        file: UploadFile,
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """Transcribe uploaded audio file with automatic chunking for long files"""
        start_time = time.time()
        temp_file_path = None
       
        try:
            # Validate file
            await self.file_service.validate_file(file)
            
            # Save uploaded file temporarily
            temp_file_path = await self.file_service.save_temp_file(file)
            
            # Verify the temp file was created and exists
            if not temp_file_path or not os.path.exists(temp_file_path):
                raise Exception(f"Failed to create temporary file: {temp_file_path}")
            
            logger.info(f"Temporary file created: {temp_file_path}")
            logger.info(f"File size: {os.path.getsize(temp_file_path)} bytes")
            
            # Get file info
            file_info = await self.file_service.get_file_info(file, temp_file_path)
            
            # Check if chunking is needed
            needs_chunking = self.audio_processor.will_need_chunking(temp_file_path)
            
            logger.info(f"Starting transcription for file: {file.filename}")
            logger.info(f"File duration: {file_info.duration:.2f}s, Chunking needed: {needs_chunking}")
            
            if needs_chunking:
                # Use chunked processing for long files
                logger.info("Using chunked processing for long audio file")
                result = await self.audio_processor.process_long_audio(
                    temp_file_path, 
                    whisper_model, 
                    language
                )
            else:
                # Use normal processing for short files
                logger.info("Using normal processing for short audio file")
                result = whisper_model.transcribe(temp_file_path, language)
           
            # Calculate processing time
            processing_time = time.time() - start_time
           
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            # Safe logging of transcription result
            safe_log_transcription(logger, result['transcription'])
           
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
   
    async def transcribe_audio_with_chunks_info(
        self,
        file: UploadFile,
        language: Optional[str] = None
    ) -> dict:
        """
        Transcribe audio and return additional information about chunking process
        Useful for debugging and monitoring long transcriptions
        """
        start_time = time.time()
        temp_file_path = None
       
        try:
            await self.file_service.validate_file(file)
            temp_file_path = await self.file_service.save_temp_file(file)
            
            # Verify the temp file was created
            if not temp_file_path or not os.path.exists(temp_file_path):
                raise Exception(f"Failed to create temporary file: {temp_file_path}")
                
            file_info = await self.file_service.get_file_info(file, temp_file_path)
           
            needs_chunking = self.audio_processor.will_need_chunking(temp_file_path)
            
            logger.info(f"Starting detailed transcription for file: {file.filename}")
            
            if needs_chunking:
                # Get chunk information
                estimated_chunks = self.audio_processor.estimate_chunks(temp_file_path)
                
                result = await self.audio_processor.process_long_audio(
                    temp_file_path, 
                    whisper_model, 
                    language
                )
                
                chunk_info = {
                    "total_chunks": estimated_chunks,
                    "chunk_duration": 25,  # seconds
                    "processing_method": "chunked"
                }
            else:
                result = whisper_model.transcribe(temp_file_path, language)
                chunk_info = {
                    "total_chunks": 1,
                    "chunk_duration": file_info.duration,
                    "processing_method": "normal"
                }
           
            processing_time = time.time() - start_time
           
            return {
                "transcription": result["transcription"],
                "language": result["language"],
                "confidence": result["confidence"],
                "processing_time": processing_time,
                "file_info": {
                    "filename": file.filename,
                    "size": file_info.size,
                    "duration": file_info.duration
                },
                "chunk_info": chunk_info
            }
           
        except Exception as e:
            logger.error(f"Error during detailed transcription: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
       
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
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

    async def get_file_info_with_chunking_preview(self, file: UploadFile) -> dict:
        """
        Get file info with information about how it would be processed
        """
        try:
            await self.file_service.validate_file(file)
            temp_file_path = await self.file_service.save_temp_file(file)
           
            try:
                file_info = await self.file_service.get_file_info(file, temp_file_path)
                needs_chunking = self.audio_processor.will_need_chunking(temp_file_path)
                
                result = {
                    "filename": file.filename,
                    "size": file_info.size,
                    "duration": file_info.duration,
                    "will_use_chunking": needs_chunking
                }
                
                if needs_chunking:
                    estimated_chunks = self.audio_processor.estimate_chunks(temp_file_path)
                    result["estimated_chunks"] = estimated_chunks
                    result["estimated_processing_time"] = f"{estimated_chunks * 5}-{estimated_chunks * 15} seconds"
                else:
                    result["estimated_chunks"] = 1
                    result["estimated_processing_time"] = "3-10 seconds"
                    
                return result
                
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                   
        except Exception as e:
            logger.error(f"Error getting file info with chunking preview: {str(e)}")
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
            "compute_type": getattr(settings, 'compute_type', 'auto'),
            "supported_languages": self.get_supported_languages(),
            "chunk_duration": 25,  # seconds
            "max_file_duration": "No limit (automatic chunking)"
        }

# Global controller instance
speech_controller = SpeechController()