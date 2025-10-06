import os
import uuid
import aiofiles
import librosa
from fastapi import UploadFile, HTTPException
from typing import Optional
import logging
from ..config import settings
from ..models.schemas import FileInfo

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size
        self.allowed_extensions = settings.allowed_extensions
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        # Check if file exists
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        if file.size and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {self.max_file_size / 1024 / 1024:.1f}MB"
            )
        
        # Check file extension
        if file.filename:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in self.allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Allowed formats: {', '.join(self.allowed_extensions)}"
                )
        else:
            raise HTTPException(status_code=400, detail="Invalid filename")
    
    async def save_temp_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location"""
        try:
            # Generate unique filename
            file_ext = os.path.splitext(file.filename)[1].lower()
            temp_filename = f"{uuid.uuid4()}{file_ext}"
            temp_file_path = os.path.join(self.upload_dir, temp_filename)
            
            # Save file
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                content = await file.read()
                await temp_file.write(content)
            
            # Reset file pointer for potential reuse
            await file.seek(0)

            logger.debug(f"Saved temporary file: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error saving temporary file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    async def get_file_info(self, file: UploadFile, file_path: str) -> FileInfo:
        """Get information about the audio file"""
        try:
            # Get basic file info
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else file.size
            
            # Get audio duration
            duration = None
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                logger.warning(f"Could not get audio duration: {str(e)}")
            
            return FileInfo(
                filename=file.filename,
                size=file_size,
                content_type=file.content_type or "unknown",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get file information")
    
    def cleanup_old_files(self, max_age_hours: int = 1) -> None:
        """Clean up old temporary files"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            logger.debug(f"Cleaned up old file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")
                            
        except Exception as e:
            logger.error(f"Error during file cleanup: {str(e)}")
            