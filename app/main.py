import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .views.speech_views import router as speech_router
from logging.handlers import RotatingFileHandler

def setup_unicode_safe_logging():
    """
    Setup logging configuration that can handle Unicode characters
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Remove all existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler with UTF-8 encoding
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # This is crucial for Unicode support
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("app").setLevel(logging.DEBUG)
    
    print("Unicode-safe logging configured successfully")

# Safe logging function for Unicode text
def safe_log(logger, level, message, *args, **kwargs):
    """
    Safely log messages that might contain Unicode characters
    """
    try:
        if args:
            message = message % args
        logger.log(level, message, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode problematic characters
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        logger.log(level, f"[Unicode Error in Log] {safe_message}", **kwargs)
    except Exception as e:
        logger.log(logging.ERROR, f"Logging error: {str(e)}")

def log_transcription_result(logger, result_text):
    """
    Safely log transcription results that might contain Persian/Arabic text
    """
    try:
        preview = result_text[:100] if len(result_text) > 100 else result_text
        safe_log(logger, logging.INFO, f"Transcription result: '{preview}{'...' if len(result_text) > 100 else ''}'")
    except Exception as e:
        logger.info(f"Transcription completed with {len(result_text)} characters (logging Unicode chars failed)")

# Setup logging at module level
setup_unicode_safe_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    # Startup
    logger.info("Starting Speech-to-Text API")
    logger.info(f"Using device: {settings.device}")
    logger.info(f"Model: {settings.model_name}")
    
    # Create upload directory
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"Upload directory created/verified: {settings.upload_dir}")
    
    # Test model loading
    try:
        from .models.speech_model import whisper_model
        logger.info("Model loaded successfully during startup")
        
        # Test if model has basic functionality
        if hasattr(whisper_model, 'get_supported_languages'):
            languages = whisper_model.get_supported_languages()
            logger.info(f"Model supports {len(languages)} languages")
        
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Don't raise here - let the app start but log the error
    
    yield
    
    # Shutdown
    logger.info("Shutting down Speech-to-Text API")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A multilingual speech-to-text API using Whisper model with support for Persian and other languages. Automatically chunks long audio files for processing.",
    version=settings.version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception on {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "detail": str(exc),
            "path": str(request.url)
        }
    )

# Handle HTTPException specifically
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Include routers
app.include_router(speech_router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "description": "Multilingual Speech-to-Text API with automatic chunking",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/speech/health",
            "transcribe": "/api/v1/speech/transcribe",
            "model_info": "/api/v1/speech/model-info"
        }
    }

# Health check at root level
@app.get("/health", tags=["Root"])
async def health():
    """Simple health check"""
    return {
        "status": "ok", 
        "version": settings.version,
        "service": "speech-to-text-whisper"
    }

# Additional utility endpoint for debugging
@app.get("/debug/config", tags=["Debug"])
async def debug_config():
    """Debug endpoint to check configuration (remove in production)"""
    return {
        "upload_dir": settings.upload_dir,
        "upload_dir_exists": os.path.exists(settings.upload_dir),
        "model_name": settings.model_name,
        "device": settings.device,
        "max_file_size_mb": settings.max_file_size / (1024 * 1024),
        "allowed_extensions": settings.allowed_extensions
    }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure logging is setup before starting server
    print(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )