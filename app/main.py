import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .views.speech_views import router as speech_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
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
    
    try:
        # Initialize model (this will load the model)
        from .models.speech_model import whisper_model
        logger.info("Model loaded successfully during startup")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Don't raise here - let the app start and handle errors in endpoints
    
    yield
    
    # Shutdown
    logger.info("Shutting down Speech-to-Text API")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A multilingual speech-to-text API using Whisper model with support for Persian and other languages",
    version=settings.version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
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
        "docs": "/docs",
        "health": "/api/v1/speech/health"
    }

# Health check at root level
@app.get("/health", tags=["Root"])
async def health():
    """Simple health check"""
    return {"status": "ok", "version": settings.version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )