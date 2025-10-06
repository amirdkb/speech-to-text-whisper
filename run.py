#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Set environment variables for Unicode support
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSFSENCODING'] = '1'

# Ensure stdout/stderr use UTF-8
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("Starting Speech-to-Text Whisper API...")
    print(f"Environment: {'Development' if settings.debug else 'Production'}")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"Documentation: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
        access_log=True
    )