# Speech-to-Text API

A multilingual speech-to-text API built with FastAPI and OpenAI's Whisper model. Supports Persian (Farsi) and many other languages with GPU acceleration.

## Features

- **Multilingual Support**: Supports 99+ languages including Persian (Farsi)
- **GPU Acceleration**: Optimized for NVIDIA GPUs (RTX 4050 6GB supported)
- **Multiple Audio Formats**: WAV, MP3, M4A, FLAC, AAC, OGG, WMA
- **RESTful API**: Clean REST endpoints with automatic documentation
- **MVC Architecture**: Well-organized codebase following MVC pattern
- **Error Handling**: Comprehensive error handling and logging
- **File Validation**: Automatic file format and size validation
- **Health Monitoring**: Health check endpoints for monitoring

## System Requirements

- **GPU**: NVIDIA RTX 4050 6GB (or better)
- **RAM**: 32GB (recommended)
- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU acceleration)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd speech-to-text-api

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file (or use the provided one):

```env
APP_NAME=Speech to Text API
DEBUG=False
HOST=127.0.0.1
PORT=8000
MODEL_NAME=openai/whisper-medium
DEVICE=cuda
MAX_FILE_SIZE=52428800  # 50MB
```

### 3. Run the API

```bash
# Using the run script
python run.py

# Or directly with uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000

# For development with auto-reload
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 4. API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/speech/transcribe` | Transcribe audio to text |
| `POST` | `/api/v1/speech/file-info` | Get audio file information |
| `GET` | `/api/v1/speech/languages` | Get supported languages |
| `GET` | `/api/v1/speech/model-info` | Get model information |
| `GET` | `/api/v1/speech/health` | Health check |

### Usage Examples

#### Transcribe Audio

```bash
# Basic transcription (auto-detect language)
curl -X POST "http://localhost:8000/api/v1/speech/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"

# Transcription with specific language (Persian)
curl -X POST "http://localhost:8000/api/v1/speech/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "language=fa"
```

#### Python Client Example

```python
import requests

# Transcribe audio file
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    data = {'language': 'fa'}  # Optional: specify language
    
    response = requests.post(
        'http://localhost:8000/api/v1/speech/transcribe',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Transcription: {result['transcription']}")
    print(f"Language: {result['language']}")
```

## Supported Languages

The Whisper-medium model supports 99+ languages including:

- **Persian (Farsi)**: `fa`
- **English**: `en`
- **Arabic**: `ar`
- **Chinese**: `zh`
- **French**: `fr`
- **German**: `de`
- **Spanish**: `es`
- **Russian**: `ru`
- **Japanese**: `ja`
- **Korean**: `ko`
- And many more...

Get the full list: `GET /api/v1/speech/languages`

## Project Structure

```
speech-to-text-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── speech_model.py    # Whisper model wrapper
│   │   └── schemas.py         # Pydantic models
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── speech_controller.py  # Business logic
│   ├── services/
│   │   ├── __init__.py
│   │   └── file_service.py    # File handling service
│   └── views/
│       ├── __init__.py
│       └── speech_views.py    # API routes
├── uploads/                   # Temporary file uploads
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables
├── run.py                   # Application runner
├── test_client.py           # Test client
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose
└── README.md
```

## MVC Architecture

- **Models** (`app/models/`): Data models and ML model wrapper
- **Views** (`app/views/`): API routes and request/response handling  
- **Controllers** (`app/controllers/`): Business logic and coordination
- **Services** (`app/services/`): Utility services (file handling, etc.)

## Testing

Run the test client:

```bash
# Basic API tests
python test_client.py

# Test with audio file
python test_client.py audio.wav

# Test with specific language
python test_client.py audio.wav fa
```

## Docker Deployment

### Build and run with Docker

```bash
# Build image
docker build -t speech-to-text-api .

# Run container
docker run -p 8000:8000 --gpus all speech-to-text-api
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Performance Optimization

### For RTX 4050 6GB:

1. **Model Selection**: `whisper-medium` is optimal for 6GB VRAM
2. **Precision**: Uses `float16` for GPU, `float32` for CPU
3. **Memory Management**: Automatic cleanup of temporary files
4. **Batch Processing**: Single file processing to manage memory

### Alternative Models:

- `whisper-small`: Faster, less accurate (1GB VRAM)
- `whisper-base`: Fastest, lowest accuracy (500MB VRAM)
- `whisper-large`: Highest accuracy (requires 10GB+ VRAM)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `openai/whisper-medium` | Hugging Face model name |
| `DEVICE` | `cuda` | Device to use (cuda/cpu) |
| `MAX_FILE_SIZE` | `52428800` | Max upload size (50MB) |
| `UPLOAD_DIR` | `uploads` | Temporary upload directory |
| `SAMPLE_RATE` | `16000` | Audio sample rate |

### Model Configuration

Edit `app/config.py` to change model settings:

```python
# For different GPU memory
model_name: str = "openai/whisper-small"  # For 2-4GB GPU
model_name: str = "openai/whisper-large"  # For 10GB+ GPU
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use smaller model (`whisper-small` or `whisper-base`)
   - Reduce `max_file_size` in config
   - Process shorter audio files

2. **Model Loading Slow**
   - First run downloads model (~1.4GB for medium)
   - Subsequent runs use cached model
   - Models cached in `~/.cache/huggingface/`

3. **Audio Format Issues**
   - Install `ffmpeg`: `sudo apt install ffmpeg`
   - Supported formats: WAV, MP3, M4A, FLAC, AAC, OGG, WMA

### Logs

Check application logs:
```bash
tail -f app.log
```

## Development

### Setup Development Environment

```bash
# Install in development mode
pip install -e .

# Run with auto-reload
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Adding New Features

1. **Models**: Add data models in `app/models/schemas.py`
2. **Business Logic**: Add logic in `app/controllers/`
3. **API Routes**: Add routes in `app/views/`
4. **Services**: Add utilities in `app/services/`

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request