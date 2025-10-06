import os
import tempfile
import asyncio
from typing import List, Tuple, Optional
from pydub import AudioSegment
from pydub.utils import which
import logging
from pathlib import Path
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, chunk_duration: int = 25):
        """
        Initialize AudioProcessor
        
        Args:
            chunk_duration: Duration of each chunk in seconds
        """
        self.chunk_duration = chunk_duration * 1000  # Convert to milliseconds
        self._setup_ffmpeg()
        
    def _setup_ffmpeg(self):
        """Setup FFmpeg path if available"""
        try:
            # Try to find ffmpeg and ffprobe
            ffmpeg_path = which("ffmpeg")
            ffprobe_path = which("ffprobe")
            
            if not ffmpeg_path:
                logger.warning("FFmpeg not found. Audio processing may be limited to WAV files.")
            if not ffprobe_path:
                logger.warning("FFprobe not found. Audio duration detection may be limited.")
                
        except Exception as e:
            logger.warning(f"Error setting up FFmpeg: {str(e)}")
        
    def load_audio(self, file_path: str) -> AudioSegment:
        """
        Load audio file using multiple fallback methods
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioSegment object
        """
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Try pydub first (requires ffmpeg for most formats)
            try:
                audio = AudioSegment.from_file(file_path)
                logger.debug(f"Successfully loaded audio with pydub: {file_path}")
                return audio
            except Exception as pydub_error:
                logger.warning(f"Pydub failed to load {file_path}: {str(pydub_error)}")
                
                # Fallback: Use librosa + soundfile
                try:
                    # Load with librosa
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Create temporary WAV file
                    temp_wav = file_path.replace(os.path.splitext(file_path)[1], '_temp.wav')
                    sf.write(temp_wav, y, sr)
                    
                    # Load with pydub
                    audio = AudioSegment.from_wav(temp_wav)
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                        
                    logger.info(f"Successfully loaded audio with librosa fallback: {file_path}")
                    return audio
                    
                except Exception as librosa_error:
                    logger.error(f"Librosa fallback also failed: {str(librosa_error)}")
                    raise Exception(f"Failed to load audio file with both pydub and librosa: {str(pydub_error)}")
            
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def chunk_audio(self, audio: AudioSegment) -> List[AudioSegment]:
        """
        Split audio into chunks of specified duration
        
        Args:
            audio: AudioSegment to chunk
            
        Returns:
            List of AudioSegment chunks
        """
        chunks = []
        total_duration = len(audio)
        
        logger.info(f"Audio duration: {total_duration/1000:.2f}s, will create {(total_duration // self.chunk_duration) + 1} chunks")
        
        for start_time in range(0, total_duration, self.chunk_duration):
            end_time = min(start_time + self.chunk_duration, total_duration)
            chunk = audio[start_time:end_time]
            chunks.append(chunk)
            
        return chunks
    
    def create_chunk_file(self, chunk: AudioSegment, temp_dir: str, chunk_index: int) -> str:
        """
        Create temporary file for audio chunk
        
        Args:
            chunk: AudioSegment chunk
            temp_dir: Temporary directory
            chunk_index: Index of the chunk
            
        Returns:
            Path to temporary chunk file
        """
        temp_file = os.path.join(temp_dir, f"chunk_{chunk_index}.wav")
        try:
            chunk.export(temp_file, format="wav")
            logger.debug(f"Created chunk file: {temp_file}")
            return temp_file
        except Exception as e:
            logger.error(f"Error creating chunk file: {str(e)}")
            raise
    
    def merge_transcriptions(self, transcription_results: List[dict]) -> dict:
        """
        Merge multiple transcription results into one
        Compatible with your existing TranscriptionResponse format
        
        Args:
            transcription_results: List of transcription dictionaries
            
        Returns:
            Merged transcription dictionary
        """
        if not transcription_results:
            return {
                "transcription": "", 
                "language": "en", 
                "confidence": None
            }
        
        merged_text = ""
        confidences = []
        detected_language = transcription_results[0].get("language", "en")
        
        for i, result in enumerate(transcription_results):
            # Add space between chunks if not the first chunk
            if i > 0 and merged_text and not merged_text.endswith(" "):
                merged_text += " "
            
            merged_text += result["transcription"]
            
            # Collect confidences (even if None)
            if result.get("confidence") is not None:
                confidences.append(result["confidence"])
        
        # Calculate average confidence if we have any
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        return {
            "transcription": merged_text.strip(),
            "language": detected_language,
            "confidence": avg_confidence
        }
    
    async def process_long_audio(
        self, 
        file_path: str, 
        whisper_model,
        language: Optional[str] = None
    ) -> dict:
        """
        Main method to process long audio files by chunking
        Works with your existing whisper model interface
        
        Args:
            file_path: Path to audio file
            whisper_model: The whisper model instance
            language: Optional language hint
            
        Returns:
            Complete transcription result
        """
        try:
            # Verify file exists first
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            logger.info(f"Processing audio file: {file_path}")
            
            # Load audio
            audio = self.load_audio(file_path)
            
            # Check if audio is shorter than chunk duration
            if len(audio) <= self.chunk_duration:
                logger.info("Audio is shorter than chunk duration, processing normally")
                return whisper_model.transcribe(file_path, language)
            
            logger.info(f"Audio is {len(audio)/1000:.2f}s long, will process in chunks")
            
            # Create chunks
            chunks = self.chunk_audio(audio)
            
            # Create temporary directory for chunk files
            with tempfile.TemporaryDirectory(prefix="whisper_chunks_") as temp_dir:
                logger.info(f"Created temporary directory: {temp_dir}")
                transcription_results = []
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Create temporary file for chunk
                    chunk_file = self.create_chunk_file(chunk, temp_dir, i)
                    
                    try:
                        # Verify chunk file was created
                        if not os.path.exists(chunk_file):
                            logger.error(f"Chunk file was not created: {chunk_file}")
                            continue
                            
                        # Transcribe chunk using your existing model
                        result = whisper_model.transcribe(chunk_file, language)
                        transcription_results.append(result)
                        
                        # Safe logging to avoid Unicode errors
                        transcription_preview = result['transcription'][:50] if result['transcription'] else ""
                        logger.info(f"Chunk {i+1} completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1}: {str(e)}")
                        # Continue with other chunks even if one fails
                        # Add empty result to maintain chunk order
                        transcription_results.append({
                            "transcription": "",
                            "language": language or "unknown",
                            "confidence": None
                        })
                    
                    finally:
                        # Clean up chunk file
                        if os.path.exists(chunk_file):
                            try:
                                os.remove(chunk_file)
                                logger.debug(f"Cleaned up chunk file: {chunk_file}")
                            except Exception as e:
                                logger.warning(f"Failed to cleanup chunk file: {str(e)}")
                
                # Merge all transcriptions
                merged_result = self.merge_transcriptions(transcription_results)
                logger.info(f"Merged transcription completed: {len(merged_result['transcription'])} total characters")
                
                return merged_result
                
        except Exception as e:
            logger.error(f"Error in process_long_audio: {str(e)}")
            raise Exception(f"Failed to process long audio: {str(e)}")
    
    def get_audio_duration(self, file_path: str) -> float:
        """
        Get audio duration in seconds using multiple fallback methods
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return 0.0
                
            # Try librosa first (more reliable)
            try:
                duration = librosa.get_duration(path=file_path)
                logger.debug(f"Got duration with librosa: {duration:.2f}s")
                return duration
            except Exception as librosa_error:
                logger.warning(f"Librosa duration detection failed: {str(librosa_error)}")
                
                # Fallback to pydub
                try:
                    audio = self.load_audio(file_path)
                    duration = len(audio) / 1000.0
                    logger.debug(f"Got duration with pydub: {duration:.2f}s")
                    return duration
                except Exception as pydub_error:
                    logger.error(f"Both librosa and pydub duration detection failed: {str(pydub_error)}")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return 0.0
    
    def will_need_chunking(self, file_path: str) -> bool:
        """
        Check if file will need chunking based on duration
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if chunking is needed
        """
        try:
            duration = self.get_audio_duration(file_path)
            chunk_duration_sec = self.chunk_duration / 1000.0
            needs_chunking = duration > chunk_duration_sec
            
            logger.debug(f"File duration: {duration:.2f}s, Chunk duration: {chunk_duration_sec}s, Needs chunking: {needs_chunking}")
            return needs_chunking
        except Exception as e:
            logger.error(f"Error determining chunking need: {str(e)}")
            return False
    
    def estimate_chunks(self, file_path: str) -> int:
        """
        Estimate number of chunks needed for a file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Estimated number of chunks
        """
        try:
            duration = self.get_audio_duration(file_path)
            if duration <= 0:
                return 1
                
            chunk_duration_sec = self.chunk_duration / 1000.0
            chunks = int(duration / chunk_duration_sec) + (1 if duration % chunk_duration_sec > 0 else 0)
            logger.debug(f"Estimated {chunks} chunks for {duration:.2f}s audio")
            return chunks
        except Exception:
            return 1