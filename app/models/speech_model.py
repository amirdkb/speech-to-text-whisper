import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Optional, Tuple
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class WhisperModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = settings.device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model and processor"""
        try:
            logger.info(f"Loading Whisper model: {settings.model_name}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Using dtype: {self.dtype}")
            
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(settings.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=self.dtype
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio file for Whisper"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=settings.sample_rate)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Transcribe audio file to text"""
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Process audio with Whisper processor
            inputs = self.processor(
                audio, 
                sampling_rate=settings.sample_rate, 
                return_tensors="pt"
            )
            
            # Move inputs to device AND convert to the same dtype as the model
            inputs = {
                k: v.to(self.device, dtype=self.dtype) if v.dtype.is_floating_point else v.to(self.device) 
                for k, v in inputs.items()
            }
            
            # Generate transcription
            with torch.no_grad():
                if language:
                    # Force specific language
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language, 
                        task="transcribe"
                    )
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        forced_decoder_ids=forced_decoder_ids
                    )
                else:
                    # Auto-detect language
                    predicted_ids = self.model.generate(inputs["input_features"])
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Detect language if not specified
            if not language:
                detected_language = self._detect_language(inputs["input_features"])
            else:
                detected_language = language
            
            return {
                "transcription": transcription,
                "language": detected_language,
                "confidence": None  # Whisper doesn't provide confidence scores
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def _detect_language(self, input_features: torch.Tensor) -> str:
        """Detect the language of the audio"""
        try:
            with torch.no_grad():
                # Generate language detection tokens
                decoder_input_ids = torch.tensor(
                    [[self.model.config.decoder_start_token_id]], 
                    device=self.device,
                    dtype=torch.long  # decoder_input_ids should always be long (int64)
                )
                
                # Ensure input_features has the correct dtype
                if input_features.dtype != self.dtype:
                    input_features = input_features.to(self.dtype)
                
                logits = self.model(input_features, decoder_input_ids=decoder_input_ids).logits
                
                # Get language probabilities
                lang_code_to_id = self.processor.tokenizer.lang_code_to_id
                language_token_ids = list(lang_code_to_id.values())
                
                # Extract logits for language tokens
                language_tokens = logits[0, 0, language_token_ids]
                language_probs = torch.softmax(language_tokens, dim=-1)
                
                # Get the most likely language
                max_prob_idx = torch.argmax(language_probs)
                lang_codes = list(lang_code_to_id.keys())
                detected_language = lang_codes[max_prob_idx]
                
                return detected_language
                
        except Exception as e:
            logger.warning(f"Error detecting language: {str(e)}")
            return "unknown"
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        if self.processor:
            return list(self.processor.tokenizer.lang_code_to_id.keys())
        return []

# Global model instance
whisper_model = WhisperModel()