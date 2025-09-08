"""
Configuration settings for the audio processing pipeline.
"""
import os
from pathlib import Path

class Config:
    # Audio settings
    AUDIO_FORMAT = "flac"
    AUDIO_QUALITY = "best"  # YouTube-dl quality
    SAMPLE_RATE = 24000  # Hz
    SEGMENT_MIN_DURATION = 10.0  # seconds
    SEGMENT_MAX_DURATION = 15.0  # seconds
    SEGMENT_OVERLAP = 0.5  # seconds for continuity
    
    # Diarization settings
    PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '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')
    
    # Voice overlap detection
    OVERLAP_THRESHOLD = 0.5  # Threshold for overlap detection
    MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to consider
    
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    AUDIOS_BAIXADOS_DIR = Path(r"C:\Users\Usuário\Desktop\katube-novo\audios_baixados")
    OUTPUT_DIR = AUDIOS_BAIXADOS_DIR / "output"
    TEMP_DIR = AUDIOS_BAIXADOS_DIR / "temp"
    SEGMENTS_DIR = OUTPUT_DIR / "segments"
    SPEAKERS_DIR = OUTPUT_DIR / "speakers"
    
    # YouTube download settings
    YOUTUBE_FORMAT = f"bestaudio[ext={AUDIO_FORMAT}]/best[ext={AUDIO_FORMAT}]/bestaudio/best"
    
    # STT preparation settings
    MAX_SEGMENT_SIZE = 25 * 1024 * 1024  # 25MB max per segment for STT
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.OUTPUT_DIR, cls.TEMP_DIR, cls.SEGMENTS_DIR, cls.SPEAKERS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
