"""Utilitaires pour le traitement audio (Version Portable GitHub)"""
import re
from pathlib import Path
from typing import Tuple, List
import torch
import torchaudio
import soundfile as sf
from loguru import logger

def clean_text(text: str) -> str:
    """Nettoie le texte (lowercase, remove special chars)"""
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def load_audio(wav_path: Path, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Charge un fichier audio 
    """
    try:
        # 1. Lecture avec soundfile (Garanti de marcher sur Windows/Mac/Linux sans FFmpeg)
        data, sr = sf.read(str(wav_path))
        
        # 2. Conversion au format TorchAudio exact : Tensor [channels, frames]
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
        # 3. Traitement avec TorchAudio (Resampling)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        
        # 4. Convertir en mono si stéréo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, sr
        
    except Exception as e:
        logger.error(f"Erreur chargement audio {wav_path}: {e}")
        raise

def load_reference(wav_path: Path) -> str:
    """Charge la référence textuelle (.txt associé au .wav)"""
    txt_path = wav_path.with_suffix('.txt')
    
    if not txt_path.exists():
        logger.error(f"Fichier .txt introuvable: {txt_path}")
        raise FileNotFoundError(f"Transcription introuvable pour {wav_path.name}")
    
    text = txt_path.read_text(encoding='utf-8')
    return clean_text(text)

def collect_wav_files(corpus_dir: Path, pattern: str = "**/*.wav") -> List[Path]:
    """Collecte tous les fichiers WAV dans un dossier"""
    wav_files = sorted(list(corpus_dir.glob(pattern)))
    
    if not wav_files:
        logger.error(f"Aucun fichier .wav trouvé dans {corpus_dir}")
        raise FileNotFoundError(f"Aucun .wav dans {corpus_dir}")
    
    logger.info(f"Trouvé {len(wav_files)} fichiers WAV dans {corpus_dir}")
    return wav_files

def get_audio_info(wav_path: Path) -> dict:
    """Récupère les informations d'un fichier audio (Portable)"""
    info = sf.info(str(wav_path))
    return {
        'sample_rate': info.samplerate,
        'num_frames': info.frames,
        'duration': info.duration,
        'num_channels': info.channels
    }