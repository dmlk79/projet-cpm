"""Fonctions d'inférence ASR"""
import torch
from pathlib import Path
from typing import List, Tuple
from loguru import logger
from tqdm import tqdm
import audio_utils
import config

@torch.no_grad()
def transcribe_greedy(
    wav_path: Path,
    processor,
    model,
    device=config.DEVICE
) -> str:
    """
    Transcription greedy (sans modèle de langage)
    
    Args:
        wav_path: Chemin vers le fichier WAV
        processor: Wav2Vec2Processor
        model: Wav2Vec2ForCTC
        device: Device (cuda/cpu)
        
    Returns:
        Texte transcrit nettoyé
    """
    waveform, sr = audio_utils.load_audio(wav_path, config.SAMPLE_RATE)
    
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    
    logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    
    return audio_utils.clean_text(text)

@torch.no_grad()
def transcribe_with_lm(
    wav_path: Path,
    processor,
    model,
    decoder,
    device=config.DEVICE
) -> str:
    """
    Transcription avec modèle de langage (beam search)
    
    Args:
        wav_path: Chemin vers le fichier WAV
        processor: Wav2Vec2Processor
        model: Wav2Vec2ForCTC
        decoder: CTC decoder avec LM
        device: Device (cuda/cpu)
        
    Returns:
        Texte transcrit nettoyé
    """
    waveform, sr = audio_utils.load_audio(wav_path, config.SAMPLE_RATE)
    
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    
    logits = model(inputs.input_values.to(device)).logits
    logits_np = logits[0].cpu().numpy()
    
    text = decoder.decode(logits_np)
    return audio_utils.clean_text(text)

def batch_transcribe(
    wav_files: List[Path],
    processor,
    model,
    decoder=None,
    use_lm: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Transcription batch avec progress bar et logs
    
    Args:
        wav_files: Liste des fichiers WAV
        processor: Wav2Vec2Processor
        model: Wav2Vec2ForCTC
        decoder: CTC decoder (optionnel)
        use_lm: Utiliser le modèle de langage
        
    Returns:
        (references, hypotheses) - listes des transcriptions
    """
    references = []
    hypotheses = []
    
    mode = "avec LM" if use_lm else "greedy"
    logger.info(f"Début transcription de {len(wav_files)} fichiers ({mode})")
    
    errors = 0
    
    for wav_path in tqdm(wav_files, desc=f"Transcription {mode}"):
        try:
            # Charger la référence
            ref = audio_utils.load_reference(wav_path)
            
            # Transcrire
            if use_lm and decoder:
                hyp = transcribe_with_lm(wav_path, processor, model, decoder)
            else:
                hyp = transcribe_greedy(wav_path, processor, model)
            
            references.append(ref)
            hypotheses.append(hyp)
            
        except Exception as e:
            logger.error(f"Erreur sur {wav_path.name}: {e}")
            errors += 1
            continue
    
    logger.info(f"Transcription terminée: {len(hypotheses)}/{len(wav_files)} fichiers")
    if errors > 0:
        logger.warning(f"{errors} erreurs rencontrées")
    
    return references, hypotheses

if __name__ == "__main__":
    print("inference.py - Module de transcription ASR")
