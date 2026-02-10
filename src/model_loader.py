"""Chargement des modèles et décodeurs"""
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
import config
from loguru import logger
import os

def load_model():
    """Charge le modèle acoustique et le processeur"""
    logger.info(f"Chargement du modèle {config.MODEL_NAME}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
        model.eval() # Mode évaluation important
        return processor, model
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        raise

def load_decoder(processor, lm_path=None):
    """
    Construit le décodeur CTC (avec ou sans Language Model)
    
    Args:
        processor: Le processeur Wav2Vec2
        lm_path: Chemin vers le fichier .arpa ou .bin (KenLM)
    """
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted((v, k) for k, v in vocab_dict.items())
    labels = [k for v, k in sorted_vocab]

    # Remplacer le token padding par "" pour pyctcdecode
    labels[processor.tokenizer.pad_token_id] = ""
    
    # Remplacer les tokens spéciaux par "" ou ?
    for i in range(len(labels)):
        if labels[i] in [processor.tokenizer.word_delimiter_token, '|']:
            labels[i] = " "
    
    logger.info(f"Construction du décodeur (LM={lm_path if lm_path else 'None'})...")
    
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(lm_path) if lm_path else None,
    )
    
    return decoder

if __name__ == "__main__":
    p, m = load_model()
    print("✅ Modèle chargé avec succès")