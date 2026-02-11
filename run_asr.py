#!/usr/bin/env python3
"""
Script principal pour la reconnaissance automatique de la parole (ASR)
Usage: python run_asr.py --corpus data/corpus --use-lm
"""
import sys
from pathlib import Path
import argparse
import time
from loguru import logger

import config
import audio_utils
import model_loader
import inference
import evaluation

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))



def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="ASR avec Wav2Vec2 - Évaluation WER avec bootstrap CI"
    )
    
    parser.add_argument(
        "--corpus",
        type=Path,
        default=config.CORPUS_DIR,
        help=f"Dossier contenant les fichiers .wav (défaut: {config.CORPUS_DIR})"
    )
    
    parser.add_argument(
        "--use-lm",
        action="store_true",
        help="Utiliser le modèle de langage n-gram pour le décodage"
    )
    
    parser.add_argument(
        "--lm-path",
        type=Path,
        default=None,
        help="Chemin vers le fichier .arpa du modèle de langage"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.wav",
        help="Pattern de recherche des fichiers WAV (défaut: **/*.wav)"
    )
    
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Nombre d'itérations bootstrap pour IC (défaut: 2000)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Nombre max de fichiers à traiter (pour tests rapides)"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()
    
    logger.info("="*70)
    logger.info("DÉMARRAGE ASR - Wav2Vec2")
    logger.info("="*70)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Modèle de langage: {'Oui' if args.use_lm else 'Non'}")
    logger.info("="*70)
    
    # Vérifier que le corpus existe
    if not args.corpus.exists():
        logger.error(f"Corpus introuvable: {args.corpus}")
        return 1
    
    # Collecter les fichiers WAV
    try:
        wav_files = audio_utils.collect_wav_files(args.corpus, args.pattern)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Limiter le nombre de fichiers si demandé
    if args.max_files:
        wav_files = wav_files[:args.max_files]
        logger.info(f"Limitation à {args.max_files} fichiers")
    
    # Charger le modèle Wav2Vec2
    logger.info("Chargement du modèle Wav2Vec2...")
    start_time = time.time()
    processor, model = model_loader.load_wav2vec2_model()
    model = model_loader.optimize_for_inference(model)
    logger.info(f"Modèle chargé en {time.time() - start_time:.2f}s")
    
    # Charger le modèle de langage si demandé
    decoder = None
    if args.use_lm:
        logger.info("Chargement du modèle de langage...")
        decoder = model_loader.load_language_model(args.lm_path)
        if decoder is None:
            logger.warning("Impossible de charger le LM, passage en mode greedy")
            args.use_lm = False
    
    # Transcription
    logger.info(f"Début de la transcription de {len(wav_files)} fichiers...")
    start_time = time.time()
    
    references, hypotheses = inference.batch_transcribe(
        wav_files=wav_files,
        processor=processor,
        model=model,
        decoder=decoder,
        use_lm=args.use_lm
    )
    
    transcription_time = time.time() - start_time
    logger.info(f"Transcription terminée en {transcription_time:.2f}s")
    logger.info(f"Temps moyen par fichier: {transcription_time/len(wav_files):.3f}s")
    
    # Évaluation
    logger.info("Calcul du WER avec bootstrap CI...")
    wer_mean, ci_low, ci_high = evaluation.bootstrap_ci(
        references=references,
        hypotheses=hypotheses,
        n_boot=args.n_boot,
        seed=args.seed
    )
    
    # Affichage des résultats
    mode_label = "Avec modèle de langage" if args.use_lm else "Greedy (sans LM)"
    evaluation.print_evaluation_results(
        wer_value=wer_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        n_files=len(references),
        label=mode_label
    )
    
    # Exemples de transcriptions
    logger.info("Exemples de transcriptions:")
    for i in range(min(3, len(references))):
        logger.info(f"\nFichier: {wav_files[i].name}")
        logger.info(f"  REF: {references[i]}")
        logger.info(f"  HYP: {hypotheses[i]}")
    
    logger.info("\n" + "="*70)
    logger.info("ÉVALUATION TERMINÉE")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
