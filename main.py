"""
Script Principal ASR - Master IA2VR
-----------------------------------
Orchestre l'ensemble du TP :
1. Chargement des modèles (Wav2Vec2 + N-gram)
2. Transcription du corpus (Greedy + LM)
3. Calcul des WER et Intervalles de Confiance (Bootstrap)
4. Génération des graphiques d'analyse (SNR, Locuteur, Longueur)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys

# === IMPORT DES MODULES DU PROJET ===
sys.path.append("src")
import config
import model_loader
import inference
import audio_utils
import evaluation

# === CONFIGURATION DES CHEMINS ===
# Adaptez ces chemins si votre structure change
CORPUS_ROOT = config.DATA_DIR / "corpus" / "td_corpus_digits_wav"
LM_PATH = config.DATA_DIR / "lm_data" / "lm-data" / "2-gram.pruned.1e-7.arpa"

# Sorties
OUTPUT_CSV = config.PROJECT_ROOT / "results_detailed.csv"
STATS_CSV = config.PROJECT_ROOT / "results_stats.csv"
PLOTS_DIR = config.PROJECT_ROOT / "plots"

def parse_metadata(wav_path: Path):
    """
    Extrait les métadonnées depuis le chemin du fichier.
    Structure attendue : .../SNRxx/Speaker/SeqXdigits.../file.wav
    """
    parts = wav_path.parts
    
    # 1. Extraction de la longueur (dossier parent)
    length_folder = parts[-2]
    if "seq1" in length_folder: length = "1"
    elif "seq3" in length_folder: length = "3"
    elif "seq5" in length_folder: length = "5"
    else: length = "Unknown"
    
    # 2. Extraction du locuteur et SNR
    speaker = parts[-3]
    snr = parts[-4]
    
    return snr, speaker, length

def plot_with_ci(df_stats, x_col, title, filename, color_no="#4c72b0", color_lm="#dd8452"):
    """
    Trace un diagramme en barres avec les intervalles de confiance (barres d'erreur).
    """
    plt.figure(figsize=(10, 6))
    
    # Paramètres de position
    bar_width = 0.35
    indices = np.arange(len(df_stats))
    
    # Préparation des données NoLM (Greedy)
    means_no = df_stats['WER_NoLM']
    # Matplotlib attend des erreurs relatives : [valeur - low, high - valeur]
    err_no = [
        means_no - df_stats['CI_Low_NoLM'],
        df_stats['CI_High_NoLM'] - means_no
    ]
    
    # Préparation des données LM
    means_lm = df_stats['WER_LM']
    err_lm = [
        means_lm - df_stats['CI_Low_LM'],
        df_stats['CI_High_LM'] - means_lm
    ]
    
    # Tracé des barres
    plt.bar(indices - bar_width/2, means_no, bar_width, yerr=err_no, capsize=5, 
            label='Sans LM (Greedy)', color=color_no, alpha=0.9)
    plt.bar(indices + bar_width/2, means_lm, bar_width, yerr=err_lm, capsize=5, 
            label='Avec LM (2-gram)', color=color_lm, alpha=0.9)
    
    # Esthétique
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel('Word Error Rate (%)', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(indices, df_stats[x_col], fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Sauvegarde
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Graphique généré : {save_path}")

def generate_analysis(df):
    """
    Analyse les résultats bruts, calcule les stats par Bootstrap et génère les graphes.
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Fonction interne pour calculer les stats sur un sous-groupe
    def calculate_group_stats(group):
        refs = group['Reference'].tolist()
        hyps_no = group['Hyp_NoLM'].tolist()
        hyps_lm = group['Hyp_LM'].fillna("").tolist()
        
        # Appel à evaluation.py pour le Bootstrap
        wer_no, low_no, high_no = evaluation.bootstrap_ci(refs, hyps_no, n_boot=1000)
        wer_lm, low_lm, high_lm = evaluation.bootstrap_ci(refs, hyps_lm, n_boot=1000)
        
        return pd.Series({
            'WER_NoLM': wer_no, 'CI_Low_NoLM': low_no, 'CI_High_NoLM': high_no,
            'WER_LM': wer_lm, 'CI_Low_LM': low_lm, 'CI_High_LM': high_lm
        })

    logger.info("--- Début de l'analyse statistique (Bootstrap) ---")

    # 1. ANALYSE DU BRUIT (SNR)
    # On filtre sur le locuteur 'man' pour avoir une comparaison valide sur tous les niveaux
    logger.info("Analyse 1/3 : Impact du Bruit (sur locuteur 'man')...")
    df_snr = df[df['Speaker'] == 'man'].groupby('SNR').apply(calculate_group_stats).reset_index()
    
    # Tri logique des SNR (pas alphabétique)
    snr_order = {'SNR05dB': 0, 'SNR15dB': 1, 'SNR25dB': 2, 'SNR35dB': 3}
    df_snr['sort_key'] = df_snr['SNR'].map(snr_order)
    df_snr = df_snr.sort_values('sort_key').drop('sort_key', axis=1)
    
    plot_with_ci(df_snr, 'SNR', "Impact du Bruit sur le WER (Locuteur : Man)", "graph1_snr_ci.png")

    # 2. ANALYSE DU LOCUTEUR
    # On filtre sur SNR35dB car c'est là que tous les locuteurs sont présents
    logger.info("Analyse 2/3 : Impact du Locuteur (à SNR 35dB)...")
    df_spk = df[df['SNR'] == 'SNR35dB'].groupby('Speaker').apply(calculate_group_stats).reset_index()
    plot_with_ci(df_spk, 'Speaker', "Impact du Locuteur (à SNR 35dB)", "graph2_speaker_ci.png")

    # 3. ANALYSE DE LA LONGUEUR
    logger.info("Analyse 3/3 : Impact de la Longueur des séquences...")
    df_len = df.groupby('Length').apply(calculate_group_stats).reset_index()
    plot_with_ci(df_len, 'Length', "Impact de la Longueur des séquences", "graph3_length_ci.png")
    
    # Sauvegarde des stats pour le rapport LaTeX
    logger.info("Sauvegarde des tableaux de statistiques...")
    with open(STATS_CSV, 'w') as f:
        f.write("# Stats SNR (Man only)\n")
        df_snr.to_csv(f, index=False)
        f.write("\n# Stats Speaker (SNR35dB only)\n")
        df_spk.to_csv(f, index=False)
        f.write("\n# Stats Length (All)\n")
        df_len.to_csv(f, index=False)
    
    logger.success(f"Statistiques sauvegardées dans {STATS_CSV}")

def main():
    # --- ETAPE 1 : CHARGEMENT ---
    logger.info("Chargement des modèles...")
    processor, model = model_loader.load_model()
    
    decoder_lm = None
    if LM_PATH.exists():
        decoder_lm = model_loader.load_decoder(processor, LM_PATH)
        logger.info(f"Modèle de langage chargé : {LM_PATH.name}")
    else:
        logger.warning(f"Fichier LM introuvable ({LM_PATH}). Mode Greedy uniquement.")

    # --- ETAPE 2 : SCAN DU CORPUS ---
    logger.info(f"Scan du dossier {CORPUS_ROOT}...")
    all_wavs = list(CORPUS_ROOT.rglob("*.wav"))
    
    if not all_wavs:
        logger.error("Aucun fichier .wav trouvé ! Vérifiez le chemin dans config.py ou main.py")
        return

    logger.info(f"Fichiers trouvés : {len(all_wavs)}")

    # --- ETAPE 3 : TRANSCRIPTION (INFERENCE) ---
    results = []
    logger.info("Démarrage de la transcription...")
    
    # Utilisation de tqdm pour la barre de progression
    for wav_path in tqdm(all_wavs, desc="Traitement", unit="wav"):
        try:
            # A. Parsing infos
            snr, speaker, length = parse_metadata(wav_path)
            ref_text = audio_utils.load_reference(wav_path)

            # B. Inférence
            # 1. Greedy (Sans LM)
            hyp_nolm = inference.transcribe_greedy(wav_path, processor, model)
            
            # 2. Avec LM (si dispo)
            hyp_lm = ""
            if decoder_lm:
                hyp_lm = inference.transcribe_with_lm(wav_path, processor, model, decoder_lm)

            # C. Stockage
            results.append({
                "Filename": wav_path.name,
                "SNR": snr,
                "Speaker": speaker,
                "Length": length,
                "Reference": ref_text,
                "Hyp_NoLM": hyp_nolm,
                "Hyp_LM": hyp_lm
            })

        except Exception as e:
            # On log l'erreur mais on ne coupe pas le script
            logger.error(f"Erreur sur {wav_path.name}: {e}")

    # --- ETAPE 4 : SAUVEGARDE RESULTATS BRUTS ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Transcriptions sauvegardées dans {OUTPUT_CSV}")

    # --- ETAPE 5 : ANALYSE ET GRAPHIQUES ---
    generate_analysis(df)
    
    print("\n" + "="*50)
    print("✅  TP TERMINÉ AVEC SUCCÈS")
    print(f"📁  Résultats détaillés : {OUTPUT_CSV}")
    print(f"📊  Tableaux statistiques : {STATS_CSV}")
    print(f"📈  Graphiques générés : {PLOTS_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()