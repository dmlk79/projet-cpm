"""Évaluation WER et bootstrap CI"""
import numpy as np
from jiwer import wer
from typing import List, Tuple
from loguru import logger

def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Calcule le Word Error Rate (WER)
    
    Args:
        references: Liste des transcriptions de référence
        hypotheses: Liste des transcriptions prédites
        
    Returns:
        WER en pourcentage (0-100)
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
    
    if len(references) == 0:
        raise ValueError("Empty lists provided")
    
    W = wer(references, hypotheses)
    logger.info(f"WER: {W*100:.2f}% (N={len(references)})")
    
    return W * 100

def bootstrap_ci(
    references: List[str],
    hypotheses: List[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calcule le WER avec intervalle de confiance par bootstrap
    
    Args:
        references: Liste des transcriptions de référence
        hypotheses: Liste des transcriptions prédites
        n_boot: Nombre d'itérations bootstrap
        alpha: Niveau de significativité (0.05 = IC à 95%)
        seed: Seed pour reproductibilité
        
    Returns:
        (wer_mean, ci_low, ci_high) en pourcentage
    """
    rng = np.random.default_rng(seed)
    
    # Calcul WER par fichier
    per_file_wer = np.array([
        wer(r, h) for r, h in zip(references, hypotheses)
    ], dtype=float)
    
    n = len(per_file_wer)
    
    # Bootstrap
    boot_means = []
    for _ in range(n_boot):
        sample_idx = rng.choice(n, size=n, replace=True)
        boot_sample = per_file_wer[sample_idx]
        boot_means.append(boot_sample.mean())
    
    boot_means = np.sort(boot_means)
    
    # Calcul IC
    wer_mean = per_file_wer.mean()
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))
    
    logger.info(f"WER: {wer_mean*100:.2f}% [IC95: {ci_low*100:.2f}% - {ci_high*100:.2f}%]")
    
    return wer_mean * 100, ci_low * 100, ci_high * 100

def print_evaluation_results(
    wer_value: float,
    ci_low: float,
    ci_high: float,
    n_files: int,
    label: str = ""
):
    """
    Affiche les résultats d'évaluation de manière formatée
    
    Args:
        wer_value: WER moyen (%)
        ci_low: Borne inférieure IC95 (%)
        ci_high: Borne supérieure IC95 (%)
        n_files: Nombre de fichiers évalués
        label: Label optionnel pour identifier l'évaluation
    """
    print("\n" + "="*60)
    if label:
        print(f"ÉVALUATION: {label}")
    print("="*60)
    print(f"Nombre de fichiers: {n_files}")
    print(f"WER moyen:          {wer_value:.2f}%")
    print(f"IC95:               [{ci_low:.2f}%, {ci_high:.2f}%]")
    print("="*60 + "\n")

def compare_results(
    results_dict: dict,
    metric: str = "WER"
):
    """
    Compare plusieurs résultats d'évaluation
    
    Args:
        results_dict: Dict {label: (wer, ci_low, ci_high, n_files)}
        metric: Nom de la métrique (par défaut "WER")
    """
    print(f"\n{'='*70}")
    print(f"COMPARAISON DES RÉSULTATS - {metric}")
    print(f"{'='*70}")
    print(f"{'Configuration':<30} {'N':>6} {metric:>8} {'IC95':>20}")
    print("-"*70)
    
    for label, (wer_val, ci_low, ci_high, n_files) in results_dict.items():
        print(f"{label:<30} {n_files:>6} {wer_val:>7.2f}% [{ci_low:>5.2f}%, {ci_high:>5.2f}%]")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    print("✅ evaluation.py - Module d'évaluation WER + bootstrap CI")
