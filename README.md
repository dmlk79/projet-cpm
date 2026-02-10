# Système ASR Robuste : Évaluation de Wav2Vec 2.0 et Modèles de Langage N-gram

## 🎯 Objectif du Projet

Ce projet a été réalisé dans le cadre du **Master 2 IA²VR (Intelligence Artificielle et ses Applications en Vision et Robotique)** à l'Université de Lorraine. L'objectif est d'évaluer la robustesse et les performances d'un système de reconnaissance automatique de la parole (ASR) de pointe face à des contraintes acoustiques et linguistiques réelles.

Le pipeline analyse le modèle **Wav2Vec 2.0 (Base-960h)** de Facebook (Meta AI) selon trois axes critiques :

1. **Robustesse acoustique** : Impact de la dégradation du signal (bruit blanc) avec des niveaux de SNR (Signal-to-Noise Ratio) allant de 5dB à 35dB.
2. **Variabilité des locuteurs** : Comparaison des performances sur des profils vocaux hétérogènes (hommes, femmes, enfants).
3. **Correction sémantique** : Quantification du gain de précision (réduction du Word Error Rate) apporté par l'intégration d'un **Modèle de Langage N-gram** via un décodage par Beam Search.

## 📂 Structure du Projet

L'organisation logicielle est modulaire pour garantir une séparation claire des responsabilités et une portabilité maximale entre différents environnements de calcul.

```text
projet-cpm/
├── main.py              # Script principal (Chef d'orchestre du pipeline)
├── requirements.txt     # Dépendances Python (Torchaudio, Pyctcdecode, etc.)
├── .gitignore           # Exclusion des environnements, données lourdes et caches
├── results_stats.csv    # Résultats consolidés (Moyennes WER et Intervalles de Confiance)
├── results_detailed.csv # Base de données complète des 2800 transcriptions brutes
├── plots/               # Visualisations scientifiques générées
│   ├── graph1_snr_ci.png    # Impact du niveau de bruit
│   ├── graph2_speaker_ci.png # Performance par type de locuteur
│   └── graph3_length_ci.png  # Influence de la longueur des séquences
├── src/                 # Cœur logique du système
│   ├── __init__.py      
│   ├── config.py        # Gestion du GPU, des chemins et hyperparamètres
│   ├── audio_utils.py   # Chargement audio portable (Soundfile/TorchAudio hybride)
│   ├── model_loader.py  # Chargement Wav2Vec2 et décodeur KenLM
│   ├── inference.py     # Algorithmes de transcription (Greedy vs Beam Search)
│   └── evaluation.py    # Métriques (WER) et Bootstrap statistique (IC 95%)
├── logs/                # Journaux d'exécution (Suivi des performances GPU et erreurs)
└── data/                # [IGNORÉ PAR GIT] Corpus audio et Modèle de Langage (.arpa)
```

## 📊 Méthodologie Scientifique

### Évaluation du Taux d'Erreur (WER)

Pour chaque échantillon audio, le système produit et évalue deux types de transcriptions :

- **WER Greedy** : Performance brute du modèle acoustique (décisions locales par frame).
- **WER with LM** : Performance après intégration des probabilités linguistiques du modèle de langage N-gram.

### Analyse Statistique (Bootstrap)

Afin de garantir la validité scientifique des conclusions, nous appliquons la méthode du Bootstrap :

- **Intervalles de Confiance (IC 95%)** : Calculés sur 1000 itérations de ré-échantillonnage pour chaque métrique.

Cette approche permet de confirmer statistiquement que les écarts de performance observés ne sont pas dus à la variance de l'échantillon mais bien aux caractéristiques intrinsèques du modèle et des données.

## 🚀 Guide de Démarrage

### Prérequis

Le projet a été développé et optimisé sur une instance Google Cloud Compute Engine équipée d'un GPU NVIDIA L4. Pour assurer la reproductibilité sur n'importe quel OS (Linux, Windows, macOS), la lecture audio est gérée via soundfile, éliminant les dépendances complexes liées aux codecs système (FFmpeg).

### Installation

```bash
# Création et activation de l'environnement virtuel
python3 -m venv asr_env
source asr_env/bin/activate

# Installation des dépendances (sans cache pour optimiser l'espace disque)
pip install --no-cache-dir -r requirements.txt
```

### Exécution

Pour traiter l'intégralité du corpus (2800 fichiers), calculer les métriques et générer les graphiques d'analyse :

```bash
python main.py
```

## 📈 Résultats et Analyse

Les graphiques générés dans le dossier `/plots` mettent en évidence la corrélation inverse entre le SNR et le WER. L'apport du modèle de langage est particulièrement significatif dans les zones de bruit modéré, où les contraintes linguistiques permettent de lever les ambiguïtés phonétiques que le modèle acoustique seul ne peut résoudre.

## 👤 Auteurs

- **El Hadji Dame Lo Kaba** - Étudiant Master 2 IA²VR, Université de Lorraine
- **Salim Fourati** - Étudiant Master 2 IA²VR, Université de Lorraine

---

## 📝 License

Ce projet a été réalisé dans un cadre académique à l'Université de Lorraine.
