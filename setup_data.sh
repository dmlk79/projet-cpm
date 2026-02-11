#!/bin/bash
set -e

# Chemin source prédéfini (à modifier selon ton cas)
SOURCE="/chemin/vers/ton/dossier/data"

# Vérifie si data existe déjà dans le dossier courant
if [ -d "data" ]; then
    echo "Data already exists in project."
    exit 0
fi

# Vérifie que le dossier source existe
if [ ! -d "$SOURCE" ]; then
    echo "Error: Source path does not exist: $SOURCE"
    exit 1
fi

# Vérifie présence des sous-dossiers requis
if [ ! -d "$SOURCE/lm_data" ] || [ ! -d "$SOURCE/corpus" ]; then
    echo "Error: The folder must contain 'lm_data' and 'corpus'."
    exit 1
fi

# Copie le dossier data dans le répertoire courant
cp -r "$SOURCE" ./data

echo "Data successfully installed in project."
