#!/bin/bash
set -e

# Vérifie si data existe déjà
if [ -d "data" ]; then
    echo "Data already exists in project."
    exit 0
fi

# Vérifie argument
if [ -z "$1" ]; then
    echo "Usage: bash setup_data.sh /path/to/data"
    exit 1
fi

SOURCE="$1"

# Vérifie que le dossier source existe
if [ ! -d "$SOURCE" ]; then
    echo "Error: Provided path does not exist."
    exit 1
fi

# Vérifie présence des sous-dossiers requis
if [ ! -d "$SOURCE/lm_data" ] || [ ! -d "$SOURCE/corpus" ]; then
    echo "Error: The folder must contain 'lm_data' and 'corpus'."
    exit 1
fi

# Copie le contenu du dossier source dans le dossier courant
cp -r "$SOURCE"/* ./

echo "Data successfully installed in current directory."
