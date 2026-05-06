#!/usr/bin/env python3
"""
Script d'indexation de la base de données des médicaments (Sujet B).
Filtre les médicaments cibles, les découpe en chunks par section,
génère les embeddings et crée l'index FAISS persistant.
"""

import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import warnings
import logging

# Supprimer silencieusement les avertissements techniques en arrière-plan
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configuration
EXCEL_PATH = Path("Creation_agent_RAG/CIS_RCP_export.xlsx")
INDEX_PATH = Path("faiss_index.bin")
METADATA_PATH = Path("metadata.json")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"



# Sections de la notice à indexer
SECTIONS_TO_INDEX = [
    "composition",
    "forme_pharmaceutique",
    "indications",
    "posologie",
    "contre_indications",
    "mises_en_garde",
    "interactions",
    "grossesse_allaitement",
    "effets_indesirables",
    "surdosage",
    "excipients",
    "conditions_prescription"
]

# Dictionnaire de traduction des sections pour le prompt/affichage
SECTION_LABELS = {
    "composition": "Composition",
    "forme_pharmaceutique": "Forme pharmaceutique",
    "indications": "Indications thérapeutiques",
    "posologie": "Posologie et mode d'administration",
    "contre_indications": "Contre-indications",
    "mises_en_garde": "Mises en garde et précautions d'emploi",
    "interactions": "Interactions avec d'autres médicaments",
    "grossesse_allaitement": "Grossesse, allaitement et fertilité",
    "effets_indesirables": "Effets indésirables",
    "surdosage": "Surdosage",
    "excipients": "Liste des excipients",
    "conditions_prescription": "Conditions de prescription et de délivrance"
}

def clean_text(text: str) -> str:
    """Nettoie le texte brut extrait de la notice."""
    if not isinstance(text, str):
        return ""
    # Remplacer le caractère spécial \x92 (apostrophe courbe mal encodée)
    text = text.replace("\x92", "'")
    text = text.replace("\x96", "-")
    # Nettoyer les espaces multiples et retours à la ligne
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunker(texte: str, taille_max: int = 800, overlap: int = 150) -> list[str]:
    """
    Découpe un texte en morceaux (chunks) cohérents.
    Utilise le séparateur ' | ' comme frontière naturelle privilégiée.
    """
    if not texte:
        return []
    
    parts = [p.strip() for p in texte.split(' | ') if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0
    
    for part in parts:
        part_len = len(part)
        # Si on peut ajouter cette partie au chunk courant
        if current_len + part_len + (3 if current_chunk else 0) <= taille_max:
            current_chunk.append(part)
            current_len += part_len + (3 if len(current_chunk) > 1 else 0)
        else:
            if current_chunk:
                chunks.append(" | ".join(current_chunk))
            
            # Si la partie elle-même dépasse la taille maximale
            if part_len > taille_max:
                # Découpage par caractères avec recouvrement
                for i in range(0, part_len, taille_max - overlap):
                    chunks.append(part[i : i + taille_max])
                current_chunk = []
                current_len = 0
            else:
                current_chunk = [part]
                current_len = part_len
                
    if current_chunk:
        chunks.append(" | ".join(current_chunk))
        
    return chunks

def prepare_documents(df: pd.DataFrame) -> list[dict]:
    """
    Parcourt le dataframe des médicaments, dédoublonne par premier mot du libellé
    (en gardant la notice la plus complète) et prépare les chunks avec leurs métadonnées.
    """
    print("Dédoublonnement et filtrage global du corpus par molécule (premier mot)...")
    
    # 1. Associer chaque ligne à son médicament canonique et calculer la longueur totale de son texte
    valid_rows = []
    for _, row in df.iterrows():
        denomination = clean_text(row["denomination"])
        if not denomination:
            continue
            
        # Extraire le premier mot comme représentant de la molécule (nom de marque ou générique principal)
        first_word = denomination.split()[0].strip().upper()
        # Supprimer les caractères non-alphabétiques à la fin du mot
        first_word = re.sub(r'[^A-Z0-9]', '', first_word)
        if not first_word or len(first_word) < 2:
            continue
            
        # Calculer la longueur cumulée du texte de toutes les rubriques
        total_length = sum(len(clean_text(row.get(s, ""))) for s in SECTIONS_TO_INDEX)
        
        valid_rows.append({
            "row": row,
            "first_word": first_word,
            "total_length": total_length
        })
        
    # 2. Regrouper par premier mot et sélectionner la notice la plus longue/complète
    best_rows_by_molecule = {}
    for item in valid_rows:
        mol = item["first_word"]
        if mol not in best_rows_by_molecule or item["total_length"] > best_rows_by_molecule[mol]["total_length"]:
            best_rows_by_molecule[mol] = item
            
    selected_items = list(best_rows_by_molecule.values())
    print(f"-> {len(selected_items)} molécules uniques sélectionnées pour l'indexation globale.")
    
    # 3. Générer les chunks pour les notices sélectionnées
    documents = []
    chunk_count = 0
    
    for item in selected_items:
        row = item["row"]
        mol = item["first_word"]
        denomination = clean_text(row["denomination"])
        code_cis = str(row["code_cis"])
        date_mise_a_jour = clean_text(row["date_mise_a_jour"])
        
        # Parcourir chaque section à indexer
        for section in SECTIONS_TO_INDEX:
            raw_content = row.get(section, "")
            content = clean_text(raw_content)
            if not content or content == "nan":
                continue
                
            # Découper en chunks
            section_chunks = chunker(content, taille_max=800, overlap=150)
            
            for idx, chunk_content in enumerate(section_chunks):
                chunk_id = f"doc_{code_cis}_{section}_{idx}"
                
                # Texte enrichi pour maximiser la pertinence sémantique
                enriched_content = (
                    f"Médicament: {mol} ({denomination}) | "
                    f"Rubrique: {SECTION_LABELS[section]} | "
                    f"Contenu: {chunk_content}"
                )
                
                documents.append({
                    "id": chunk_id,
                    "contenu_enrichi": enriched_content,
                    "contenu_original": chunk_content,
                    "metadata": {
                        "id": chunk_id,
                        "medicament": mol,
                        "denomination": denomination,
                        "section": section,
                        "section_label": SECTION_LABELS[section],
                        "code_cis": code_cis,
                        "date_mise_a_jour": date_mise_a_jour
                    }
                })
                chunk_count += 1
                
    print(f"-> Total de chunks générés pour la base complète : {chunk_count}")
    return documents

def main():
    if not EXCEL_PATH.exists():
        print(f"Erreur : Le fichier {EXCEL_PATH} est introuvable.")
        return

    print(f"Lecture de {EXCEL_PATH}...")
    df = pd.read_excel(EXCEL_PATH)
    print(f"Chargement réussi : {len(df)} médicaments trouvés dans la base globale.")

    # Préparation des chunks
    documents = prepare_documents(df)
    if not documents:
        print("Aucun document n'a été préparé. Vérifiez le filtrage.")
        return

    # Chargement du modèle d'embedding
    print(f"Chargement du modèle d'embedding : {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Extraction du texte enrichi à embedder
    texts_to_embed = [doc["contenu_enrichi"] for doc in documents]
    
    print("Calcul des embeddings (cette étape peut prendre quelques minutes)...")
    embeddings = model.encode(
        texts_to_embed, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    # Conversion des embeddings en float32 pour FAISS
    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    print(f"Embeddings calculés. Dimension : {dimension}, Nombre : {embeddings.shape[0]}")

    # Création de l'index FAISS L2
    print("Création de l'index FAISS...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Index FAISS créé avec {index.ntotal} vecteurs.")

    # Sauvegarde sur le disque
    print(f"Sauvegarde de l'index FAISS dans '{INDEX_PATH}'...")
    faiss.write_index(index, str(INDEX_PATH))

    # Sauvegarde des métadonnées associées (dans le même ordre)
    print(f"Sauvegarde des métadonnées dans '{METADATA_PATH}'...")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print("\n=== Indexation terminée avec succès ! ===")
    print(f"- Index FAISS : {INDEX_PATH.resolve()}")
    print(f"- Métadonnées : {METADATA_PATH.resolve()}")

if __name__ == "__main__":
    main()
