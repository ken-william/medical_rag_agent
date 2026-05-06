#!/usr/bin/env python3
"""
Script de validation de la recherche vectorielle (Étape 5 - Test conseillé).
Recherche des chunks pertinents pour 5 questions types et affiche les scores L2.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

def main():
    # Charger index, métadonnées et modèle
    print("Chargement de la base vectorielle...")
    index = faiss.read_index(INDEX_PATH)
    
    print("Chargement des métadonnées...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Chargement du modèle d'embedding {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    test_questions = [
        "Quels sont les effets secondaires du Doliprane ?",
        "Quelle est la posologie recommandée pour l'amoxicilline adulte ?",
        "Y a-t-il des contre-indications entre l'ibuprofène et l'aspirine ?",
        "Que faire en cas de surdosage d'imodium ?",
        "Puis-je prendre du Smecta pour des troubles gastriques ?"
    ]
    
    print("\n=== Lancement des tests de recherche vectorielle ===\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i} : \"{question}\"")
        
        # Encoder la question
        query_vector = model.encode([question], convert_to_numpy=True)
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # Rechercher les top 3 chunks les plus proches
        distances, indices = index.search(query_vector, k=3)
        
        print("Top 3 Chunks récupérés :")
        for idx, (dist, chunk_idx) in enumerate(zip(distances[0], indices[0]), 1):
            if chunk_idx == -1 or chunk_idx >= len(chunks):
                print(f"  {idx}. [Vide]")
                continue
                
            chunk = chunks[chunk_idx]
            meta = chunk["metadata"]
            print(f"  {idx}. [Médicament: {meta['medicament']}] - Rubrique: \"{meta['section_label']}\" [Score L2: {dist:.2f}]")
            # Afficher les 120 premiers caractères du contenu original
            original_text = chunk["contenu_original"]
            preview = original_text[:120].replace("\n", " ") + ("..." if len(original_text) > 120 else "")
            print(f"     Texte: \"{preview}\"")
            
if __name__ == "__main__":
    main()
