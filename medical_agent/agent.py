#!/usr/bin/env python3
"""
Définition officielle de l'Agent ADK pour google-adk (agent.py).
"""

import os
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.adk.agents import Agent
from google.adk.tools import FunctionTool


# Local path references
INDEX_PATH = Path("faiss_index.bin")
METADATA_PATH = Path("metadata.json")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Load resources safely
if INDEX_PATH.exists() and METADATA_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embedding_model = SentenceTransformer(MODEL_NAME)
else:
    index, metadata, embedding_model = None, None, None


def rechercher_notice_medicament(requete: str) -> str:
    """
    Recherche les rubriques de notices de médicaments les plus pertinentes par rapport à une requête.
    Utilisez cet outil pour toute question sur la posologie, les effets indésirables, les indications,
    les contre-indications ou la composition d'un médicament.
    
    Args:
        requete: La question de l'utilisateur ou mots-clés (ex: 'posologie doliprane' ou 'effets indésirables nurofen').
    """
    if index is None:
        return "Erreur : La base vectorielle locale n'est pas initialisée. Veuillez lancer indexation.py."
        
    query_vector = embedding_model.encode([requete], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype=np.float32)
    
    distances, indices = index.search(query_vector, k=4)
    
    results = []
    for idx, (dist, chunk_idx) in enumerate(zip(distances[0], indices[0]), 1):
        if chunk_idx == -1 or chunk_idx >= len(metadata):
            continue
            
        # Filtrage par seuil L2 (seuil de confiance de 24.0)
        if dist > 24.0:
            continue
            
        chunk = metadata[chunk_idx]
        meta = chunk["metadata"]
        
        results.append(
            f"Source #{idx} - Notice de {meta['denomination']} (Rubrique: {meta['section_label']})\n"
            f"Date de mise à jour: {meta['date_mise_a_jour']}\n"
            f"Contenu de la notice: {chunk['contenu_original']}\n"
        )
        
    if not results:
        return "Je ne trouve pas cette information dans ma base de connaissances des notices de médicaments courants."
        
    return "\n---\n".join(results)

# Consignes de comportement de médecin bienveillant pour l'agent ADK
INSTRUCTIONS = (
    "Vous êtes un médecin expert et un assistant d'information médicale chaleureux, humain et très professionnel.\n"
    "Votre rôle est d'accueillir l'utilisateur, de dialoguer courtoisement avec lui, de répondre à ses questions sur les médicaments et de le guider s'il présente des symptômes.\n\n"
    
    "CONSIGNES DE COMPORTEMENT :\n"
    "1. Utilisez l'outil 'rechercher_notice_medicament' s'il s'agit d'obtenir des informations spécifiques (posologie, effets indésirables, contre-indications) sur les médicaments indexés.\n"
    "2. Si l'utilisateur vous salue, vous décrit des symptômes ou vous pose des questions médicales générales (ex: le paludisme / palu), comportez-vous comme un médecin bienveillant. "
    "Fournissez-lui des explications claires, instructives et fiables, et conseillez-lui de consulter un professionnel de santé en cabinet.\n"
    "3. Si vous utilisez des informations issues de l'outil 'rechercher_notice_medicament', indiquez clairement de quel médicament et de quelle rubrique provient chaque information clé présentée (ex: [Notice Doliprane - Posologie]).\n"
    "4. Vous devez obligatoirement ajouter EXACTEMENT la mention légale suivante à la toute fin de votre réponse, sur une nouvelle ligne :\n"
    "   \"**Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien.**\"\n"
)

# Instancier l'Agent ADK officiel
root_agent = Agent(
    name="medical_rag_agent",
    model="gemini-3-flash-preview",
    instruction=INSTRUCTIONS,
    tools=[FunctionTool(func=rechercher_notice_medicament)]
)
