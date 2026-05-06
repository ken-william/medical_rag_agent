#!/usr/bin/env python3
"""
Système RAG interactif (Sujet B - Assistant Médicaments).
Permet de poser des questions sur les médicaments cibles,
effectue une recherche vectorielle sur l'index FAISS et
génère une réponse sécurisée avec Groq.
"""

import os
import json
import sys
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

import warnings
import logging

# Supprimer silencieusement les avertissements techniques en arrière-plan
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configuration
INDEX_PATH = Path("faiss_index.bin")
METADATA_PATH = Path("metadata.json")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "llama-3.1-8b-instant"  # Rapide et adapté au RAG

# Charger le fichier .env
load_dotenv()

def verify_api_key() -> str:
    """Vérifie la présence de la clé API Groq et demande à l'utilisateur si absente."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n🔑 Clé API Groq manquante !")
        print("Vous pouvez la générer gratuitement sur https://console.groq.com/")
        api_key = input("Veuillez coller votre GROQ_API_KEY : ").strip()
        if not api_key:
            print("Erreur : Une clé API valide est obligatoire pour utiliser le RAG.")
            sys.exit(1)
        
        # Sauvegarder dans .env pour les futurs lancements
        with open(".env", "a", encoding="utf-8") as f:
            f.write(f"\nGROQ_API_KEY={api_key}\n")
        print("✓ Clé enregistrée dans le fichier .env\n")
        
        # Recharger les variables d'environnement
        load_dotenv()
    return api_key

def load_rag_system():
    """Charge l'index FAISS, les métadonnées et le modèle d'embedding."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        print("\n❌ Base vectorielle introuvable !")
        print("Veuillez d'abord exécuter le script d'indexation : python3 indexation.py")
        sys.exit(1)

    print("Chargement de la base vectorielle FAISS...")
    index = faiss.read_index(str(INDEX_PATH))
    
    print("Chargement des métadonnées...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks_with_meta = json.load(f)
        
    print(f"Chargement du modèle d'embedding ({MODEL_NAME})...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    return index, chunks_with_meta, embedding_model

def rechercher(question: str, embedding_model, index, chunks_with_meta: list, k: int = 4) -> list[dict]:
    """
    Recherche les k chunks les plus proches de la question dans l'index FAISS.
    """
    # Calculer l'embedding de la question
    query_vector = embedding_model.encode([question], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype=np.float32)
    
    # Effectuer la recherche L2 dans FAISS
    # distances: tableau numpy (1, k) avec les distances L2 au carré (plus petit = plus proche)
    # indices: tableau numpy (1, k) contenant les indices des chunks correspondants
    distances, indices = index.search(query_vector, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(chunks_with_meta):
            continue
        
        chunk = chunks_with_meta[idx]
        results.append({
            "id": chunk["id"],
            "contenu_original": chunk["contenu_original"],
            "contenu_enrichi": chunk["contenu_enrichi"],
            "metadata": chunk["metadata"],
            "distance": float(dist)
        })
        
    return results

def reformuler_question(client: Groq, question: str, historique: list) -> str:
    """
    Bonus C : Reformule la question actuelle en utilisant l'historique des échanges
    pour la rendre autonome et optimale pour la recherche vectorielle.
    """
    if not historique:
        return question
        
    prompt_reformulation = (
        "Vous êtes un assistant chargé de reformuler les questions de l'utilisateur.\n"
        "En vous basant sur l'historique des échanges et la nouvelle question, reformulez une question de recherche autonome, "
        "précise, contenant tous les noms de médicaments et mots-clés nécessaires pour faire une recherche dans une base de données.\n"
        "IMPORTANT : Retournez UNIQUEMENT la question reformulée, sans aucun autre texte d'introduction ou de conclusion.\n\n"
        "Historique des échanges :\n"
    )
    for q, r in historique[-3:]:  # Garder les 3 derniers échanges
        prompt_reformulation += f"Utilisateur : {q}\nAssistant : (Réponse résumée)\n"
    prompt_reformulation += f"Nouvelle question : {question}\nQuestion reformulée : "
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt_reformulation}],
            temperature=0.0
        )
        reformulation = response.choices[0].message.content.strip()
        # Nettoyer d'éventuelles guillemets entourant la réponse
        reformulation = re.sub(r'^["\']|["\']$', '', reformulation)
        return reformulation
    except Exception as e:
        # En cas d'erreur, retourner la question originale par sécurité
        return question

def generer_reponse(client: Groq, question: str, chunks_pertinents: list[dict], historique: list) -> str:
    """
    Génère la réponse finale à l'aide de Groq en agissant comme un médecin chaleureux et professionnel,
    en se basant sur les notices si disponibles, ou sur des connaissances générales si absentes.
    """
    # Assembler le contexte s'il y a des notices
    if chunks_pertinents:
        contexte_elements = []
        for idx, chunk in enumerate(chunks_pertinents):
            meta = chunk["metadata"]
            contexte_elements.append(
                f"Source #{idx+1} - Médicament: {meta['medicament']} ({meta['denomination']}) | "
                f"Rubrique: {meta['section_label']} (Mise à jour: {meta['date_mise_a_jour']})\n"
                f"Texte source: {chunk['contenu_original']}\n"
            )
        contexte_text = "\n---\n".join(contexte_elements)
    else:
        contexte_text = "Aucune notice locale pertinente trouvée dans la base de connaissances pour cette question."
    
    prompt_systeme = (
        "Vous êtes un médecin expert et un assistant d'information médicale chaleureux, humain et très professionnel.\n"
        "Votre rôle est d'accueillir l'utilisateur, de dialoguer courtoisement avec lui, de répondre à ses questions sur les médicaments et de le guider s'il présente des symptômes.\n\n"
        
        "CONSIGNES STRICTES DE COMPORTEMENT :\n"
        "1. Si le contexte contient des extraits de notices sémantiquement proches (indiqués par des sources valides), "
        "répondez de manière extrêmement factuelle en vous basant sur ces extraits officiels et en citant explicitement les notices "
        "(ex: [Notice Doliprane - Posologie] ou 'Selon la notice officielle du Doliprane...').\n"
        "2. Si l'utilisateur vous salue, vous parle de sa santé en général, vous décrit des maladies (ex: le paludisme / palu) ou des symptômes "
        "pour lesquels il n'y a pas de notice correspondante dans le contexte, agissez comme un médecin bienveillant. "
        "Fournissez-lui des explications médicales claires, instructives et générales (ex: causes du paludisme, traitements généraux recommandés sur les sites de santé, comportement à adopter), "
        "et conseillez-lui chaleureusement de consulter un médecin.\n"
        "3. Vous devez obligatoirement ajouter EXACTEMENT la mention suivante à la toute fin de votre réponse, sur sa propre ligne :\n"
        "   \"**Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien.**\"\n"
        "4. Soyez toujours bienveillant, rassurant, précis et rigoureux. Ne dites jamais 'Je ne trouve pas l'information dans ma base' pour de simples salutations ou pour des questions médicales générales sur des maladies; expliquez la maladie sémantiquement en agissant comme un médecin traitant."
    )
    
    messages = [{"role": "system", "content": prompt_systeme}]
    
    # Ajouter l'historique des 3 derniers échanges
    for q, r in historique[-3:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": r})
        
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.25  # Température légèrement plus élevée pour un dialogue fluide et naturel
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Une erreur est survenue avec l'API Groq : {str(e)}"

import re

def main():
    print("=" * 60)
    print("   💊 ASSISTANT MÉDICAMENTS RAG - BIENVENUE 💊   ")
    print("=" * 60)
    
    # Vérifier et charger la clé API Groq
    api_key = verify_api_key()
    client = Groq(api_key=api_key)
    
    # Charger les fichiers FAISS et métadonnées
    index, chunks_with_meta, embedding_model = load_rag_system()
    
    print("\n■ Système RAG prêt. Vous pouvez poser vos questions.")
    print("Le système gère l'historique et reformule vos questions si nécessaire.")
    print("Tapez 'quit' ou 'exit' pour quitter.\n")
    
    # Historique de la conversation sous forme de liste de tuples (question, reponse)
    historique = []
    
    while True:
        try:
            question = input("\n👤 Votre question : ").strip()
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Au revoir et prenez soin de vous !")
                break
                
            # 1. Reformulation si historique existant (Bonus C)
            question_recherche = question
            if historique:
                question_recherche = reformuler_question(client, question, historique)
                if question_recherche != question:
                    print(f"🔍 (Recherche optimisée : \"{question_recherche}\")")
            
            # 2. Recherche vectorielle
            chunks_pertinents = rechercher(question_recherche, embedding_model, index, chunks_with_meta, k=4)
            
            # Filtrer les chunks sémantiquement proches
            chunks_valides = [c for c in chunks_pertinents if c["distance"] <= 24.0]
            
            # 3. Génération de la réponse avec Groq
            reponse = generer_reponse(client, question, chunks_valides, historique)
            
            # 4. Affichage du résultat
            print("\n🤖 Réponse :")
            print("-" * 50)
            print(reponse)
            print("-" * 50)
            
            # Afficher les sources sémantiques
            print("\n📄 Sources consultées :")
            seen_sources = set()
            for chunk in chunks_valides:
                meta = chunk["metadata"]
                source_id = f"{meta['medicament']} - {meta['section_label']}"
                if source_id not in seen_sources:
                    seen_sources.add(source_id)
                    print(f" - Notice de {meta['denomination']} (Rubrique : {meta['section_label']}) [Score L2: {chunk['distance']:.2f}]")
            
            # Mettre à jour l'historique
            historique.append((question, reponse))
            
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"\n❌ Une erreur inattendue est survenue : {str(e)}")

if __name__ == "__main__":
    main()
