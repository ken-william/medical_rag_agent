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

# Configuration
INDEX_PATH = Path("faiss_index.bin")
METADATA_PATH = Path("metadata.json")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "llama3-8b-8192"  # Rapide et adapté au RAG

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
    Génère la réponse finale à l'aide de Groq en injectant les chunks comme contexte
    et en appliquant scrupuleusement les consignes de sécurité médicale.
    """
    # Assembler le contexte textuel pour le LLM
    contexte_elements = []
    for idx, chunk in enumerate(chunks_pertinents):
        meta = chunk["metadata"]
        contexte_elements.append(
            f"Source #{idx+1} - Médicament: {meta['medicament']} ({meta['denomination']}) | "
            f"Rubrique: {meta['section_label']} (Mise à jour: {meta['date_mise_a_jour']})\n"
            f"Texte source: {chunk['contenu_original']}\n"
        )
    contexte_text = "\n---\n".join(contexte_elements)
    
    prompt_systeme = (
        "Vous êtes un assistant médical expert d'information sur les médicaments.\n"
        "Votre tâche est de répondre de manière précise, rigoureuse et claire à la question posée par l'utilisateur, "
        "en utilisant UNIQUEMENT les sources textuelles fournies dans le contexte ci-dessous.\n\n"
        
        "RÈGLES ABSOLUES À RESPECTER :\n"
        "1. Vous devez obligatoirement inclure EXACTEMENT cette mention à la toute fin de votre réponse, sur sa propre ligne :\n"
        "   \"**Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien.**\"\n"
        "2. Vous devez indiquer clairement de quel médicament et de quelle rubrique provient chaque information clé présentée dans votre réponse "
        "(ex: [Notice Doliprane - Posologie] ou 'Selon la notice officielle du Doliprane (rubrique Posologie)...').\n"
        "3. Si l'information demandée n'est pas dans les sources fournies ou si la question dépasse le périmètre des médicaments indexés, "
        "vous devez obligatoirement déclarer explicitement : \"Je ne trouve pas cette information dans ma base de connaissances.\" "
        "Ne tentez jamais d'inventer, d'extrapoler ou d'utiliser des connaissances externes.\n"
        "4. Soyez courtois, neutre et professionnel.\n\n"
        
        f"Contexte des notices officielles :\n{contexte_text}"
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
            temperature=0.1  # Faible température pour maximiser la fidélité au contexte
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
            
            if not chunks_pertinents:
                print("⚠️ Aucun résultat trouvé dans la base de données.")
                continue
                
            # Bonus B : Évaluation du score de confiance (distance L2)
            # Pour paraphrase-multilingual-mpnet-base-v2 avec distance FlatL2 :
            # - distance < 12-15 : excellente similarité sémantique.
            # - distance > 25-30 : très éloigné, probablement hors sujet.
            best_distance = chunks_pertinents[0]["distance"]
            if best_distance > 24.0:
                print("\n⚠️ [Note] La question semble hors de ma base de connaissances des médicaments courants.")
                print("Je ne trouve pas cette information dans ma base de connaissances.")
                print("\n**Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien.**")
                continue
            
            # 3. Génération de la réponse avec Groq
            reponse = generer_reponse(client, question, chunks_pertinents, historique)
            
            # 4. Affichage du résultat
            print("\n🤖 Réponse :")
            print("-" * 50)
            print(reponse)
            print("-" * 50)
            
            # Afficher les sources sémantiques
            print("\n📄 Sources consultées :")
            seen_sources = set()
            for chunk in chunks_pertinents:
                meta = chunk["metadata"]
                source_id = f"{meta['medicament']} - {meta['section_label']}"
                if source_id not in seen_sources and chunk["distance"] <= 24.0:
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
