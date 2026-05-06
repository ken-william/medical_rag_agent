#!/usr/bin/env python3
"""
Interface Web Streamlit de l'Assistant Médicaments RAG.
Design inspiré de Gemini : épuré, moderne, intuitif et sans distraction.
Intègre un comportement de médecin bienveillant capable de dialoguer courtoisement,
de conseiller sur des maladies ou symptômes en utilisant des connaissances médicales générales
quand aucune notice locale n'est disponible, tout en citant rigoureusement les notices locales
quand elles sont trouvées.
"""

import os
import json
from pathlib import Path
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import requests
import warnings
import logging

# Supprimer silencieusement les avertissements techniques en arrière-plan
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configurer la page Streamlit
st.set_page_config(
    page_title="Assistant Médicaments",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour répliquer EXACTEMENT l'interface officielle Gemini en MODE CLAIR
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #0f172a;
    }
    
    /* Fond dégradé blanc lumineux officiel de Gemini */
    .stApp {
        background: radial-gradient(circle at 50% 30%, #ffffff 0%, #f0f2f5 100%) !important;
    }
    
    /* En-tête dégradé "Sparkle" officiel (Bleu/Violet/Rose) */
    .gemini-header {
        background: linear-gradient(120deg, #4285f4 0%, #9b72cb 45%, #d96570 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
        filter: drop-shadow(0px 4px 20px rgba(155, 114, 203, 0.15));
    }
    
    .gemini-subheader {
        color: #4b5563;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Bulles de discussion translucides officielles de Gemini en Mode Clair */
    .stChatMessage:has([data-testid="user-avatar"]) {
        background: rgba(66, 133, 244, 0.06) !important;
        border: 1px solid rgba(66, 133, 244, 0.15) !important;
        border-radius: 22px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.03);
        backdrop-filter: blur(16px);
        color: #0f172a !important;
    }
    
    .stChatMessage:has([data-testid="assistant-avatar"]) {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-radius: 22px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.03);
        backdrop-filter: blur(16px);
        color: #0f172a !important;
    }
    
    /* Force la couleur noire de la police dans les messages Streamlit */
    .stChatMessage p, .stChatMessage li, .stChatMessage span {
        color: #0f172a !important;
    }
    
    /* Style du champ de saisie en forme de pilule claire officielle */
    div[data-testid="stChatInput"] {
        border-radius: 32px !important;
        background-color: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        padding: 0.4rem 1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #0f172a !important;
        font-size: 1rem !important;
    }
    
    /* Style de la barre latérale claire officielle de Gemini */
    section[data-testid="stSidebar"] {
        background-color: #f3f4f6 !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    /* Avertissement médical en bas style carte de contrôle épurée claire */
    .medical-disclaimer {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 1.2rem;
        border-radius: 20px;
        margin-top: 3.5rem;
        color: #4b5563;
        font-size: 0.88rem;
        text-align: center;
        line-height: 1.5;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    
    /* Cartes de sources dégradées claires */
    .clean-source-card {
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.04) 0%, rgba(155, 114, 203, 0.04) 100%);
        border: 1px solid rgba(0, 0, 0, 0.05);
        padding: 0.9rem;
        border-radius: 12px;
        margin-top: 0.6rem;
        font-size: 0.88rem;
        transition: all 0.3s ease;
        color: #4b5563 !important;
    }
    
    .clean-source-card:hover {
        border-color: rgba(155, 114, 203, 0.3);
        transform: scale(1.01);
        box-shadow: 0 0 15px rgba(155, 114, 203, 0.08);
    }
    
    /* Animations et conteneur du faisceau lumineux Gemini Mode Clair */
    @keyframes geminiPulse {
        0% { background-position: 0% 50%; opacity: 0.6; }
        50% { background-position: 100% 50%; opacity: 1; }
        100% { background-position: 0% 50%; opacity: 0.6; }
    }

    .gemini-loader-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 600px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(16px);
    }

    .gemini-loader-bar {
        width: 100%;
        height: 4px;
        border-radius: 4px;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570, #4285f4);
        background-size: 400% 400%;
        animation: geminiPulse 2.5s ease infinite;
        margin-bottom: 1.2rem;
    }

    .gemini-loader-text {
        color: #4b5563;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.03em;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Charger le fichier .env automatiquement
load_dotenv()

def rechercher_api_medicament(nom: str) -> dict:
    """
    Interroge l'API publique en temps réel pour obtenir la composition
    et les informations AMM d'un médicament si la notice locale est absente.
    """
    try:
        url = f"https://medicaments-api.giygas.dev/medicament/{nom.strip().lower()}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            records = response.json()
            if records:
                # Retourner le premier enregistrement trouvé
                return records[0]
    except Exception:
        pass
    return None

# Charger la clé API Groq silencieusement
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Clé API GROQ_API_KEY introuvable dans le fichier .env. Veuillez configurer votre environnement.")
    st.stop()

# Initialiser le client Groq de manière invisible
groq_client = Groq(api_key=api_key)

# Initialiser l'historique des conversations
if "historique" not in st.session_state:
    st.session_state.historique = []

# Chemins et constantes locales invisibles pour l'utilisateur
INDEX_PATH = Path("faiss_index.bin")
METADATA_PATH = Path("metadata.json")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
K_VALUE = 4
THRESHOLD_L2 = 24.0

@st.cache_resource(show_spinner=False)
def load_rag_resources():
    """Charge et met en cache l'index FAISS, les métadonnées et le modèle d'embedding."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model

# Chargement premium unique lors du premier accès avec le chargeur sémantique style Gemini
if "resources_loaded" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="gemini-loader-container">
        <div class="gemini-loader-bar"></div>
        <div class="gemini-loader-text">Chargement de la base de connaissances médicale...</div>
    </div>
    """, unsafe_allow_html=True)
    index, metadata, embedding_model = load_rag_resources()
    loading_placeholder.empty()
    st.session_state.resources_loaded = True
else:
    index, metadata, embedding_model = load_rag_resources()

# Si l'index FAISS n'est pas encore généré
if index is None:
    st.error("❌ Base vectorielle introuvable !")
    st.info("Veuillez d'abord exécuter le script d'indexation pour initialiser la base de données sémantique : `python3 indexation.py`")
    st.stop()

# Barre latérale minimaliste (Uniquement pour effacer la conversation)
with st.sidebar:
    st.markdown("### 💬 Options")
    if st.button("🧹 Nouvelle conversation", use_container_width=True):
        st.session_state.historique = []
        st.rerun()

# Titre style Gemini
st.markdown('<h1 class="gemini-header">💊 Assistant Médicaments</h1>', unsafe_allow_html=True)
st.markdown('<p class="gemini-subheader">Posez vos questions sur les médicaments, maladies, posologies ou effets secondaires.</p>', unsafe_allow_html=True)

# Afficher la discussion existante
for q, r, sources in st.session_state.historique:
    with st.chat_message("user", avatar="👤"):
        st.markdown(q)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(r)
        # Affichage très discret des sources si applicables
        if sources and len(sources) > 0:
            with st.expander("🔍 Sources consultées"):
                for src in sources:
                    st.markdown(f"""
                    <div class="clean-source-card">
                        <strong>Notice : {src['medicament']}</strong> - Rubrique : <em>{src['section_label']}</em>
                    </div>
                    """, unsafe_allow_html=True)

# Saisie de l'utilisateur
if prompt := st.chat_input("Comment puis-je vous aider aujourd'hui ?"):
    
    # Afficher la question
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    # Reformulation contextuelle invisible à l'aide de l'historique (Bonus C)
    question_recherche = prompt
    if st.session_state.historique:
        prompt_reformulation = (
            "Vous êtes un assistant chargé de reformuler les questions de l'utilisateur.\n"
            "En vous basant sur l'historique des échanges et la nouvelle question, reformulez une question de recherche autonome, "
            "précise, contenant tous les noms de médicaments et mots-clés nécessaires pour faire une recherche dans une base de données.\n"
            "IMPORTANT : Retournez UNIQUEMENT la question reformulée, sans aucun autre texte d'introduction ou de conclusion.\n\n"
            "Historique des échanges :\n"
        )
        for q_prev, r_prev, _ in st.session_state.historique[-2:]:
            prompt_reformulation += f"Utilisateur : {q_prev}\nAssistant : (Réponse résumée)\n"
        prompt_reformulation += f"Nouvelle question : {prompt}\nQuestion reformulée : "
        
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt_reformulation}],
                temperature=0.0
            )
            question_recherche = response.choices[0].message.content.strip().strip('"\'')
        except Exception:
            question_recherche = prompt

    # Recherche vectorielle FAISS sémantique
    query_vector = embedding_model.encode([question_recherche], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype=np.float32)
    distances, indices = index.search(query_vector, K_VALUE)
    
    chunks_pertinents = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            chunk = metadata[idx]
            # On n'ajoute que si la pertinence L2 est bonne (<= THRESHOLD_L2)
            if dist <= THRESHOLD_L2:
                chunks_pertinents.append({
                    "contenu_original": chunk["contenu_original"],
                    "metadata": chunk["metadata"],
                    "distance": float(dist)
                })
            
    # Générer la réponse avec Groq
    with st.chat_message("assistant", avatar="🤖"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="margin-bottom: 1rem; width: 100%;">
            <div class="gemini-loader-bar"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assembler le contexte si notices locales trouvées, sinon interroger l'API nationale
        api_record = None
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
            # Tenter une recherche sur l'API publique en temps réel
            api_record = rechercher_api_medicament(question_recherche)
            if api_record:
                comp_list = []
                for comp in api_record.get("composition", []):
                    comp_list.append(f"- Substance active: {comp.get('denominationSubstance')} ({comp.get('dosage')} par {comp.get('referenceDosage')})")
                compositions = "\n".join(comp_list)
                
                contexte_text = (
                    f"Source API en temps réel - Médicament: {api_record.get('elementPharmaceutique')} ({api_record.get('formePharmaceutique')})\n"
                    f"Titulaire AMM: {api_record.get('titulaire')} (AMM délivrée le {api_record.get('dateAMM')})\n"
                    f"Composition officielle :\n{compositions}\n"
                    f"Conditions de prescription et de délivrance: {', '.join(api_record.get('conditions', []))}\n"
                )
            else:
                contexte_text = "Aucune notice locale pertinente ni données API en temps réel trouvées pour cette question."
            
        prompt_systeme = (
            "Vous êtes un médecin expert et un assistant d'information médicale chaleureux, humain et très professionnel.\n"
            "Votre rôle est d'accueillir l'utilisateur, de dialoguer courtoisement avec lui, de répondre à ses questions sur les médicaments et de le guider s'il présente des symptômes.\n\n"
            
            "CONSIGNES STRICTES DE COMPORTEMENT :\n"
            "1. Si le contexte contient des extraits de notices sémantiquement proches (indiqués par des sources locales) ou des informations issues de l'API nationale en temps réel, répondez de manière extrêmement factuelle en vous basant sur ces extraits officiels et en citant les sources (ex: [Notice Doliprane - Posologie] ou [API ANSM - Composition]).\n"
            "2. Si l'utilisateur vous salue, vous décrit des symptômes ou vous pose des questions médicales générales (ex: le paludisme / palu) ou si aucune donnée officielle n'est trouvée pour un médicament, agissez comme un médecin bienveillant. "
            "Fournissez-lui des explications médicales claires, instructives et générales (ex: causes du paludisme, traitements généraux recommandés sur les sites de santé, comportement à adopter), "
            "et conseillez-lui chaleureusement de consulter un médecin.\n"
            "3. Vous devez obligatoirement ajouter EXACTEMENT la mention suivante à la toute fin de votre réponse, sur sa propre ligne :\n"
            "   \"**Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien.**\"\n"
            "4. Soyez toujours bienveillant, rassurant, précis et rigoureux. Ne dites jamais 'Je ne trouve pas l'information dans ma base' pour de simples salutations ou pour des questions médicales générales sur des maladies; expliquez la maladie sémantiquement en agissant comme un médecin traitant."
        )
        
        messages = [{"role": "system", "content": prompt_systeme}]
        for q_prev, r_prev, _ in st.session_state.historique[-3:]:  # Plus d'historique pour une conversation fluide
            messages.append({"role": "user", "content": q_prev})
            messages.append({"role": "assistant", "content": r_prev})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.25  # Température légèrement plus élevée pour un dialogue fluide et naturel
            )
            reponse = response.choices[0].message.content.strip()
            loading_placeholder.empty()  # Effacer la barre de chargement
            st.markdown(reponse)
            
            # Enregistrer les sources de manière épurée s'il y a lieu
            sources_affichees = []
            seen = set()
            for chunk in chunks_pertinents:
                meta = chunk["metadata"]
                sid = f"{meta['medicament']} - {meta['section_label']}"
                if sid not in seen:
                    seen.add(sid)
                    sources_affichees.append({
                        "medicament": meta["medicament"],
                        "section_label": meta["section_label"]
                    })
            
            # Enregistrer la source de l'API en temps réel s'il y en a une
            if api_record and not chunks_pertinents:
                sources_affichees.append({
                    "medicament": api_record.get("elementPharmaceutique"),
                    "section_label": "Base API Nationale Temps Réel"
                })
                    
            st.session_state.historique.append((prompt, reponse, sources_affichees))
            st.rerun()
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Une erreur est survenue : {str(e)}")

# Avertissement fixe de responsabilité médicale en bas de page
st.markdown("""
<div class="medical-disclaimer">
    Ces informations proviennent de la Base de Données Publique des Médicaments (ANSM) et de connaissances médicales générales de santé publique. Elles sont fournies à titre d'information générale et ne doivent pas être considérées comme des avis médicaux personnalisés. <strong>Consultez systématiquement un professionnel de santé pour tout avis médical ou décision thérapeutique.</strong>
</div>
""", unsafe_allow_html=True)
