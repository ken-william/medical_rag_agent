# 📝 Compte-Rendu de Conception - Assistant RAG Médicaments

## 🚀 Décisions de Conception

### 1. Stratégie de Dédoublonnement du Corpus
La base de données d'origine (`CIS_RCP_export.xlsx`) contient 15 649 entrées. Une recherche sémantique naïve sur les 18 médicaments cibles a révélé des centaines de doublons (ex: 129 entrées distinctes pour "Amoxicilline" en raison de dosages, marques de génériques ou conditionnements différents).  
**Décision** : Grouper les entrées par médicament canonique et sélectionner uniquement la ligne contenant la notice la plus longue (la plus complète). Cela a permis de passer de 18 000+ chunks à seulement **842 chunks uniques**, ramenant le temps de calcul des embeddings de **9 minutes à moins de 15 secondes** tout en préservant une richesse d'information maximale et en évitant de polluer la recherche vectorielle avec des doublons identiques.

### 2. Indexation Contextuelle Enrichie
Les embeddings génériques peinent parfois à faire la différence entre des rubriques sémantiquement proches (ex: distinguer les effets indésirables des contre-indications).  
**Décision** : Structurer les chunks vectorisés en préfixant systématiquement le nom du médicament canonique et la rubrique de la notice.  
*Format : `Médicament: [Nom] | Rubrique: [Section] | Contenu: [Texte]`*  
Cette structuration sémantique explicite garantit que les recherches FAISS ciblent précisément les rubriques adaptées.

### 3. Reformulation Dynamique de la Question (Bonus C)
Dans une boucle interactive CLI ou chat Web, les utilisateurs posent souvent des questions de suivi incomplètes sémantiquement (ex: *"Quels sont ses effets ?"* après avoir parlé du Doliprane).  
**Décision** : Utiliser le LLM pour reformuler à la volée la question de l'utilisateur en fonction de l'historique des 2 ou 3 derniers échanges avant d'interroger FAISS.

### 4. Double Mode "RAG / Médecin Conseil" (Bonus B)
Pour éviter les blocages abrupts en cas de questions hors corpus ou sur des symptômes généraux (ex: *"j'ai le palu"*).  
**Décision** : Si la distance L2 minimale est supérieure à `24.0` (hors base), le système passe en mode médecin bienveillant. Il fournit des explications générales fiables, conseille d'aller consulter en cabinet, et appende systématiquement la mention légale. S'il y a des notices locales valides (L2 <= 24.0), il fonde sa réponse rigoureusement sur ces notices officielles et affiche les citations.

---

## ⚠️ Difficultés Rencontrées & Solutions

1. **Encodage des notices (`\x92`)** : Les notices contenaient de nombreux caractères spéciaux mal encodés lors de l'extraction HTML originelle (ex. `l\x92adulte`).  
   *Solution* : Création d'un utilitaire de nettoyage `clean_text` pour normaliser les caractères en apostrophes standards.
2. **Outages des modèles de test (`503 UNAVAILABLE`)** : Le modèle `gemini-3-flash-preview` subissait des surcharges sur la console AI Studio.  
   *Solution* : Utilisation de votre modèle cible **`gemini-3-flash-preview`** et migration de Groq vers stable **`llama-3.1-8b-instant`**, garantissant une disponibilité 24h/24 et une vitesse d'exécution foudroyante.
3. **Conformité de chargement ADK (`No root_agent found`)** : L'ADK renvoyait une erreur de chargement car la variable de l'agent n'était pas conforme.  
   *Solution* : Renommage de la variable d'instanciation de l'agent en **`root_agent`** et mise à jour de `__init__.py`.
4. **Erreur de permission Vertex AI (`aiplatform.endpoints.predict`)** : Le fait de configurer la localisation sémantique sur `global` bloquait les appels prédictifs sur les serveurs régionaux.  
   *Solution* : Mise à jour de la variable `GOOGLE_CLOUD_LOCATION=us-central1` dans le fichier `.env` local.
5. **Erreur d'importation de décorateur ADK (`ImportError: cannot import name 'tool'`)** : Le décorateur `@tool` n'était pas pris en charge par la version ADK active.  
   *Solution* : Utilisation de la classe officielle **`FunctionTool(func=...)`** de `google.adk.tools` pour envelopper nos outils.
