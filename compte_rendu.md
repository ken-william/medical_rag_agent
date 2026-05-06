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
Dans une boucle interactive CLI, les utilisateurs posent souvent des questions de suivi incomplètes sémantiquement (ex: *"Quels sont ses effets ?"* après avoir parlé du Doliprane).  
**Décision** : Utiliser l'API Groq pour reformuler à la volée la question de l'utilisateur en fonction de l'historique des 3 derniers échanges avant d'interroger FAISS.

### 4. Seuil de Confiance (Bonus B)
Pour éviter les hallucinations lorsque l'utilisateur pose des questions hors du corpus, nous mesurons la distance L2 fournie par FAISS.  
**Décision** : Si la distance L2 minimale est supérieure à `24.0`, le système refuse de répondre sémantiquement et affiche une mention neutre de sécurité.

---

## ⚠️ Difficultés Rencontrées & Solutions

1. **Encodage des notices (`\x92`)** : Les notices contenaient de nombreux caractères spéciaux mal encodés lors de l'extraction HTML originelle (ex. `l\x92adulte`).  
   *Solution* : Création d'un utilitaire de nettoyage `clean_text` pour normaliser les caractères en apostrophes standards.
2. **Temps de calcul sur CPU** : L'indexation initiale prenait trop de temps en raison de la redondance extrême.  
   *Solution* : Implémentation du regroupement et de la déduplication par molécule canonique.
3. **Absence de clé API au premier démarrage** : Empêche le RAG de s'exécuter correctement.  
   *Solution* : Interface d'accueil intelligente qui demande interactivement la clé, la valide et l'enregistre automatiquement dans le fichier `.env`.
