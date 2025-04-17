# 📘 Assistant Éducation – Chatbot RAG

## <h2>🔍 Description</h2>

Application Gradio de question/réponse (RAG) spécialisée dans l’enseignement supérieur au Maroc, basée sur un rapport PDF. Elle utilise **LangChain**, **Groq**, et **Ollama** pour combiner recherche documentaire et génération de texte.

---

## <h2>⚙️ Fonctionnalités</h2>

- Lecture automatique d’un fichier PDF
- Extraire le texte depuis les images via <strong> PaddleOCR</strong>
- Découpage du contenu et génération d’embeddings
- Recherche contextuelle avec LangChain
- Résumés compressés via Groq LLM
- Interface conversationnelle avec Gradio

---

## <h2>📁 Fichiers importants</h2>

- `app.py` : le script principal de l'application
- `requirements.txt` : liste des dépendances
- `pdf/RP.pdf` : le fichier PDF à analyser (à ajouter)
- `image` : le fichier qui contient les images 

---

## <h2>🚀 Installation</h2>

```bash
git clone https://github.com/ton-utilisateur/assistant-education-maroc.git
python -m venv venv
source venv/bin/activate      # ou `venv\Scripts\activate` sur Windows
pip install -r requirements.txt

```
## <h2>🔐 Sécurité</h2>
Remplacez la clé API Groq dans app.py par une variable d’environnement :
```bash
api_key = os.getenv("GROQ_API_KEY")
```

