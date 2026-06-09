# Pulsar 🔬

**P**ubMed **U**nsupervised **L**iterature **S**ummary **A**nd **R**eview

Pulsar is an automated scientific literature digest that fetches the latest PubMed papers, clusters them into research topics using unsupervised ML, extracts biomedical entities via NER, and delivers a weekly HTML email summary — powered by BERTopic, BioBERT, weak supervision stance detection, and LLM summarization.

---

## Pipeline Overview

```
PubMed (Entrez API)
    ↓
Fetch & parse abstracts (last 30 days)
    ↓
Score & filter by journal impact factor + recency
    ↓
BERTopic clustering (UMAP + HDBSCAN + zero-shot labeling)
    ↓
BioBERT NER per cluster (chemicals, diseases, genes, proteins, cell types)
    ↓
LLM summarization (Gemini API)
    ↓
HTML email digest (SMTP)
```

---

## ML/NLP Components

| Component | Method | Details |
|---|---|---|
| Topic clustering | BERTopic | UMAP dimensionality reduction + HDBSCAN density clustering |
| Topic labeling | Zero-shot classification | `facebook/bart-large-mnli` with curated scientific domain labels |
| Chemical NER | BioBERT fine-tuned | `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M` |
| Disease NER | BioBERT fine-tuned | `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M` |
| Gene/Protein NER | BioBERT fine-tuned | `pruas/BENT-PubMedBERT-NER-Gene` |
| Cell type/line NER | BioBERT fine-tuned | `siddharthtumre/biobert-finetuned-ner` |
| Summarization | Gemini API | Structured prompts with NER entities + stance breakdown |

---

## Setup

**Python 3.11 required**

```bash
# 1. create conda environment
conda create -n pulsar python=3.11
conda activate pulsar

# 2. install dependencies
pip install -r requirements.txt

# 3. configure environment variables
cp .env.example .env

# 4. verify setup
python3 setup.py
```

### Environment Variables

```bash
# .env
GEMINI_API_KEY=your_key_here        # free at https://aistudio.google.com
EMAIL_ADDRESS=your_gmail@gmail.com
APP_PASSWORD=your_gmail_app_password # see Gmail setup below
```

### Getting Your API Keys

**Gemini API (free):**
1. Go to https://aistudio.google.com
2. Click "Get API Key"
3. Copy into `.env` as `GEMINI_API_KEY`

**Gmail App Password:**
1. Go to `myaccount.google.com` → Security
2. Enable 2-Step Verification
3. Search "App Passwords" → Generate for Mail
4. Copy 16-character password into `.env` as `APP_PASSWORD`

---

## Running Pulsar

```bash
conda activate pulsar
cd /path/to/pulsar
python3 main.py
```

This runs the full pipeline and sends the digest to your configured email address. With ~500 papers and 30 topic clusters, a full run takes approximately 15-20 minutes.

### Running Individual Steps

```bash
# topic modeling only
python3 -m src.models.topic_model

# NER only
python3 -m src.models.ner_model

# summarization only
python3 -m src.summarization.llm_summarization

# email delivery only
python3 -m src.delivery.email_delivery
```

---

## Dependencies

Key libraries:
- `bertopic` — topic modeling pipeline
- `sentence-transformers` — text embeddings
- `transformers` — BioBERT NER + PubMedBERT stance classification
- `biopython` — Entrez/PubMed API wrapper
- `impact-factor` — journal impact factor lookup
- `google-genai` — Gemini API summarization
- `umap-learn` — dimensionality reduction
- `hdbscan` — density-based clustering
- `gensim` — topic coherence evaluation
- `scikit-learn` — evaluation metrics

---

## License

MIT — see [LICENSE](LICENSE)