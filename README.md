## Book Recommendation System (Streamlit + Sentence-Transformers)

A minimal semantic similarity demo that recommends books based on embeddings from `all-MiniLM-L6-v2`. The UI is built with Streamlit. Pick a book, and the app suggests similar titles using cosine similarity over sentence embeddings.

## Features

- **Interactive UI**: built with Streamlit
- **Semantic search**: `sentence-transformers` + `all-MiniLM-L6-v2`
- **Similarity metric**: `scikit-learn` cosine similarity

## Prerequisites

- Python 3.10–3.12
- macOS/Linux/Windows

## Setup

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Upgrade pip and install dependencies
python -m pip install -U pip wheel
pip install streamlit sentence-transformers scikit-learn "numpy<2"
# Optional: faster file watching for Streamlit (macOS)
# xcode-select --install
# pip install watchdog
```

## Run the app

```bash
source .venv/bin/activate
streamlit run rag_streamlit.py
```

- The first run downloads the model (~100 MB).
- Streamlit will print a URL like `http://localhost:8501` (or another available port). Open it in your browser.

## Troubleshooting

- **NumPy 2.x compatibility**: If you see errors like “A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x” or “RuntimeError: Numpy is not available”, pin NumPy 1.x:

```bash
pip install "numpy<2"
```

- **Port already in use**: Run Streamlit on a different port:

```bash
streamlit run rag_streamlit.py --server.port 8502
```

- **Slow first run**: Model download happens once and is cached.
