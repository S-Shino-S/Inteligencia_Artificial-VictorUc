# RAG

A Retrieval-Augmented Generation pipeline you can read line by line
(**no NumPy, no vector-DB service, no LLM API**): embed text into vectors,
chunk and index a corpus, retrieve the nearest chunks by cosine, then build
the prompt and answer with citations. The default corpus is a handful of notes
about the course itself, so every score can be checked by hand.

Embeddings here are plain **bag-of-words counts** and the store is an exact
**k-NN** loop. A production system swaps these for learned embeddings and an
approximate index (HNSW, IVF) plus a real LLM — but the plumbing (vectorize →
cosine → top-k → prompt) is exactly the same.

## Setup

```bash
cd "RAG/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd "RAG/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Later sessions: activate the venv again, then run the programs.
Deactivate with `deactivate`.

## Programs

Each program is one stage of the pipeline.

| File | Stage |
|---|---|
| `01_embed.py` | Text → vector; cosine similarity between examples |
| `02_index.py` | Chunk the documents and build the vector store (offline) |
| `03_retrieve.py` | Embed a query and pull the nearest chunks (k-NN) |
| `04_answer.py` | Full loop: retrieve → prompt → grounded answer / abstain |

```bash
python 01_embed.py
python 02_index.py
python 03_retrieve.py
python 03_retrieve.py --all
python 03_retrieve.py --query "¿qué es un chunk?" --top-k 3
python 03_retrieve.py --query "índices de vecinos" --source rag.pptx
python 04_answer.py
python 04_answer.py --all
python 02_index.py --chunk-words 20 --overlap 4
```

All four default to `data/curso.yaml`. Override with `--data`, `--query`,
`--top-k`, `--chunk-words`, `--overlap`, `--source`, or `--all`. Without
`--query`, programs 03–04 run the first YAML query (`--all` runs every one).

## The four stages

1. **Embed** (`rag/embed.py`) — build a vocabulary from the corpus (minus
   stopwords) and map each text to a vector of word counts.
2. **Index** (`rag/chunk.py`, `rag/store.py`) — split documents into
   overlapping chunks, embed each, and store the vector next to its metadata.
3. **Retrieve** (`rag/retrieve.py`) — embed the query and score it against
   every stored vector with `cos(q, d) = (q · d) / (||q|| ||d||)`.
4. **Answer** (`rag/generate.py`) — assemble the prompt from the top-k chunks
   and quote the sentences that overlap the question, with `[n]` citations.
   Below `min_score`, the system **abstains** instead of inventing.

## Edit the corpus

Open `data/curso.yaml` (or copy it).

```yaml
name: Notas del curso de IA
chunk_words: 30       # words per chunk
overlap: 6            # words shared between neighboring chunks
top_k: 2              # chunks retrieved per query
min_score: 0.05       # minimum cosine before abstaining
stopwords: [el, la, de, que, y, ...]

documents:
  - title: Qué es RAG
    source: rag.pptx
    text: >
      RAG significa generación aumentada por recuperación. El sistema calcula
      el embedding de la pregunta y recupera los chunks más similares...

queries:
  - ¿cómo sabe RAG qué fragmentos enviar al modelo de lenguaje?

examples:               # program 01 only: texts to compare pairwise
  - el coseno mide la similitud entre vectores
  - la similitud del coseno compara vectores guardados
  - cómo se cocina una paella valenciana
```

Each `documents` entry needs `text`; `title` and `source` are metadata
(`source` also drives the `--source` filter). `stopwords` are dropped before
counting, so they never enter a vector.

## What to expect

**`01_embed.py`** — the first two examples share words
(`coseno`, `similitud`, `vectores`), so their cosine is **0.67**; the paella
text shares none, so it maps to the empty vector and scores **0.00**.

**`02_index.py`** — six documents become **12 chunks** (each long document
splits in two, with a 6-word overlap). Every chunk becomes one vector with its
`source` metadata.

**`03_retrieve.py`** — each course question retrieves the on-topic chunk with
the highest cosine (≈ 0.43–0.45). The paella question scores **0.000** against
everything: nothing in the corpus is relevant.

**`04_answer.py`** — the three course questions return grounded answers with
`[n]` citations pointing at the retrieved chunks. The paella question falls
below `min_score`, so the system **abstains** ("no tengo evidencia
suficiente") instead of making something up.

A production RAG is this loop at scale: the same cosine and the same top-k,
with learned embeddings, an approximate index, and an LLM writing the final
answer from the same retrieved context. This project is the plumbing, not the
wrapper.
