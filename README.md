# AI Security — Embedding Model Identification

Text embeddings are widely used in production systems (vector databases, RAG, semantic search) and are often treated as opaque vectors — safe to store and transmit. This first part of the practical session demonstrates that **even the model used to produce an embedding can be identified from the vector alone**.

---

## Objective

An attacker has stolen embeddings from a target vector database. Before launching any inversion attack, they need to determine which model produced those vectors.

The target database contains a known probe: entry `TEST` encodes the plain text `"test"`. For each candidate model, encode `"test"` and compare against this stolen vector using cosine similarity. A score of 1.0 means a perfect match — that is your model.

---

## Project Structure

```
data/
  sentences_target_text_db.parquet   # target database (target_id, text)
  sentences_target_vector_db.index   # FAISS index — target vectors

01_model_identification.ipynb        # identify the embedding model from a stolen vector
```

---

## Installation

**Requirements** : Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
uv run jupyter notebook
```
