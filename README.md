# 🔍 Extended DSA - Text Deduplication & Semantic Search

A comprehensive toolkit for **text deduplication** and **semantic similarity search** using multiple techniques: **FAISS**, **SimHash**, and **MinHash**. Features a web-based Gradio interface for easy interaction.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-yellow)


---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Benchmarks](#-benchmarks)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## ✨ Features

- **Multiple Deduplication Methods**
  - 🎯 **FAISS** - Exact cosine similarity search (Facebook AI Similarity Search)
  - ⚡ **SimHash** - Fast approximate search using random hyperplanes + LSH
  - 🔗 **MinHash** - Approximate search using locality-sensitive hashing

- **Embedding Models Support**
  - `all-MiniLM-L6-v2` - Fast, general-purpose (384 dimensions)
  - `bge-base-en-v1.5` - High accuracy (768 dimensions)
  - `e5-base-v2` - Balanced performance
  - `instructor-base` - Instruction-following embeddings

- **Web Interface**
  - Upload Excel datasets (`.xlsx`)
  - Automatic text reformatting to fit model token limits
  - Interactive deduplication with configurable thresholds
  - Real-time semantic search on deduplicated data
  - Performance metrics (time, memory usage)
  - Export results to Excel

---
## ▶️ Try the Web App on Hugging Face Spaces
🔗 https://huggingface.co/spaces/Namiek/Deduplication

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Web Interface                     │
├─────────────────────────────────────────────────────────────────┤
│  📁 Load Dataset  │  ⚡ Deduplication  │  🔍 Search              │
└────────┬──────────┴────────┬───────────┴────────┬───────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
┌─────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│ Text Processing │  │ Search Techniques │  │ Query Processing │
│ - Tokenization  │  │ - FaissTechnique  │  │ - Encode query   │
│ - Chunking      │  │ - SimHashTechnique│  │ - Top-k search   │
│ - Embedding     │  │ - MinHashTechnique│  │ - Ranking        │
└────────┬────────┘  └────────┬──────────┘  └────────┬─────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Sentence Transformers                         │
│            (all-MiniLM-L6-v2 / bge-base-en-v1.5)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- GPU optional (CUDA support for faster embeddings)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kriss-Nevile/Extended-DSA.git
cd Extended-DSA
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requiresments.txt
```

Or install manually:

```bash
pip install pandas sentence-transformers numpy torch faiss-cpu psutil tqdm openpyxl xlsxwriter gradio nltk
```

### Step 4: Download NLTK Data (First Run)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## ⚡ Quick Start

### Launch the Web Interface

```bash
python gradio_app.py
```

The app will start and display:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live  (if share=True)
```

Open the URL in your browser to access the interface.

### Quick Demo

1. **Load Dataset**: Upload an Excel file with a text column (`text1`, `text`, or `content`)
2. **Select Model**: Choose `all-MiniLM-L6-v2` (faster) or `bge-base-en-v1.5` (more accurate)
3. **Run Deduplication**: Select method (FAISS/SimHash/MinHash) and threshold (0.9 = 90% similar)
4. **Search**: Enter queries to find similar texts in the deduplicated dataset

---

## 📖 Usage Guide

### 1. Preparing Your Data

Create an Excel file (`.xlsx`) with a column containing text:

| text1 |
|-------|
| How can I learn programming? |
| What are the best coding tutorials? |
| Tips for learning to code |
| ... |

Supported column names: `text1`, `text`, `content`, `Text`, `Content`

### 2. Loading and Processing

```
1. Click "📁 1. Load Dataset" tab
2. Upload your .xlsx file
3. Select embedding model:
   - all-MiniLM-L6-v2: Faster, 256 max tokens
   - bge-base-en-v1.5: More accurate, 512 max tokens
4. Click "Load & Process"
```

The app will:
- Auto-detect the text column
- Split long texts to fit token limits (sentence-based)
- Generate embeddings for all texts

### 3. Deduplication

```
1. Click "⚡ 2. Deduplication" tab
2. Choose method:
   - FAISS: Most accurate, guaranteed results
   - SimHash: Fast, good for large datasets
   - MinHash: Fast, alternative LSH approach
3. Set similarity threshold (0.5 - 1.0)
   - 0.9 = 90% similar texts are grouped
   - Higher = stricter, fewer duplicates removed
4. Click "Run Deduplication"
```

### 4. Searching

```
1. Click "🔎 3. Search" tab
2. Enter your search query
3. Set number of results (1-50)
4. Click "Search"
```

### 5. Exporting Results

Click "📤 Export Results" to download the deduplicated dataset as Excel.

---

## 🔧 Technical Details

### Embedding Models

| Model | Dimensions | Max Tokens | Speed | Accuracy |
|-------|------------|------------|-------|----------|
| all-MiniLM-L6-v2 | 384 | 256 | ⚡⚡⚡ Fast | Good |
| bge-base-en-v1.5 | 768 | 512 | ⚡⚡ Medium | Excellent |
| e5-base-v2 | 768 | 512 | ⚡⚡ Medium | Excellent |
| instructor-base | 768 | 512 | ⚡ Slower | Excellent |

### Search Methods

#### FAISS (Facebook AI Similarity Search)
- **Type**: Exact search
- **Similarity**: Cosine similarity via inner product on normalized vectors
- **Pros**: Always returns exactly k results, highest accuracy
- **Cons**: Slower on very large datasets (>1M documents)

#### SimHash
- **Type**: Approximate search (LSH)
- **How it works**:
  1. Project embeddings onto random hyperplanes
  2. Create binary hash (256 bits default)
  3. Use banding for fast candidate retrieval
  4. Re-rank candidates with cosine similarity
- **Pros**: Very fast, memory efficient
- **Cons**: May return fewer than k results

#### MinHash
- **Type**: Approximate search (LSH)
- **How it works**:
  1. Convert embeddings to sets (top-k dimensions)
  2. Apply multiple hash functions
  3. Use banding for candidate retrieval
  4. Estimate Jaccard similarity
- **Pros**: Fast, good for high-dimensional data
- **Cons**: May return fewer than k results

### Default Parameters

```python
# SimHash
n_bits = 256          # Number of hash bits
n_bands = 32          # Number of LSH bands
min_match_bands = 5   # Minimum bands to match
max_rerank = 500      # Max candidates for re-ranking

# MinHash  
n_hashes = 256        # Number of hash functions
topk_dims = 256       # Top dimensions to consider
n_bands = 64          # Number of LSH bands
min_match_bands = 2   # Minimum bands to match
```

---

## 📊 Benchmarks

Performance on sample dataset (~80K texts):

| Method | Build Time | Query Time | Memory | Accuracy@10 |
|--------|------------|------------|--------|-------------|
| FAISS | 2.1s | 0.5ms | 150MB | 100% |
| SimHash | 3.5s | 0.8ms | 80MB | ~95% |
| MinHash | 4.2s | 1.2ms | 90MB | ~92% |

*Benchmarks performed on CPU (Intel i7-10th gen), batch_size=256*

---