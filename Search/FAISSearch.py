import faiss
import os
import pandas as pd
from pprint import pprint
from sentence_transformers import SentenceTransformer


names = ['all-miniLM-L6-v2', 'bge-base-en-v1.5', 'e5-base-v2', 'instructor-base']

def load_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model by name."""
    if model_name == 'all-miniLM-L6-v2':
        model_name = 'sentence-transformers/all-miniLM-L6-v2'
    elif model_name == 'bge-base-en-v1.5':
        model_name = 'BAAI/bge-base-en-v1.5'
    elif model_name == 'e5-base-v2':
        model_name = 'intfloat/e5-base-v2'
    elif model_name == 'instructor-base':
        model_name = 'hkunlp/instructor-base'

    return SentenceTransformer(model_name, device='cuda')

def create_embeddings(model, text_list):
    embeddings = model.encode(text_list, convert_to_numpy=True, show_progress_bar=True, batch_size=16)
    return embeddings

def get_faiss_index(name: str, load_model_ = False):
    """Returns the FAISS index, texts, and embedding dimension for the specified model name."""
    
    save_path = f'Data/FAISSINDEX/{name}.bin'
    model = None
    
    if os.path.exists(save_path):
        index = faiss.read_index(save_path)
        print(f"FAISS index loaded from {save_path} with {index.ntotal} vectors.")
        dataset = pd.read_excel(f'Data/{name}_dataset.xlsx')
        text1 = dataset.text1
        text2 = dataset.text2[dataset.text2.notnull()]
        texts = pd.concat([text1, text2]).tolist()
        dimension = index.d
        if load_model:
            model = load_model(name)
        return index, texts, dimension, model

    dataset = pd.read_excel(f'Data/{name}_dataset.xlsx')
    text1 = dataset.text1
    text2 = dataset.text2[dataset.text2.notnull()]
    texts = pd.concat([text1, text2]).tolist() 

    model = load_model(name)
    embeddings = create_embeddings(model, texts)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"FAISS index created for model: {name} with {index.ntotal} vectors.")

    if not os.path.exists('Data/FAISSINDEX'):
        os.makedirs('Data/FAISSINDEX')

    faiss.write_index(index, save_path)
    print(f"FAISS index saved to {save_path}")

    return index, texts, dimension, model

# Function to search for similar texts
def search_similar_texts(model, index, texts, query, k=5):
    """
    Search for k most similar texts to the query
    
    Args:
        query: Query text string
        k: Number of similar results to return
    
    Returns:
        DataFrame with similar texts and their distances
    """
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search the index
    distances, indices = index.search(query_embedding, k)
    
    # Create results dataframe
    results = pd.DataFrame({
        'rank': range(1, k+1),
        'text': [texts[idx] for idx in indices[0]],
        'distance': distances[0],
        'index': indices[0]
    })
    
    return results

indexes = {}
for name in names:
    index, texts, dimension, model = get_faiss_index(name, load_model_=False)
    indexes[name] = {
        'index': index,
        'texts': texts,
        'dimension': dimension,
        'model': model
    }