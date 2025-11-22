import faiss
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer

dataset = pd.read_excel('Data/all-MiniLM-L6-v2_dataset.xlsx')