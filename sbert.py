import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SBERTSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None

    def embeddings_exist(self, path):
        """Check if embeddings file exists at the given path."""
        return os.path.isfile(path)

    def save_embeddings(self, documents, save_path):
        """Generate and save embeddings."""
        self.doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
        embeddings_array = self.doc_embeddings.cpu().numpy()
        np.save(save_path, embeddings_array)

    def load_embeddings(self, load_path):
        """Load embeddings from a file."""
        embeddings_array = np.load(load_path)
        self.doc_embeddings = [torch.tensor(embedding) for embedding in embeddings_array]

    def query(self, query_text):
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)  # This should be 2D now
        similarities = []
        for doc_emb in self.doc_embeddings:
            similarity = cosine_similarity(query_embedding.unsqueeze(0), doc_emb.unsqueeze(0))[0][0]
            similarities.append(similarity)

        # Pair each similarity score with its document index
        doc_similarity_pairs = [(index, score) for index, score in enumerate(similarities)]

        # Sort the document-similarity pairs in descending order of similarity
        sorted_doc_similarity_pairs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)

        # Filter out pairs with similarity less than a threshold (e.g., 0.5)
        filtered_pairs = [pair for pair in sorted_doc_similarity_pairs if pair[1] >= 0.5]

        return filtered_pairs
