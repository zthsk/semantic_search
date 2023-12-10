import numpy as np
import os
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

class BERTSearchEngine:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.doc_embeddings = None

    def embeddings_exist(self, path):
        """Check if embeddings file exists at the given path."""
        return os.path.isfile(path)

    def save_embeddings(self, documents, save_path):
        """Generate and save embeddings."""
        self.doc_embeddings = [self.get_bert_embedding(doc) for doc in documents]
        embeddings_array = torch.stack(self.doc_embeddings).detach().numpy()
        np.save(save_path, embeddings_array)

    def load_embeddings(self, load_path):
        """Load embeddings from a file."""
        embeddings_array = np.load(load_path)
        self.doc_embeddings = [torch.tensor(embedding) for embedding in embeddings_array]

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings.mean(dim=1).squeeze()  # Ensure two dimensions

    def query(self, query):
        query_embedding = self.get_bert_embedding(query)  # This should be 2D now
        similarities = []
        for doc_emb in self.doc_embeddings:
            doc_emb_2d = doc_emb.squeeze()  # Ensure doc embeddings are also 2D
            similarity = cosine_similarity(query_embedding.unsqueeze(0), doc_emb_2d.unsqueeze(0))[0][0]
            similarities.append(similarity)

        # Pair each similarity score with its document index
        doc_similarity_pairs = [(index, score) for index, score in enumerate(similarities)]

        # Sort the document-similarity pairs in descending order of similarity
        sorted_doc_similarity_pairs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)

        # Filter out pairs with similarity less than 0.5
        filtered_pairs = [pair for pair in sorted_doc_similarity_pairs if pair[1] >= 0.5]

        return filtered_pairs
