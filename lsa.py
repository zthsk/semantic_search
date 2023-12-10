import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle

class LSAmodel:
    def __init__(self, n_components=300, similarity='cosine'):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer()
        self.svd_model = TruncatedSVD(n_components=self.n_components)
        self.similarity = similarity
        self.document_term_matrix_svd = None

    def fit(self, documents):
        # Convert text data into a term-document matrix
        document_term_matrix = self.vectorizer.fit_transform(documents)
        # Apply Singular Value Decomposition (SVD) to reduce dimensionality
        self.document_term_matrix_svd = self.svd_model.fit_transform(document_term_matrix)

    def transform(self, documents):
        # Transform documents to the LSA space
        return self.svd_model.transform(self.vectorizer.transform(documents))

    def query(self, query_text):
        # Transform the query to the LSA space
        query_vector = self.transform([query_text])[0]

        # Compute cosine similarity with the SVD-transformed matrix
        if self.similarity == 'cosine':
            similarity = cosine_similarity(query_vector.reshape(1, -1), self.document_term_matrix_svd).flatten()
        else:
            # Other similarity measures can be implemented as needed
            raise NotImplementedError("Currently, only cosine similarity is implemented.")

        # Pair each similarity score with its document index
        doc_similarity_pairs = [(index, score) for index, score in enumerate(similarity)]
        # Sort the document-similarity pairs in descending order of similarity
        sorted_doc_similarity_pairs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)
        # Filter out pairs with similarity less than 0.8
        filtered_pairs = [pair for pair in sorted_doc_similarity_pairs if pair[1] >= 0.5]

        return filtered_pairs

    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump((self.vectorizer, self.svd_model, self.document_term_matrix_svd), file)

    def load_model(self, filepath):
        with open(filepath, 'rb') as file:
            self.vectorizer, self.svd_model, self.document_term_matrix_svd = pickle.load(file)

