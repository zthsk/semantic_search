import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RelevanceScorerError(Exception):
    """Custom exception class for RelevanceScorer."""
    pass

class RelevanceScorer:
    def __init__(self, mode=None, threshold=0.5, glove_file = 'glove.840B.300d.txt'):
        self.embeddings_dict = None
        self.doc_embeddings = None
        self.mode = mode
        self.threshold = threshold

        if mode == 'train':
            if not glove_file:
                raise RelevanceScorerError("GloVe file not provided. Train mode requires a GloVe file.")
            self.embeddings_dict = self.load_glove_embeddings(glove_file)
        elif mode == 'query':
            print("Query mode selected. Document embeddings must be provided.")
            self.embeddings_dict = self.load_glove_embeddings(glove_file)
        elif mode is None:
            raise RelevanceScorerError("Specify mode as 'train' or 'query'.")
        
    @staticmethod
    def load_glove_embeddings(glove_file):
        embeddings_dict = {}
        with open(glove_file, 'r', encoding='utf8') as file:
            for line in file:
                values = line.split()
                if len(values) <= 1:
                    print(f"Invalid line format for word: '{values[0]}' (if present). Skipping.")
                    continue
                word = values[0]
                try:
                    vector = np.asarray(values[1:], "float32")
                except ValueError:
                    #print(f"Error parsing line for word: {word}. Skipping.")
                    continue
                embeddings_dict[word] = vector
        return embeddings_dict

    @staticmethod
    def preprocess_text(text):
        return text.lower().split()

    def text_to_vector(self, text):
        words = self.preprocess_text(text)
        word_vectors = [self.embeddings_dict.get(word, np.zeros((300,))) for word in words]
        return np.mean(word_vectors, axis=0)
    
    def save_embeddings(self, file_path):
        np.save(file_path, np.array(self.doc_embeddings))

    def train(self, documents, save_path):
        if not self.embeddings_dict:
            raise RelevanceScorerError("GloVe embeddings not loaded. Train mode requires a GloVe file.")
        self.doc_embeddings = [self.text_to_vector(doc) for doc in documents]
        self.save_embeddings(save_path)

    def load_embeddings(self, file_path):
        self.doc_embeddings = np.load(file_path, allow_pickle=True)

    def query(self, query):
        if self.doc_embeddings is None:
            raise RelevanceScorerError("Document embeddings not loaded. Use load_embeddings or train mode.")

        query_vector = self.text_to_vector(query)
        doc_similarity_pairs = []

        for index, doc_vector in enumerate(self.doc_embeddings):
            similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
            doc_similarity_pairs.append((index, similarity))

        # Sort the document-similarity pairs in descending order of similarity
        sorted_doc_similarity_pairs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)

        # Filter out pairs with similarity less than the threshold
        filtered_pairs = [pair for pair in sorted_doc_similarity_pairs if pair[1] >= self.threshold]

        return filtered_pairs


