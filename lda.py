

from gensim import corpora, models, matutils
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class LDAModel:
    def __init__(self, num_topics=300, similarity='cosine', chunksize=20000):
        self.num_topics = num_topics
        self.dictionary = None
        self.model = None
        self.similarity = similarity
        self.chunksize = chunksize
        self.doc_topic_distributions = []

    def train(self, documents):
        total_docs = len(documents)
        num_chunks = (total_docs + self.chunksize - 1) // self.chunksize

        print(f"Total documents: {total_docs}, Processing in {num_chunks} chunks.")

        for i in range(0, total_docs, self.chunksize):
            chunk = documents[i:i + self.chunksize]
            corpus_chunk = self.preprocess(chunk)

            # Rebuild the model if dictionary size changed
            if self.dictionary_updated:
                self.model = models.LdaModel(corpus_chunk, num_topics=self.num_topics, id2word=self.dictionary, passes=15)
                self.dictionary_updated = False
            else:
                self.model.update(corpus_chunk)

            self.doc_topic_distributions.extend(self.model[corpus_chunk])
            print(f"Processed chunk {i // self.chunksize + 1}/{num_chunks}")

    def preprocess(self, documents):
        tokenized_docs = [doc.split() for doc in documents]
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(tokenized_docs)
            self.dictionary_updated = True
        else:
            old_size = len(self.dictionary)
            self.dictionary.add_documents(tokenized_docs)
            new_size = len(self.dictionary)
            if new_size > old_size:
                self.dictionary_updated = True
            print(f"Dictionary size before update: {old_size}")
            print(f"Dictionary size after update: {new_size}")
        return [self.dictionary.doc2bow(doc) for doc in tokenized_docs]


    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump((self.dictionary, self.model, self.doc_topic_distributions), file)

    def load_model(self, filepath):
        with open(filepath, 'rb') as file:
            self.dictionary, self.model, self.doc_topic_distributions = pickle.load(file)

    def cosine_similarity(self, query_lda):
        query_dense = matutils.sparse2full(query_lda, self.num_topics).reshape(1, -1)
        doc_dense = np.vstack([matutils.sparse2full(dist, self.num_topics) for dist in self.doc_topic_distributions])
        return cosine_similarity(query_dense, doc_dense)[0]

    def jensen_shannon(self, query_lda):
        sims = []
        for doc_lda in self.doc_topic_distributions:
            sim = jensenshannon(matutils.sparse2full(query_lda, self.num_topics),
                                matutils.sparse2full(doc_lda, self.num_topics))
            sims.append(sim)
        return sims

    def query(self, query_text):
        query_bow = self.dictionary.doc2bow(query_text.split())
        query_lda = self.model[query_bow]

        # Compute similarities based on the selected method
        if self.similarity == 'cosine':
            sims = self.cosine_similarity(query_lda)
        elif self.similarity == 'jensen_shannon':
            sims = self.jensen_shannon(query_lda)
        else:
            raise NotImplementedError("Currently, only cosine and jensen_shannon similarities are implemented.")

        # Pair each similarity score with its document index
        doc_similarity_pairs = [(index, score) for index, score in enumerate(sims)]

        # Sort the document-similarity pairs in descending order of similarity
        sorted_doc_similarity_pairs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)

        # Filter out pairs with similarity less than 0.5
        filtered_pairs = [pair for pair in sorted_doc_similarity_pairs if pair[1] >= 0.5]

        return filtered_pairs

