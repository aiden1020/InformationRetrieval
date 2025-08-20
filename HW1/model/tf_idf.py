import numpy as np


class TFIDFModel:
    def __init__(self):
        self.vocab_index = {}
        self.idf_values = None

    def build_vocabulary(self, tokenized_texts):
        vocab = set([token for tokens in tokenized_texts for token in tokens])
        self.vocab_index = {word: i for i, word in enumerate(vocab)}

    def compute_tf(self, tokenized_texts):
        tf_matrix = np.zeros(
            (len(tokenized_texts), len(self.vocab_index)), dtype=int)
        for i, tokens in enumerate(tokenized_texts):
            for token in tokens:
                if token in self.vocab_index:
                    tf_matrix[i, self.vocab_index[token]] += 1
        doc_lengths = np.array([len(tokens) for tokens in tokenized_texts])
        return tf_matrix / doc_lengths[:, None]

    def compute_idf(self, tf_matrix):
        N = len(tf_matrix)
        df_count = np.sum(tf_matrix > 0, axis=0)
        self.idf_values = np.log(N / (1 + df_count))

    def fit_transform(self, tokenized_corpus, tokenized_queries):
        self.build_vocabulary(tokenized_corpus)

        doc_tf_matrix = self.compute_tf(tokenized_corpus)
        query_tf_matrix = self.compute_tf(tokenized_queries)

        self.compute_idf(doc_tf_matrix)
        doc_tfidf_matrix = doc_tf_matrix * self.idf_values
        query_tfidf_matrix = query_tf_matrix * self.idf_values

        return doc_tfidf_matrix, query_tfidf_matrix
