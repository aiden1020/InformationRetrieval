import pandas as pd
import numpy as np
from model.bm25 import BM25
from utils import preprocess_text

df_docs = pd.read_csv("documents_data.csv")
df_train = pd.read_csv("train_question.csv")

corpus = list(df_docs['Document_HTML'])
query = list(df_train['Question'])

processed_corpus = [preprocess_text(doc) for doc in corpus]

tokenized_corpus = [doc.split(" ") for doc in processed_corpus]

bm25 = BM25(tokenized_corpus)
hit = 0
for i, q in enumerate(query):
    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1][:3]
    print(f"Query {i}")
    print("Top 3 :", top_n_indices)
    if i in top_n_indices:
        hit += 1
print('Recall@3 accuracy : ', hit/len(query)*100)
