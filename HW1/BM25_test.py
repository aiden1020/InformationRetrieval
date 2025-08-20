import pandas as pd
import math
import numpy as np
from model.bm25 import BM25
from utils import preprocess_text


df_docs = pd.read_csv("documents_data.csv")
df_test = pd.read_csv("test_question.csv")

corpus = list(df_docs['Document_HTML'])
query = list(df_test['Question'])

processed_corpus = [preprocess_text(doc) for doc in corpus]

tokenized_corpus = [doc.split(" ") for doc in processed_corpus]

bm25 = BM25(tokenized_corpus)
df_result = pd.DataFrame(columns=['index', 'answer'])

for i, q in enumerate(query):
    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1][:3]
    answers = " ".join(str(idx+1) for idx in top_n_indices.tolist())
    row = {'index': i+1, 'answer': answers}
    df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
df_result.to_csv("output_BM25.csv", index=False)

print("Results saved to output.csv")
