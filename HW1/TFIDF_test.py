
import pandas as pd
import numpy as np
from utils import preprocess_text, cosine_similarity
from model.tf_idf import TFIDFModel

df_docs = pd.read_csv("documents_data.csv")
df_test = pd.read_csv("test_question.csv")

corpus = list(df_docs['Document_HTML'])
query = list(df_test['Question'])

processed_corpus = [preprocess_text(doc) for doc in corpus]
tokenized_corpus = [doc.split(" ") for doc in processed_corpus]

processed_query = [preprocess_text(q) for q in query]
tokenized_queries = [q.split(" ") for q in processed_query]

tfidf = TFIDFModel()

doc_tfidf_matrix, query_tfidf_matrix = tfidf.fit_transform(
    tokenized_corpus, tokenized_queries)

cosine_sim_matrix = cosine_similarity(query_tfidf_matrix, doc_tfidf_matrix)

top_n = 3
top_doc_indices = np.argsort(cosine_sim_matrix, axis=1)[:, -top_n:][:, ::-1]

df_result = pd.DataFrame(columns=['index', 'answer'])
for i, indices in enumerate(top_doc_indices):
    answer = " ".join(str(idx+1) for idx in indices.tolist())
    row = {'index': i+1, 'answer': answer}
    df_result = pd.concat(
        [df_result, pd.DataFrame([row])], ignore_index=True)

df_result.to_csv("output_tfidf.csv", index=False)
print("output csv file -> output.csv")
