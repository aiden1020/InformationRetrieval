import json
import os
import re
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

def clean_text(text):
    if '*' in text or '@' in text or '[' in text:
        return None
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\_+', '', text)
    text = re.sub(r'\-+', '', text)
    text = re.sub(r'\.+', '', text)
    text = re.sub(r'\,+', '', text)
    text = re.sub(r'[^\w\s,]', '', text)
    return text.strip().lower()

def postprocess(text, words_list):
    words_in_text = word_tokenize(text)
    if len(words_in_text) < 2:
        return None
    if any(word in words_list for word in words_in_text):
        return None
    return ' '.join(words_in_text)

def remove_high_frequency_words(text, high_frequency_words):
    words_in_text = text.split()
    filtered_words = [
        word for word in words_in_text if word not in high_frequency_words]
    return ' '.join(filtered_words)

def process_article(file_path, words_list):
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as article_file:
        article_content = json.load(article_file)

    processed_sentences = []
    for text in article_content:
        sentences = sent_tokenize(text.lower())
        cleaned_sentences = [clean_text(sentence) for sentence in sentences]
        cleaned_sentences = [
            sentence for sentence in cleaned_sentences if sentence is not None]

        sentences_postprocessed = [postprocess(
            sentence, words_list) for sentence in cleaned_sentences]
        sentences_postprocessed = [
            sentence for sentence in sentences_postprocessed if sentence is not None]
        processed_sentences.extend(sentences_postprocessed)

    return processed_sentences

def rank_with_bm25(sentences, query):
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    bm25 = BM25Okapi(tokenized_sentences)
    tokenized_query = word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [sentences[i] for i in sorted_indices]

def rank_with_tfidf(sentences, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences + [query])
    query_vector = tfidf_matrix[-1]
    sentence_vectors = tfidf_matrix[:-1]
    scores = (sentence_vectors @ query_vector.T).toarray().flatten()
    sorted_indices = np.argsort(scores)[::-1]
    return [sentences[i] for i in sorted_indices]

def rank_with_faiss(sentences, query, k=10):
    """
    使用 Faiss 查詢最相關的句子。
    :param sentences: 候選句子列表
    :param query: 查詢文本
    :param k: 返回的最相關句子數量
    :return: top-k 相關句子列表
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])

    embedding_size = sentence_embeddings.shape[1]
    res = faiss.StandardGpuResources() 
    index_cpu = faiss.IndexFlatL2(embedding_size)
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index_gpu.add(np.array(sentence_embeddings, dtype=np.float32))

    distances, indices = index_gpu.search(np.array(query_embedding, dtype=np.float32), k)

    top_k_sentences = [sentences[i] for i in indices[0]]
    return top_k_sentences

def process_data(data, articles_directory, save_label, ranking_model):
    words_list = [
        'link', 'facebook', 'twitter', 'factcheckorg', 'cookie',
        'cookies', 'browsers', 'browser', 'retweets', 'retweet', 'account', 'profile'
    ]
    processed_data = []

    for item in data[:1]:
        claim_text = item['metadata']['claim']
        data_id = item['metadata']['id']
        premise_articles = item['metadata']['premise_articles']
        if save_label:
            label = item.get('label')

        all_sentences = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_article, os.path.join(articles_directory, file_name), words_list): file_name
                for url, file_name in premise_articles.items()
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_sentences.extend(result)

        sentence_counter = Counter(all_sentences)
        unique_sentences = [sentence for sentence,
                            count in sentence_counter.items() if count == 1]

        word_counter = Counter()
        for text in unique_sentences:
            for word in text.split():
                word_counter[word] += 1
        most_common_words = word_counter.most_common(10)
        high_frequency_words = {word for word, _ in most_common_words}

        final_filtered_premises = [
            remove_high_frequency_words(text, high_frequency_words)
            for text in unique_sentences
        ]
        final_filtered_premises = [
            text for text in final_filtered_premises if len(text.split()) >= 2
        ]

        if not final_filtered_premises:
            top_10_sentences = ["no data"]
        else:
            if ranking_model == "bm25":
                top_10_sentences = rank_with_bm25(
                    final_filtered_premises, claim_text)[:10]
            elif ranking_model == "tf-idf":
                top_10_sentences = rank_with_tfidf(
                    final_filtered_premises, claim_text)[:10]
            elif ranking_model == "faiss":
                top_10_sentences = rank_with_faiss(
                    final_filtered_premises, claim_text)[:10]

        processed_item = {
            'claim': claim_text,
            'id': data_id,
            'top_10_sentences': top_10_sentences,
        }
        if save_label:
            processed_item['label'] = label
        processed_data.append(processed_item)

    return processed_data

def main():
    parser = argparse.ArgumentParser(description="處理生成資訊檢索數據")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "train", "valid"],
        required=True,
        help="指定執行模式：test, train, valid"
    )
    parser.add_argument(
        "--ranking_model",
        type=str,
        choices=["bm25", "tf-idf", "faiss"],
        default="bm25",
        help="選擇排名模型：bm25, tf-idf 或 faiss"
    )
    args = parser.parse_args()

    mode_settings = {
        "test": {
            "input_path": "2024-generative-information-retrieval-hw-2/test.json",
            "output_path": "test_dataset.json",
            "save_label": False,
        },
        "train": {
            "input_path": "2024-generative-information-retrieval-hw-2/train.json",
            "output_path": "train_dataset.json",
            "save_label": True,
        },
        "valid": {
            "input_path": "2024-generative-information-retrieval-hw-2/valid.json",
            "output_path": "valid_dataset.json",
            "save_label": True,
        },
    }

    mode = args.mode
    ranking_model = args.ranking_model
    input_path = mode_settings[mode]["input_path"]
    output_path = mode_settings[mode]["output_path"]
    save_label = mode_settings[mode]["save_label"]
    articles_directory = "2024-generative-information-retrieval-hw-2/articles"

    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    processed_data = process_data(
        data=data,
        articles_directory=articles_directory,
        save_label=save_label,
        ranking_model=ranking_model
    )

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

    print(f"處理後的數據已保存到 {output_path}")
if __name__ == "__main__":
    main()
