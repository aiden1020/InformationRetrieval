import numpy as np


def remove_html_tags(text):
    result = []
    inside_tag = False
    for char in text:
        if char == '<':
            inside_tag = True
        elif char == '>':
            inside_tag = False
        elif not inside_tag:
            result.append(char)
    return ''.join(result)


def remove_https_links(text):
    words = text.split()
    filtered_words = []
    for word in words:
        if not word.startswith('https://'):
            filtered_words.append(word)
    return ' '.join(filtered_words)


def preprocess_text(text):
    text = remove_https_links(text)
    text = remove_html_tags(text)
    text = ''.join([char if char.isalnum() or char.isspace()
                   else ' ' for char in text])
    text = text.lower()
    text = ' '.join(text.split())
    return text


def cosine_similarity(tfidf_matrix_1, tfidf_matrix_2):
    dot_product = np.dot(tfidf_matrix_1, tfidf_matrix_2.T)
    norm_1 = np.linalg.norm(tfidf_matrix_1, axis=1, keepdims=True)
    norm_2 = np.linalg.norm(tfidf_matrix_2, axis=1, keepdims=True)
    norm_1[norm_1 == 0] = 1e-9
    norm_2[norm_2 == 0] = 1e-9
    similarity = dot_product / (norm_1 * norm_2.T)
    return similarity
