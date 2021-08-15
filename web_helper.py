from functools import lru_cache
from itertools import cycle
import re

import numpy as np

from data.db import DbConnection
from nlp.embedding import sort_words_by_distance, EmbeddingExtractor, classify_embedding

db = DbConnection('css')
extractor = EmbeddingExtractor()

target_word_regex = re.compile(r'\[.+\]')
cluster_color_iter = cycle(('rgb(0, 59, 27)', 'rgb(186, 83, 19)', 'rgb(110, 46, 5)'))


def get_data_for_search(text_input: str):
    match = next(target_word_regex.finditer(text_input))
    target_start = match.span()[0]
    cleaned_string = text_input.replace('[', '').replace(']', '')

    return cluster_data(cleaned_string, target_start)


@lru_cache(maxsize=100)
def cluster_data(cleaned_text: str, target_start: int, display_limit: int = 50):
    doc = extractor.nlp(cleaned_text)
    embeddings = extractor.get_word_embeddings(doc)
    token, embedding = next((token, embedding) for token, embedding in embeddings if token.idx == target_start)

    cluster, word_list = db.get_cluster_for_token(token)
    cluster_labels = sorted(set(cluster.labels))
    cluster_colors = [next(cluster_color_iter) for _ in cluster_labels]
    input_label = classify_embedding(embedding, cluster)

    words_by_cluster_label = {
        label: sort_words_by_distance([word for word, word_label in zip(word_list, cluster.labels)
                                       if word_label == label], centroid)
        for label, centroid in zip(cluster_labels, cluster.cluster_centers)
    }

    display_data = {'clusters': [
        {
            'name': f'({token.pos_}) {token.lemma_} {label}',
            'data': [
                        {'x': word.display_embedding[0].item(),
                         'y': word.display_embedding[1].item(),
                         'z': word.display_embedding[2].item(),
                         'text': word.sentence} for word in cluster_words[:display_limit]
                    ],
            'color': cluster_colors[label],
            'pos': cluster.pos,
            'lemma': token.lemma_,
            'label': label.item(),
            'tree': 'r',
            'is_user_input': False
        } for label, cluster_words in words_by_cluster_label.items()
    ]}

    input_display_embedding = cluster.pca.transform(np.expand_dims(embedding, axis=0)).squeeze()

    display_data['clusters'].append({
        'name': f'({token.pos_}) {token.lemma_} input',
        'data': [
            {'x': input_display_embedding[0].item(),
             'y': input_display_embedding[1].item(),
             'z': input_display_embedding[2].item(),
             'text': token.doc.text}
        ],
        'color': cluster_colors[input_label],
        'is_user_input': True

    })

    return display_data


if __name__ == '__main__':
    d = get_data_for_search('I like to [play].')
    print(d)