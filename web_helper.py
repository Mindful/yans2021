from functools import lru_cache
from itertools import cycle
import re
from typing import List, Dict

from pydantic import BaseModel

import numpy as np

from spacy.parts_of_speech import NAMES
from data.db import DbConnection, Word
from nlp.embedding import sort_words_by_distance, EmbeddingExtractor, classify_embedding
from cluster import cluster_kmeans, cluster_dbscan

db = DbConnection('css')
extractor = EmbeddingExtractor()

target_word_regex = re.compile(r'\[.+\]')
cluster_color_iter = cycle(('rgb(0, 59, 27)', 'rgb(186, 83, 19)', 'rgb(110, 46, 5)'))


class ClusterSearchData(BaseModel):
    lemma: str
    pos: int
    tree: str

    sentence: str
    word_start: int
    word_end: int
    word: str

    display_embedding: List[int]
    embedding: List[int]


def get_data_for_search(text_input: str):
    match = next(target_word_regex.finditer(text_input))
    target_start = match.span()[0]
    cleaned_string = text_input.replace('[', '').replace(']', '')

    return compute_search_data(cleaned_string, target_start)


#TODO: this currently only goes one cluster deep. to go deeper we need to save subclusters, and have a junction table
#TODO: this doesn't seem to return sensible clusters. is using the old PCA bad, or are we returning garbage data somehow?
def subcluster_search(search_data: ClusterSearchData):
    parent_cluster, word_list = db.get_cluster_for_token(search_data.lemma, search_data.pos)
    target_label = int(search_data.tree.split('-')[-1])

    # TODO: should we worry about the previously displayed sentences being in the new cluster?
    # if so, we want to sort the words by distance to the old cluster so we get teh same words, and then
    # use those words plus another 2xcluster_size. either way there will be new sentences
    # words = sort_words_by_distance([word for word, word_label in zip(word_list, cluster.labels)
    #                                 if word_label == target_label], cluster.cluster_centers[target_label])

    words = [word for word, word_label in zip(word_list, parent_cluster.labels) if word_label == target_label]

    child_cluster = cluster_kmeans(search_data.lemma, search_data.pos, words, parent_cluster.pca, search_data.tree)

    cluster_labels = sorted(set(child_cluster.labels))
    input_embedding = np.array(search_data.embedding)
    input_label = classify_embedding(input_embedding, child_cluster)

    words_by_cluster_label = {
        label.item(): sort_words_by_distance([word for word, word_label in zip(word_list, child_cluster.labels)
                                              if word_label == label], centroid)
        for label, centroid in zip(cluster_labels, child_cluster.cluster_centers)
    }

    return _format_output(search_data, words_by_cluster_label, cluster_labels, input_label)


@lru_cache(maxsize=100)
def compute_search_data(cleaned_text: str, target_start: int, display_limit: int = 50):
    doc = extractor.nlp(cleaned_text)
    embeddings = extractor.get_word_embeddings(doc)
    token, embedding = next((token, embedding) for token, embedding in embeddings if token.idx == target_start)

    cluster, word_list = db.get_cluster_for_token(token.lemma_, token.pos)
    cluster_labels = sorted(set(cluster.labels))
    input_label = classify_embedding(embedding, cluster)
    input_display_embedding = cluster.pca.transform(np.expand_dims(embedding, axis=0)).squeeze()

    words_by_cluster_label = {
        label.item(): sort_words_by_distance([word for word, word_label in zip(word_list, cluster.labels)
                                              if word_label == label], centroid)
        for label, centroid in zip(cluster_labels, cluster.cluster_centers)
    }

    search_data = ClusterSearchData(
        lemma=cluster.lemma,
        pos=cluster.pos,
        tree=cluster.tree,
        sentence=cleaned_text,
        word_start=token.idx,
        word_end=token.idx + len(token),
        word=token.text,
        display_embedding=list(input_display_embedding),
        embedding=list(embedding),
    )

    return _format_output(search_data, words_by_cluster_label, cluster_labels, input_label)


def _format_output(search_data: ClusterSearchData, words_by_cluster_label: Dict[int, List[Word]],
                   cluster_labels: List[int], input_label: int, display_limit: int = 50):
    cluster_colors = [next(cluster_color_iter) for _ in cluster_labels]

    return {'clusters': [
        {
            'name': f'({NAMES[search_data.pos]}) {search_data.lemma} {label}',
            'data': [
                        {'x': word.display_embedding[0].item(),
                         'y': word.display_embedding[1].item(),
                         'z': word.display_embedding[2].item(),
                         'text': word.sentence} for word in cluster_words[:display_limit]
                    ],
            'color': cluster_colors[label],
            'label': label,
            'is_user_input': False
        } for label, cluster_words in words_by_cluster_label.items()
    ] + [{
        'name': f'({NAMES[search_data.pos]}) {search_data.lemma} input',
        'data': [
            {'x': search_data.display_embedding[0],
             'y': search_data.display_embedding[1],
             'z': search_data.display_embedding[2],
             'text': search_data.sentence}
        ],
        'color': cluster_colors[input_label],
        'is_user_input': True

    }],
        'search_data': search_data.dict()
    }


if __name__ == '__main__':
    d = get_data_for_search('I like to [play].')
    print(d)