import spacy
from functools import lru_cache
from itertools import cycle
import re
from typing import List, Dict

from pydantic import BaseModel

import numpy as np

from spacy.parts_of_speech import NAMES
from data.db import DbConnection, Word, WordCluster
from nlp.embedding import sort_words_by_distance, EmbeddingExtractor, classify_embedding
from cluster import cluster_kmeans, cluster_dbscan, compute_display_embeddings

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


def create_new_cluster(lemma, pos) -> WordCluster:
    word_data = list(db.read_words(where_clause=f"WHERE lemma='{lemma}' and pos={pos}"))
    if len(word_data) == 0:
        raise RuntimeError(f'{lemma}/{pos} not found in database')

    cluster_pca, display_embeddings = compute_display_embeddings(word_data)
    cluster = cluster_kmeans(lemma, pos, word_data, cluster_pca)
    db.save_cluster(cluster)
    db.add_display_embedding_to_words(display_embeddings)

    # we need sentences and display embeddings, so we reload the words along with the cluster
    return db.get_cluster(lemma, pos, 'r')


def get_or_create_cluster(lemma: str, pos: int, tree: str) -> WordCluster:
    cluster_from_db = db.get_cluster(lemma, pos, tree)
    if cluster_from_db is None:
        if tree == 'r':
            return create_new_cluster(lemma, pos)

        tree_data = tree.split('-')
        target_label = int(tree_data[-1])
        parent_tree = '-'.join(tree_data[:-1])

        parent_cluster = db.get_cluster(lemma, pos, parent_tree)
        if parent_cluster is None:
            raise RuntimeError(f'could not find parent {lemma}/{pos}/{parent_tree}')

        # TODO: should we worry about the previously displayed sentences being in the new cluster?
        # if so, we want to sort the words by distance to the old cluster so we get teh same words, and then
        # use those words plus another 2xcluster_size. either way there will be new sentences

        child_word_list = [word for word in parent_cluster.words if word.cluster_label == target_label]
        child_cluster = cluster_kmeans(lemma, pos, child_word_list, parent_cluster.pca, tree)
        db.save_cluster(child_cluster)

        return child_cluster
    else:
        return cluster_from_db


def subcluster_search(search_data: ClusterSearchData):
    child_cluster = get_or_create_cluster(search_data.lemma, search_data.pos, search_data.tree)

    input_embedding = np.array(search_data.embedding)
    input_label = classify_embedding(input_embedding, child_cluster)

    return _format_output(search_data, child_cluster, input_label)


@lru_cache(maxsize=100)
def compute_search_data(text_input: str):
    match = next(target_word_regex.finditer(text_input))
    target_start = match.span()[0]
    cleaned_string = text_input.replace('[', '').replace(']', '')

    doc = extractor.nlp(cleaned_string)
    embeddings = extractor.get_word_embeddings(doc)
    token, embedding = next((token, embedding) for token, embedding in embeddings if token.idx == target_start)

    cluster = get_or_create_cluster(token.lemma_, token.pos, 'r')
    input_label = classify_embedding(embedding, cluster)
    input_display_embedding = cluster.pca.transform(np.expand_dims(embedding, axis=0)).squeeze()

    search_data = ClusterSearchData(
        lemma=cluster.lemma,
        pos=cluster.pos,
        tree=cluster.tree,
        sentence=cleaned_string,
        word_start=token.idx,
        word_end=token.idx + len(token),
        word=token.text,
        display_embedding=list(input_display_embedding),
        embedding=list(embedding),
    )

    return _format_output(search_data, cluster, input_label)


def _format_output(search_data: ClusterSearchData, cluster: WordCluster, input_label: int, display_limit: int = 50):

    words_by_cluster_label = {
        label: sort_words_by_distance([word for word in cluster.words if word.cluster_label == label], centroid)
        for label, centroid in zip(cluster.labels, cluster.cluster_centers)
    }

    closest_to_input = sort_words_by_distance(cluster.words, np.array(search_data.embedding))

    cluster_colors = [next(cluster_color_iter) for _ in cluster.labels]
    tree_suffix = '-'.join(search_data.tree.split('-')[1:])
    if len(tree_suffix) > 0:
        tree_suffix = '-' + tree_suffix

    return {'clusters': [
        {
            'name': f'{search_data.lemma}{tree_suffix}-{label}',
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
        'name': f'Input',
        'data': [
            {'x': search_data.display_embedding[0],
             'y': search_data.display_embedding[1],
             'z': search_data.display_embedding[2],
             'text': search_data.sentence}
        ],
        'color': cluster_colors[input_label],
        'is_user_input': True

    }],
        'similar_sentences': [
            {
                'text': word.sentence,
                'cluster': word.cluster_label
            } for word in closest_to_input[:display_limit]
        ],
        'title': f'Clusters for {search_data.lemma}{tree_suffix} ({NAMES[search_data.pos]})',
        'search_data': search_data.dict()
    }


if __name__ == '__main__':
    d = create_new_cluster('dog', spacy.parts_of_speech.VERB)
    print(d)
