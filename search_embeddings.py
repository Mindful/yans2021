import argparse
import logging
from typing import Tuple, List
import re

from spacy.tokens import Token

from data.db import DbConnection, Word
from nlp.parsing import EmbeddingExtractor
from cluster_by_key import WordCluster
import numpy as np
from scipy.spatial.distance import cdist

target_word_regex = re.compile(r'\[.+\]')


def get_target_embedding(text: str) -> Tuple[Token, np.ndarray]:
    # this assumes there's only one pair of brackets (if there are brackets before the regex match, -1 is wrong)
    match = next(target_word_regex.finditer(text))
    target_start = match.span()[0]
    cleaned_string = text.replace('[', '').replace(']', '')
    assert cleaned_string[target_start:match.span()[1]-2] == match.group(0)[1:-1]
    extractor = EmbeddingExtractor()

    doc = extractor.nlp(cleaned_string)
    embeddings = extractor.get_word_embeddings(doc)
    return next((token, embedding) for token, embedding in embeddings if token.idx == target_start)


def classify_embedding(embedding: np.ndarray, clusters: WordCluster) -> int:
    xa = np.expand_dims(embedding, axis=0)
    xb = clusters.cluster_centers
    distances = cdist(xa, xb, metric='cosine')  # TODO:should be whatever we used to cluster, which for kmeans is euclid
    return np.argmin(distances)


def sort_words_by_distance(words: List[Word], vector: np.ndarray) -> List[Word]:
    xa = np.stack([x.embedding for x in words])
    xb = np.expand_dims(vector, axis=0)
    distances = cdist(xa, xb, metric='cosine')  # TODO: again, metric should be selectable

    words_with_index = list(enumerate(words))
    return [word for _, word in sorted(words_with_index, key=lambda x: distances[x[0]])]


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=True)

    args = parser.parse_args()

    token, embedding = get_target_embedding(args.input)

    db = DbConnection(args.run+'_clusters')
    logger.info('Loading cluster data')

    # TODO: this is defaulting to lemma, we should be able to select keys
    key = token.lemma_

    cluster = next(db.read_clusters(use_tqdm=False, where_clause=f"where key='{key}'"))
    target_label = classify_embedding(embedding, cluster)

    possible_labels = sorted(set(cluster.labels))

    words_db = DbConnection(args.run+"_words")
    words = list(words_db.read_words(include_sentences=True, use_tqdm=True, where_clause=f"where lemma='{key}'"))

    assert len(words) == len(cluster.labels), f'Word count {len(words)} and label count {len(cluster.labels)} must match'

    for cluster_label, cluster_centroid in zip(possible_labels, cluster.cluster_centers):
        examples = [word for word, label in zip(words, cluster.labels) if label == cluster_label]

        print()
        print('------------ cluster', cluster_label, '------------')
        print(len(examples), 'total words in cluster')
        if cluster_label == target_label:
            print('<<MATCHING CLUSTER>>')

            print('-----sorted by distance to centroid-----')
            examples_close_to_centroid = sort_words_by_distance(examples, cluster_centroid)
            for example in examples_close_to_centroid[0:5]:
                print(example.sentence)

            print('-----sorted by distance to input-----')
            examples_close_to_example = sort_words_by_distance(examples, cluster_centroid)
            for example in examples_close_to_example[0:5]:
                print(example.sentence)

        print('-----first cluster examples-----')
        for example in examples[0:5]:
            print(example.sentence)


if __name__ == '__main__':
    main()
