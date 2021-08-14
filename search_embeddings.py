import argparse
import logging
from typing import Tuple
import re

from spacy.tokens import Token

from data.db import DbConnection
from nlp.embedding import EmbeddingExtractor, classify_embedding, sort_words_by_distance
import numpy as np

target_word_regex = re.compile(r'\[.+\]')


def print_sentence(sentence: str):
    if len(sentence) > 550:
        sentence = sentence[:550] + ".....(cont)"

    print(sentence.replace('\n', r"\n"))


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
    #TODO: account for returned IDS
    words = list(words_db.read_words(include_sentences=True, use_tqdm=True, where_clause=f"where lemma='{key}'"))

    assert len(words) == len(cluster.labels), f'Word count {len(words)} and label count {len(cluster.labels)} must match'

    unclustered = 0
    print("found", len(cluster.cluster_centers), "clusters")
    for cluster_label, cluster_centroid in zip(possible_labels, cluster.cluster_centers):
        examples = [word for word, label in zip(words, cluster.labels) if label == cluster_label]
        if cluster_label == -1:
            unclustered = len(examples)

        print()
        print('------------ cluster', cluster_label, '------------')
        print(len(examples), 'total words in cluster', str(100 * round(len(examples) / len(words), 2)) + "%")
        if cluster_label == target_label:
            print('<<MATCHING CLUSTER>>')

        print('-----sorted by distance to centroid-----')
        examples_close_to_centroid = sort_words_by_distance(examples, cluster_centroid)
        for example in examples_close_to_centroid[0:5]:
            print_sentence(example.sentence)

        print('-----sorted by distance to input-----')
        examples_close_to_example = sort_words_by_distance(examples, embedding)
        for example in examples_close_to_example[0:5]:
            print_sentence(example.sentence)

    if unclustered != 0:
        print(unclustered, 'unclustered examples from', len(words), 'total examples:',
              str(100 * round(unclustered / len(words), 2)) + "%", 'unclustered')



if __name__ == '__main__':
    main()
