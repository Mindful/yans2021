import argparse
import pickle
from typing import Dict
import re

from nlp.parsing import EmbeddingExtractor
from embed_words import Word
from cluster_words import WordCluster
from sklearn.cluster import KMeans
import numpy as np

target_word_regex = re.compile(r'\[.+\]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--data', required=True)

    args = parser.parse_args()

    # this assumes there's only one pair of brackets (if there are brackets before the regex match, -1 is wrong)
    match = next(target_word_regex.finditer(args.input))
    target_start = match.span()[0]
    cleaned_string = args.input.replace('[', '').replace(']', '')
    assert cleaned_string[target_start:match.span()[1]-2] == match.group(0)[1:-1]
    extractor = EmbeddingExtractor()

    doc = extractor.nlp(cleaned_string)
    embeddings = extractor.get_word_embeddings(doc)
    token, embedding = next((token, embedding) for token, embedding in embeddings if token.idx == target_start)

    with open(args.data, 'rb') as pfile:
        data: Dict[str, WordCluster] = pickle.load(pfile)

    word_cluster = data[token.lemma_]
    target_label = word_cluster.cluster.predict(np.expand_dims(embedding, 0))

    for cluster_label in word_cluster.cluster.labels_:
        examples = [word for word, label in zip(word_cluster.words, word_cluster.cluster.labels_)
                    if label == cluster_label]

        print()
        print('------------ cluster', cluster_label, '------------')
        if cluster_label == target_label:
            print('<<MATCHING CLUSTER>>')
        for example in examples[0:5]:
            print(example.sentence)


if __name__ == '__main__':
    main()
