import argparse
import pickle
from typing import Dict
import re

from nlp.parsing import EmbeddingExtractor
from embed_words import Word
from cluster_by_key import WordCluster
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

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

    centroids = word_cluster.cluster.cluster_centers_
    labels = sorted(set(word_cluster.cluster.labels_))

    for cluster_label, cluster_centroid in zip(labels, centroids):
        examples = [word for word, label in zip(word_cluster.words, word_cluster.cluster.labels_)
                    if label == cluster_label]

        print()
        print('------------ cluster', cluster_label, '------------')
        print(sum(1 for x in word_cluster.cluster.labels_ if x == cluster_label), 'total words in cluster')
        if cluster_label == target_label:
            print('<<MATCHING CLUSTER>>')

            print('-----sorted by distance to centroid-----')
            examples_close_to_centroid = sorted(examples,
                                                key=lambda x: euclidean_distances(x.embedding.reshape(1, -1),
                                                                                  cluster_centroid.reshape(1, -1)))
            for example in examples_close_to_centroid[0:5]:
                print(example.sentence)

            print('-----sorted by distance to input-----')
            examples_close_to_example = sorted(examples,
                                               key=lambda x: euclidean_distances(x.embedding.reshape(1, -1),
                                                                                 embedding.reshape(1, -1)))
            for example in examples_close_to_example[0:5]:
                print(example.sentence)


        print('-----first cluster examples-----')
        for example in examples[0:5]:
            print(example.sentence)


if __name__ == '__main__':
    main()
