import argparse
import pickle
from sklearn.cluster import KMeans
from collections import defaultdict, namedtuple
import numpy as np
from tqdm import tqdm

from data.db import DbConnection

WordCluster = namedtuple('WordCluster', ['cluster', 'words'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group_by', required=False, default='lemma')
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()

    db = DbConnection(args.run)

    groups = defaultdict(list)

    for word in db.read_words(use_tqdm=True, include_sentences=False):
        key = getattr(word, args.group_by)
        groups[key].append(word)

    result_groups = dict()

    for key, words in tqdm(groups.items(), 'clustering word embeddings'):
        cluster_count = 2
        embedding_array = np.stack([word.embedding for word in words])
        if embedding_array.shape[0] >= cluster_count:
            kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(embedding_array)
            result_groups[key] = WordCluster(kmeans, words)
        else:
            print('Skipped clustering for key', key, 'because it had an insufficient number of embeddings')

    with open(args.output, 'wb') as outfile:
        pickle.dump(result_groups, outfile, protocol=5)

    db.done()


if __name__ == '__main__':
    main()
