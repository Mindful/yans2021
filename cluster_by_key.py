import argparse
import logging
from typing import List

from sklearn.cluster import KMeans, DBSCAN
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from data.db import DbConnection, Word, WordCluster, WriteBuffer


class ClusteringException(Exception):
    pass


def cluster_kmeans(key: str, words: List[Word]) -> WordCluster:
    cluster_count = 2
    embedding_array = np.stack([word.embedding for word in words])
    if embedding_array.shape[0] >= cluster_count:
        kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(embedding_array)
        return WordCluster(key, kmeans.cluster_centers_, kmeans.labels_)
    else:
        raise ClusteringException('Insufficient number of embeddings')


def cluster_dbscan(key: str, words: List[Word]) -> WordCluster:
    embedding_array = np.stack([word.embedding for word in words])
    dbscan = DBSCAN(eps=0.1, min_samples=10, metric='cosine').fit(embedding_array)

    label_groups = defaultdict(list)
    for word, label in zip(words, dbscan.labels_):
        if label != -1:
            label_groups[label].append(word.embedding)

    cluster_averages = {
        label: np.mean(embeddings, axis=0) for label, embeddings in label_groups.items()
    }
    cluster_averages = np.stack([embedding for _, embedding in sorted(cluster_averages.items(), key=lambda x: x[0])])

    return WordCluster(key, cluster_averages, dbscan.labels_)


cluster_func_dict = {
    'kmeans': cluster_kmeans,
    'dbscan': cluster_dbscan
}


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_by', required=False, default='lemma')
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--algo', required=False, choices=list(cluster_func_dict.keys()), default='dbscan')
    parser.add_argument('--key', required=False)

    args = parser.parse_args()
    cluster_function = cluster_func_dict[args.algo]

    db = DbConnection(args.run+'_words')

    if args.key:
        where_clause = f'where {args.group_by}=\'{args.key}\''
    else:
        where_clause = None

    groups = defaultdict(list)
    logger.info('Reading words...')
    for word in db.read_words(use_tqdm=True, include_sentences=False, where_clause=where_clause):
        key = getattr(word, args.group_by)
        groups[key].append(word)

    logger.info(f'Found {len(groups)} groups to cluster')

    write_db = DbConnection(args.run+'_clusters')
    write_buffer = WriteBuffer('cluster', write_db.save_clusters, buffer_size=5000)

    for key, words in tqdm(groups.items(), 'clustering word embeddings'):
        try:
            write_buffer.add(cluster_function(key, words))
        except ClusteringException as ex:
            logger.warning(f'Could not cluster key {key} due to {ex}')

    write_buffer.flush()


if __name__ == '__main__':
    main()
