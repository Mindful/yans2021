import argparse
import logging
from typing import List, Tuple

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from data.db import DbConnection, Word, WordCluster, WriteBuffer


class ClusteringException(Exception):
    pass


def compute_display_embeddings(word_data: List[Tuple[int, Word]]) -> List[Tuple[int, np.ndarray]]:
    embedding_array = np.stack([word.embedding for _, word in word_data])
    pca = PCA(n_components=3)
    display_embedding_array = pca.fit_transform(embedding_array)

    return list(zip((word_id for word_id, _ in word_data), display_embedding_array))


def cluster_kmeans(key: str, words: List[Word]) -> WordCluster:
    cluster_count = 3
    embedding_array = np.stack([word.embedding for word in words])
    if embedding_array.shape[0] >= cluster_count:
        kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(embedding_array)
        return WordCluster(key, kmeans.cluster_centers_, kmeans.labels_)
    else:
        raise ClusteringException('Insufficient number of embeddings')


def cluster_dbscan(key: str, words: List[Word]) -> WordCluster:
    embedding_array = np.stack([word.embedding for word in words])
    dbscan = DBSCAN(eps=0.25, min_samples=100, metric='euclidean').fit(embedding_array)

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
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--algo', required=False, choices=list(cluster_func_dict.keys()), default='dbscan')
    parser.add_argument('--key', required=False)

    args = parser.parse_args()
    cluster_function = cluster_func_dict[args.algo]

    db = DbConnection(args.run)

    if args.key:
        keys = args.key.split(',')
        in_group = ', '.join([f"'{x}'" for x in keys])
        where_clause = f'where lemma in ({in_group})'
    else:
        where_clause = None

    groups = defaultdict(list)
    logger.info('Reading words...')
    for word_id, word in db.read_words(use_tqdm=True, include_sentences=False, where_clause=where_clause):
        key = (word.lemma, word.pos)
        groups[key].append((word_id, word))

    logger.info(f'Found {len(groups)} groups to cluster')

    write_db = DbConnection(args.run+'_clusters')

    for key, word_data in tqdm(groups.items(), 'clustering word embeddings'):
        words = [word for word_id, word in word_data]
        try:
            write_db.save_clusters([cluster_function(key, words)])
            display_embeddings = compute_display_embeddings(word_data)

        except ClusteringException as ex:
            logger.warning(f'Could not cluster key {key} due to {ex}')

    write_buffer.flush()


if __name__ == '__main__':
    main()
