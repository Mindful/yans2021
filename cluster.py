import argparse
import numpy as np
from sklearn.cluster import DBSCAN

from data.db import DbConnection

# arase paper has distance as 0.1 and minpts as 10


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--output', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()
    db = DbConnection(args.run)

    all_words = list(db.read_words(include_sentences=False, use_tqdm=True))
    #word_ids = np.ndarray([w.id for w in all_words])
    embedding_array = np.stack([word.embedding for word in all_words])

    dbscan = DBSCAN(eps=0.1, min_samples=10, metric='cosine')
    dbscan.fit(embedding_array)

    print('debuggy')


if __name__ == '__main__':
    main()



