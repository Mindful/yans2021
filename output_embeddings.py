import argparse
import pickle
from typing import Dict
from collections import Counter

from cluster_words import WordCluster
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--wordlist', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    with open(args.wordlist, 'r') as wordfile:
        words = [x.strip() for x in wordfile]

    with open(args.data, 'rb') as pfile:
        data: Dict[str, WordCluster] = pickle.load(pfile)

    with open(args.output, 'w') as outfile:
        for word in words:
            cluster = data[word]
            cluster_sizes = Counter(cluster.cluster._labels)

            largest_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]
            centroid = list(cluster.cluster.cluster_centers_)[largest_cluster]

            output_list = [word] + list(centroid)
            outfile.write(' '.join(output_list) + '\n')


if __name__ == '__main__':
    main()
