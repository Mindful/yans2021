import argparse
import pickle
from typing import Dict
from collections import Counter
import numpy as np

from cluster import WordCluster
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

    vector_shape = None
    with open(args.output, 'w') as outfile:
        for word in words:
            if word in data:
                cluster = data[word]
                cluster_sizes = Counter(cluster.cluster.labels_)

                largest_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]
                centroid = list(cluster.cluster.cluster_centers_)[largest_cluster]

                vector_shape = centroid.shape
                output_list = [word] + list(centroid)
                outfile.write(' '.join(str(x) for x in output_list) + '\n')
            else:
                if vector_shape is not None:
                    print('Warning: blank vector for word', word)
                    output_list = [word] + list(np.zeros(vector_shape))
                    outfile.write(' '.join(str(x) for x in output_list) + '\n')
                else:
                    raise RuntimeError(f'No vector found for word {word} and unknown vector shape')


if __name__ == '__main__':
    main()
