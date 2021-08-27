# https://pilehvar.github.io/xlwic/data/README.txt
import argparse
import csv
from collections import namedtuple
from typing import Optional

from data.db import WordCluster
from helpers import extractor, get_or_create_cluster, classify_embedding, ClusterConstructionError
from spacy.parts_of_speech import NOUN, VERB
from tqdm import tqdm

pos_map = {
    'N': NOUN,
    'V': VERB
}


# The files follow a tab-separated format:
# target_word <tab> PoS <tab> start-char-index_1 <tab> end-char-index_1 <tab> start-char-index_2 <tab>
# end-char-index_2 <tab> example_1 <tab> example_2 <tab> label
RowData = namedtuple('RowData', ['lemma', 'target_word', 'pos', 'start_1', 'end_1', 'start_2', 'end_2',
                                 'example_1', 'example_2', 'label'])


def get_cluster(lemma: str, pos: int) -> Optional[WordCluster]:
    try:
        return get_or_create_cluster(lemma, pos, 'r')
    except (ClusterConstructionError, ValueError):
        return None


def compute_row_label(row: RowData, clusters: Optional[WordCluster]) -> int:
    if clusters is None:
        return -1

    embeddings_1 = extractor.get_word_embeddings(extractor.nlp(row.example_1))
    embeddings_2 = extractor.get_word_embeddings(extractor.nlp(row.example_2))

    target_embedding_1 = next(embedding for token, embedding in embeddings_1 if token.idx == row.start_1)
    target_embedding_2 = next(embedding for token, embedding in embeddings_2 if token.idx == row.start_2)

    label_1 = classify_embedding(target_embedding_1, clusters)
    label_2 = classify_embedding(target_embedding_2, clusters)

    return 1 if label_1 == label_2 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()
    output_name = args.input.split('.')[0] + '_hyp.csv'

    with open(args.input, 'r') as f:
        input_rows = list(csv.reader(f, delimiter='\t'))

    with extractor.nlp.select_pipes(disable=['transformer']):
        rows = [
            RowData(extractor.nlp(row[0])[0].lemma_, row[0], pos_map[row[1]], *row[2:])
            for row in tqdm(input_rows, 'processing rows')
        ]

    lemma_pos_set = {(row.lemma, row.pos) for row in rows}
    clusters_by_lemma = {
        (lemma, pos): get_cluster(lemma, pos) for lemma, pos
        in tqdm(sorted(lemma_pos_set, key=lambda x: x[0]), 'fetching or building clusters')
    }

    labels = [compute_row_label(row, clusters_by_lemma[(row.lemma, row.pos)]) for row in rows]
    failed_rows = sum(1 for x in labels if x == -1)
    labels = [label if label != -1 else 0 for label in labels]

    with open(output_name, 'w') as outfile:
        outfile.writelines(str(x) for x in labels)

    print('Had to default labels for', failed_rows, '/', len(labels), 'rows',
          f'{round(failed_rows/len(labels)) * 100}%')
    print('Wrote output to', outfile)


if __name__ == '__main__':
    main()
