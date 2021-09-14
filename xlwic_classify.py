# https://pilehvar.github.io/xlwic/data/README.txt
import argparse
import csv
import logging
from collections import namedtuple
from typing import Optional

from data.db import WordCluster
from helpers import extractor, classify_embedding, ClusterConstructionError, db
from spacy.parts_of_speech import NOUN, VERB, PROPN
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


def find_word_in_embeddings(embeddings, token_start, lemma):
    if sum(1 for token, _ in embeddings if token.lemma_ == lemma) == 1:
        return next(embedding for token, embedding in embeddings if token.lemma_ == lemma)
    else:
        return next(embedding for token, embedding in embeddings if token.idx == token_start)


def compute_row_label(row: RowData, clusters: Optional[WordCluster]) -> int:
    if clusters is None:
        logging.warning(f'No cluster for {row.lemma}/{row.pos}')
        return -1

    embeddings_1 = extractor.get_word_embeddings(extractor.nlp(row.example_1), include_extra_pos={PROPN})
    embeddings_2 = extractor.get_word_embeddings(extractor.nlp(row.example_2), include_extra_pos={PROPN})

    try:
        target_embedding_1 = find_word_in_embeddings(embeddings_1, int(row.start_1), row.lemma)
        target_embedding_2 = find_word_in_embeddings(embeddings_2, int(row.start_2), row.lemma)
    except StopIteration:
        logging.warning(f'Unable to find token for {row.lemma}/{row.pos}')
        return -1

    label_1 = classify_embedding(target_embedding_1, clusters)
    label_2 = classify_embedding(target_embedding_2, clusters)

    return 1 if label_1 == label_2 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()
    output_name = args.input.split('.')[0] + '_hyp.csv'
    label_file_name = args.input.split('.')[0] + '_labels.csv'

    with open(args.input, 'r') as f:
        input_rows = list(csv.reader(f, delimiter='\t'))

    with extractor.nlp.select_pipes(disable=['transformer']):
        rows = [
            RowData(extractor.nlp(row[0])[0].lemma_, row[0], pos_map[row[1]], *row[2:])
            for row in tqdm(input_rows, 'processing rows')
        ]

    lemma_pos_set = {(row.lemma, row.pos) for row in rows}
    clusters_by_lemma = {
        (lemma, pos): db.get_cluster(lemma, pos, 'r', include_words=False) for lemma, pos
        in tqdm(sorted(lemma_pos_set, key=lambda x: x[0]), 'fetching or building clusters')
    }

    labels = [compute_row_label(row, clusters_by_lemma[(row.lemma, row.pos)]) for row in tqdm(rows, 'labeling')]
    failed_rows = sum(1 for x in labels if x == -1)
    labels = [label if label != -1 else 0 for label in labels]

    with open(output_name, 'w') as outfile:
        outfile.writelines(str(x)+'\n' for x in labels)

    print('Had to default labels for', failed_rows, '/', len(labels), 'rows',
          f'{round(failed_rows/len(labels), 2) * 100}%')
    print('Wrote output to', output_name)

    if rows[0].label is not None:
        with open(label_file_name, 'w') as label_outfile:
            label_outfile.writelines(str(x.label) + '\n' for x in rows)

        print('Wrote labels to', label_file_name)


if __name__ == '__main__':
    main()
