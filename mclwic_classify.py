# https://github.com/SapienzaNLP/mcl-wic
import argparse
import json
from typing import Dict
from tqdm import tqdm

from helpers import extractor, get_or_create_cluster, ClusterConstructionError, classify_embedding


#     {
#         "id": "training.en-en.0",
#         "lemma": "play",
#         "pos": "NOUN",
#         "sentence1": "In that context of coordination and integration, Bolivia holds a key play in any process of infrastructure development.",
#         "sentence2": "A musical play on the same subject was also staged in Kathmandu for three days.",
#         "start1": "69",
#         "end1": "73",
#         "start2": "10",
#         "end2": "14"
#     },


def compute_row_label(row: Dict) -> Dict:
    embeddings_1 = extractor.get_word_embeddings(extractor.nlp(row['sentence1']))
    embeddings_2 = extractor.get_word_embeddings(extractor.nlp(row['sentence2']))

    token_1, target_embedding_1 = next((token, embedding) for token, embedding in embeddings_1
                                       if token.idx == row['start1'])
    token_2, target_embedding_2 = next((token, embedding) for token, embedding in embeddings_2
                                       if token.idx == row['start2'])

    try:
        clusters = get_or_create_cluster(token_1.lemma_, token_1.pos, 'r')
        label_1 = classify_embedding(target_embedding_1, clusters)
        label_2 = classify_embedding(target_embedding_2, clusters)
        return {
            'id': row['id'],
            'tag': 'T' if label_1 == label_2 else 'F'
        }

    except (ClusterConstructionError, ValueError):
        return {
            'id': row['id'],
            'tag': 'UNK'
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()
    output_name = args.input.split('.')[0] + '_hyp.json'

    with open(args.input, 'r') as f:
        data = json.load(f)

    labels = [compute_row_label(row) for row in tqdm(data, 'processing rows')]
    with open(output_name, 'w') as outfile:
        json.dump(labels, outfile)

    #TODO: handle UNK labels, etc.

    print('Had to default labels for', failed_rows, '/', len(labels), 'rows',
          f'{round(failed_rows/len(labels)) * 100}%')
    print('Wrote output to', outfile)



if __name__ == '__main__':
    main()
