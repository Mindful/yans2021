import argparse
import os
import pickle
from collections import defaultdict

from tqdm import tqdm

from nlp.parsing import EmbeddingExtractor, nlp
from read_data.raw_file import RawFileReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    results = defaultdict(list)

    reader = RawFileReader(args.input)
    extractor = EmbeddingExtractor()
    for doc in tqdm(nlp.pipe(reader, batch_size=50)):
        embeddings = extractor.get_word_embeddings(doc)
        for token_text, pos, embedding in embeddings:
            results[token_text].append((pos, embedding))

    with open(args.output, 'wb') as outfile:
        pickle.dump(dict(results), outfile, protocol=5)


if __name__ == '__main__':
    main()
