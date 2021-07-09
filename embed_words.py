import argparse
import os
import pickle
from collections import namedtuple

from tqdm import tqdm

from nlp.parsing import EmbeddingExtractor
from read_data.raw_file import RawFileReader

Word = namedtuple('Word', ['form', 'lemma', 'pos', 'sentence', 'embedding'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    reader = RawFileReader(args.input)
    extractor = EmbeddingExtractor()
    all_words = []
    for doc in tqdm(extractor.nlp.pipe(reader, batch_size=50)):
        word_gen = (Word(token.text, token.lemma_, token.pos_, token.doc.text, embedding)
                    for token, embedding in extractor.get_word_embeddings(doc))
        all_words.extend(word_gen)

    with open(args.output, 'wb') as outfile:
        pickle.dump(all_words, outfile, protocol=5)


if __name__ == '__main__':
    main()
