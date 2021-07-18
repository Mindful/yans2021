import argparse
import os
import logging

from tqdm import tqdm
from nlp.parsing import EmbeddingExtractor, reduction_function
from data.input import RawFileReader
from data.db import DbConnection, Word

logging.getLogger().setLevel(logging.INFO)


def main():
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', required=False, default='default')
    parser.add_argument('--reduction', required=False, default='first')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    reader = RawFileReader(args.input, max_line_length=1000)
    extractor = EmbeddingExtractor(embedding_reducer=reduction_function[args.reduction])
    db = DbConnection(args.run)
    db.connect_for_saving()

    for doc in tqdm(extractor.nlp.pipe(reader, batch_size=50)):
        try:
            word_gen = (Word(token.text, token.lemma_, token.pos, token.doc.text, embedding)
                        for token, embedding in extractor.get_word_embeddings(doc))
            db.add_words(word_gen)
        except Exception as e:
            logger.error('Failed processing doc')
            print(doc)
            raise e

    db.done()


if __name__ == '__main__':
    main()
