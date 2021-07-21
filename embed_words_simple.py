import argparse
import os
import logging

from nlp.parsing import EmbeddingExtractor, reduction_function
from data.db import DbConnection, Word, WriteBuffer


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--reduction', required=False, default='first')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    db = DbConnection(args.run)
    db_write = DbConnection(args.run)
    write_buffer = WriteBuffer('word', db_write.save_words, buffer_size=1)
    extractor = EmbeddingExtractor(embedding_reducer=reduction_function[args.reduction])

    sentence_generator = ((text, ident) for ident, text in db.read_sentences(use_tqdm=True))
    for doc, ident in extractor.nlp.pipe(sentence_generator, batch_size=50, as_tuples=True):
        try:
            word_gen = (Word(token.text, token.lemma_, token.pos, ident, embedding)
                        for token, embedding in extractor.get_word_embeddings(doc))
            # db.cur.close()
            # db.con.close()
            write_buffer.add_many(word_gen)
        except Exception as e:
            logger.error('Failed processing doc')
            print(doc)
            raise e

    write_buffer.flush()


if __name__ == '__main__':
    main()