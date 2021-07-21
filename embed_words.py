import argparse
import os
import logging
import queue

import spacy
from multiprocessing import Process, Queue

from tqdm import tqdm

from nlp.parsing import EmbeddingExtractor, reduction_function
from data.db import DbConnection, Word, WriteBuffer


def embedding_executor(q: Queue, process_num: int, bound: range, reduction: str, run: str):
    logging.info(f'Acquiring GPU {process_num}')
    spacy.require_gpu(process_num)
    extractor = EmbeddingExtractor(embedding_reducer=reduction_function[reduction])

    db = DbConnection(run)
    write_buffer = WriteBuffer('words', db.save_words)

    sentence_generator = ((text, ident) for ident, text in db.read_sentences(use_tqdm=False, bound=bound))
    for doc, ident in extractor.nlp.pipe(sentence_generator, batch_size=500, as_tuples=True):
        try:
            word_gen = (Word(token.text, token.lemma_, token.pos, ident, embedding)
                        for token, embedding in extractor.get_word_embeddings(doc))
            write_buffer.add_many(word_gen)
            q.put(1)
        except Exception as e:
            print(doc)
            raise e

    logging.info(f'Proc {process_num} done')


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--reduction', required=False, default='first')
    parser.add_argument('--gpus', required=False, type=int, default=8)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    db = DbConnection(args.run)
    logger.info('Counting sentences')
    total_sents = db.count_sentences()
    logger.info(f'Found {total_sents} sentences')

    devices = args.gpus
    batch_size = total_sents // devices
    process_metadata = [
        (idx, idx * batch_size, ((idx + 1) * batch_size) if idx != devices - 1 else total_sents + 1)
        for idx in range(devices)
    ]
    q = Queue()

    processes = [
        Process(target=embedding_executor, args=(q, num, range(start, stop), args.reduction, args.run))
        for num, start, stop in process_metadata
    ]

    for proc in processes:
        proc.start()

    pbar = tqdm(total=total_sents, desc='processing sentences')
    counter = 0
    while counter < total_sents:
        try:
            result = q.get(timeout=600)
            pbar.update(1)
            counter += 1
        except queue.Empty:
            logging.error('Empty queue')
            break

    pbar.close()
    logging.info('Cleaning up')

    for proc in processes:
        proc.join()

    logging.info('Done')


if __name__ == '__main__':
    main()
